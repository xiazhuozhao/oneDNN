/*******************************************************************************
* Copyright 2025 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "gpu/intel/jit/dsl/dsl.hpp"
#include "gpu/intel/jit/gemm/include/gemmstone/strategy.hpp"
#include "gpu/intel/jit/gemm/ir/kernel_desc.hpp"
#include "gpu/intel/jit/utils/trace.hpp"
#include "gpu/intel/utils.hpp"

namespace gemmstone {

using namespace ir::dsl;

const ir::pvar_t &m_var = ir::pvars::m;
const ir::pvar_t &n_var = ir::pvars::n;
const ir::pvar_t &k_var = ir::pvars::k;

struct kloop_iterator_t {

    virtual const global_tensor_t &A_prefetch() const = 0;
    virtual const global_tensor_t &A_load() const = 0;
    virtual const global_tensor_t &B_prefetch() const = 0;
    virtual const global_tensor_t &B_load() const = 0;
    virtual const global_tensor_t &C_store() const = 0;

    virtual void inc_prefetch_A(int k_block) = 0;
    virtual void inc_prefetch_B(int k_block) = 0;
    virtual void inc_load_A(int k_block) = 0;
    virtual void inc_load_B(int k_block) = 0;
    virtual void inc_kloop(int k_block) = 0;

    virtual ir::expr_t update_C() const = 0;

    // Returns whether the given increment is in bounds
    virtual ir::expr_t is_inbounds(int increment) const = 0;
};

const char *to_str(AccessType t) {
    switch (t) {
        case AccessType::Scattered: return "Scattered";
        case AccessType::ChannelScattered: return "ChannelScattered";
        case AccessType::Block: return "Block";
        case AccessType::PseudoBlock: return "PseudoBlock";
        case AccessType::Block2D: return "Block2D";
        case AccessType::Block2DTranspose: return "Block2DTranspose";
        case AccessType::Block2DVNNI: return "Block2DVNNI";
        case AccessType::CacheLine: return "CacheLine";
        default: stub(); return "(unknown)";
    }
}

transform_t get_plan(const MatrixAddressing &matrix_type,
        const MatrixAddressingStrategy &matrix_strategy,
        std::array<ir::pvar_t, 2> dims, bool is_prefetch = false) {
    switch (matrix_strategy.accessType) {
        case AccessType::Scattered:
            // TODO: Remove workaround unimplemented scattered->vnni support.
            if (is_prefetch)
                return transform_t(transform_t::kind_t::none, 0,
                        matrix_strategy.cachingR, dims);

            return transform_t(transform_t::kind_t::transpose_vnni,
                    matrix_strategy.tileR, matrix_strategy.cachingR, dims);

        case AccessType::ChannelScattered: stub(); return {};
        case AccessType::Block2DTranspose:
            return transform_t(transform_t::kind_t::transpose_vnni,
                    matrix_strategy.tileR, matrix_strategy.cachingR, dims);
        case AccessType::Block:
        case AccessType::PseudoBlock:
            return transform_t(transform_t::kind_t::block,
                    matrix_strategy.tileR, matrix_strategy.cachingR, dims);
        case AccessType::Block2D: {
            return transform_t(transform_t::kind_t::block,
                    matrix_strategy.tileR, matrix_strategy.cachingR, dims);
        };
        case AccessType::Block2DVNNI: {
            return transform_t(transform_t::kind_t::vnni, matrix_strategy.tileR,
                    matrix_strategy.cachingR, dims);
        }
        default: stub(); return {};
    }
};
ir::pvar_map_t<ir::expr_t> get_strides(
        MatrixLayout layout, std::array<ir::pvar_t, 2> pvars, ir::expr_t ld) {
    switch (layout) {
        case MatrixLayout::N: return {{pvars[0], 1}, {pvars[1], ld}};
        case MatrixLayout::T: return {{pvars[0], ld}, {pvars[1], 1}};
        default: stub(); return {};
    };
}

const ir::v2::layout_desc_t &gemm_var_desc() {
    static const ir::v2::layout_desc_t desc {
            {{m_var, 'm'}, {n_var, 'n'}, {k_var, 'k'}}};
    return desc;
};

struct tensor_config_t {
    tensor_config_t(const global_tensor_t &g, transform_t t, int copies)
        : transform(t) {
        tile = g.tile;
        layout = t.get_layout(g.tile, g.type, gemm_var_desc());

        copy_layout = layout;
        copy_layout.add_block(k_var, copies, layout.elems());
        copy_tile = copy_layout.int_dim_sizes();
    }

    ir::v2::layout_t layout;
    ir::tile_t tile;

    ir::tile_t copy_tile;
    ir::v2::layout_t copy_layout;

    transform_t transform;
};

// Basic iterator with no iteration over m and n.
struct basic_iterator_t : kloop_iterator_t {
    basic_iterator_t(ir::expr_t m, ir::expr_t n, ir::expr_t k, int m_blk,
            int n_blk, int k_blk, ir::expr_t A_buffer, ir::expr_t A_offset,
            ir::type_t A_type, ir::pvar_map_t<ir::expr_t> A_strides,
            int A_prefetch_k_blk, int A_load_k_blk, ir::expr_t B_buffer,
            ir::expr_t B_offset, ir::type_t B_type,
            ir::pvar_map_t<ir::expr_t> B_strides, int B_prefetch_k_blk,
            int B_load_k_blk, ir::expr_t C_buffer, ir::expr_t C_offset,
            ir::type_t C_type, ir::pvar_map_t<ir::expr_t> C_strides,
            ir::pvar_t subgroup_dim, const int subgroup_size,
            const std::array<ir::expr_t, 3> &group_ids,
            const std::array<ir::expr_t, 3> &local_ids,
            const std::array<ir::expr_t, 3> &local_sizes)
        : m_idx_ {let("m_idx",
                (group_ids[0] * local_sizes[0] + local_ids[0])
                        * (m_blk
                                / (subgroup_dim == m_var ? subgroup_size : 1)))}
        , m_(m)
        , n_idx_ {let("n_idx",
                  (group_ids[1] * local_sizes[1] + local_ids[1])
                          * (n_blk
                                  / (subgroup_dim == n_var ? subgroup_size
                                                           : 1)))}
        , n_(n)
        , k_idx_ {def(k.type(), "k_idx", 0)}
        , k_prefetch_idx_A_ {k_idx_}
        , k_prefetch_idx_B_ {k_idx_}
        , k_load_idx_A_ {k_idx_}
        , k_load_idx_B_ {k_idx_}
        , k_ {k}
        , A_prefetch_ {A_buffer, A_type, A_offset,
                  {{m_var, m_idx_}, {k_var, k_prefetch_idx_A_}}, A_strides,
                  {{m_var, m}, {k_var, k}},
                  {{m_var, m_blk}, {k_var, A_prefetch_k_blk}}}
        , A_load_ {A_buffer, A_type, A_offset,
                  {{m_var, m_idx_}, {k_var, k_load_idx_A_}}, A_strides,
                  {{m_var, m}, {k_var, k}},
                  {{m_var, m_blk}, {k_var, A_load_k_blk}}}
        , B_prefetch_ {B_buffer, B_type, B_offset,
                  {{k_var, k_prefetch_idx_B_}, {n_var, n_idx_}}, B_strides,
                  {{k_var, k}, {n_var, n}},
                  {{k_var, B_prefetch_k_blk}, {n_var, n_blk}}}
        , B_load_ {B_buffer, B_type, B_offset,
                  {{k_var, k_load_idx_B_}, {n_var, n_idx_}}, B_strides,
                  {{k_var, k}, {n_var, n}},
                  {{k_var, B_load_k_blk}, {n_var, n_blk}}}
        , C_store_ {C_buffer, C_type, C_offset,
                  {{m_var, m_idx_}, {n_var, n_idx_}}, C_strides,
                  {{m_var, m}, {n_var, n}}, {{m_var, m_blk}, {n_var, n_blk}}}

    {
        assume(m_idx_ % m_blk == 0);
        assume(n_idx_ % n_blk == 0);

        assume(m_idx_ >= 0);
        assume(n_idx_ >= 0);
        assume(k_idx_ >= 0);
    }

    const global_tensor_t &A_prefetch() const override { return A_prefetch_; }
    const global_tensor_t &A_load() const override { return A_load_; }
    const global_tensor_t &B_prefetch() const override { return B_prefetch_; }
    const global_tensor_t &B_load() const override { return B_load_; }
    const global_tensor_t &C_store() const override { return C_store_; }

    void inc_prefetch_A(int k_block) override {
        k_prefetch_idx_A_ = ir::simplify(k_prefetch_idx_A_ + k_block);
        A_prefetch_.idxs[k_var] = k_prefetch_idx_A_;
    }

    void inc_prefetch_B(int k_block) override {
        k_prefetch_idx_B_ = ir::simplify(k_prefetch_idx_B_ + k_block);
        B_prefetch_.idxs[k_var] = k_prefetch_idx_B_;
    }

    void inc_load_A(int k_block) override {
        k_load_idx_A_ = ir::simplify(k_load_idx_A_ + k_block);
        A_load_.idxs[k_var] = k_load_idx_A_;
    }

    void inc_load_B(int k_block) override {
        k_load_idx_B_ = ir::simplify(k_load_idx_B_ + k_block);
        B_load_.idxs[k_var] = k_load_idx_B_;
    }

    void inc_kloop(int k_block) override {
        // Prefetch/load computation is relative to k_idx
        inc_prefetch_A(-k_block);
        inc_prefetch_B(-k_block);
        inc_load_A(-k_block);
        inc_load_B(-k_block);

        assign(k_idx_, k_idx_ + k_block);
    }

    ir::expr_t update_C() const override { return false; }

    ir::expr_t is_inbounds(int increment) const override {
        return (m_idx_ < m_) & (n_idx_ < n_) & (k_idx_ < k_ - increment);
    }

private:
    static ir::expr_t offset(const ir::pvar_map_t<ir::expr_t> &idxs,
            const ir::pvar_map_t<ir::expr_t> &strides,
            const ir::icoord_t &coord) {
        ir::expr_t ret = 0;
        for (auto &c : coord) {
            ret += (idxs[c] + coord[c]) * strides[c];
        }
        return ir::simplify(ret);
    }

    ir::expr_t m_idx_;
    ir::expr_t m_;
    ir::expr_t n_idx_;
    ir::expr_t n_;
    ir::expr_t k_idx_;
    ir::expr_t k_prefetch_idx_A_;
    ir::expr_t k_prefetch_idx_B_;
    ir::expr_t k_load_idx_A_;
    ir::expr_t k_load_idx_B_;
    ir::expr_t k_;

    global_tensor_t A_prefetch_;
    global_tensor_t A_load_;
    global_tensor_t B_prefetch_;
    global_tensor_t B_load_;
    global_tensor_t C_store_;
};

struct gemm_ir {
    gemm_ir(const gemm_ir_desc_t &desc)
        : problem(desc.problem), strategy(desc.strategy) {}

    ir::stmt_t build(ir::kernel_iface_t iface, ir::ir_context_t &ctx) {
        if (strategy.kParallel || strategy.kParallelLocal) {
            gpu_warning() << "kParallel support is unimplemented";
            return {};
        }
        if (strategy.persistentLoop()) {
            gpu_warning() << "persistentLoop support is unimplemented";
            return {};
        }
        if (strategy.slmA || strategy.slmB) {
            gpu_warning()
                    << "slm copy support is unimplemented, disabling slm copy";
        }

        if (problem.Ta != problem.Ta_ext || problem.Tb != problem.Tb_ext
                || problem.Tc != problem.Tc_ext) {
            gpu_warning() << "Type conversion support is unimplemented";
            return {};
        }

        declare_kernel(iface, ctx);

        const auto m = arg("m");
        const auto n = arg("n");
        const auto k = arg("k");

        auto m_blk = strategy.unroll[LoopM];
        auto n_blk = strategy.unroll[LoopN];
        auto k_blk = strategy.unroll[LoopK];

        std::array<ir::pvar_t, 2> A_vars = {m_var, k_var};
        std::array<ir::pvar_t, 2> B_vars = {k_var, n_var};
        std::array<ir::pvar_t, 2> C_vars = {m_var, n_var};

        auto A_prefetch_plan
                = get_plan(problem.A, strategy.A_prefetch, A_vars, true);
        auto A_load_plan = get_plan(problem.A, strategy.A, A_vars);

        auto B_prefetch_plan
                = get_plan(problem.B, strategy.B_prefetch, B_vars, true);
        auto B_load_plan = get_plan(problem.B, strategy.B, B_vars);

        ir::tile_t C_dims {{{m_var, m_blk}, {n_var, n_blk}}};
        auto C_store_plan = get_plan(problem.C, strategy.C, C_vars);

        tensor_t C = def(C_store_plan.get_layout(
                                 C_dims, into_ir(problem.Tc), gemm_var_desc()),
                "C_blk", 0);

        basic_iterator_t kloop_it(m, n, k, m_blk, n_blk, k_blk, arg("A"),
                arg("offset_A"), into_ir(problem.Ta_ext),
                get_strides(problem.A.layout, A_vars, arg("lda")),
                strategy.ka_prefetch, strategy.ka_load, arg("B"),
                arg("offset_B"), into_ir(problem.Tb_ext),
                get_strides(problem.B.layout, B_vars, arg("ldb")),
                strategy.kb_prefetch, strategy.kb_load, arg("C"),
                arg("offset_C"), into_ir(problem.Tc_ext),
                get_strides(problem.C.layout, C_vars, arg("ldc")),
                C.layout.blocks()[0].dim, strategy.subgroupSize, group_ids(),
                local_ids(), local_sizes());

        auto store_C
                = [&]() { store(kloop_it.C_store(), C, C_store_plan, {}); };

        tensor_config_t A_load(
                kloop_it.A_load(), A_load_plan, strategy.A_copies);
        tensor_config_t B_load(
                kloop_it.B_load(), B_load_plan, strategy.B_copies);

        auto prefetchA = strategy.prefetchA ? dnnl::impl::utils::rnd_dn(
                                 strategy.prefetchA, strategy.ka_prefetch)
                                            : 0;
        if (prefetchA != strategy.prefetchA)
            gpu_warning() << "Unimplemented partial B tile prefetch, modifying "
                             "prefetch distance "
                          << strategy.prefetchA << " -> " << prefetchA;
        auto prefetchB = strategy.prefetchB ? dnnl::impl::utils::rnd_dn(
                                 strategy.prefetchB, strategy.kb_prefetch)
                                            : 0;
        if (prefetchB != strategy.prefetchB)
            gpu_warning() << "Unimplemented partial B tile prefetch, modifying "
                             "prefetch distance "
                          << strategy.prefetchB << " -> " << prefetchB;

        k_loop_config_t k_loop_main {k_blk, prefetchA, prefetchB, kloop_it,
                A_load, B_load, A_prefetch_plan, B_prefetch_plan, C};

        gpu_assert(k_loop_main.warmup_load_A() % kloop_it.A_load().tile[k_var]
                == 0);
        gpu_assert(k_loop_main.warmup_load_B() % kloop_it.B_load().tile[k_var]
                == 0);

        tensor_config_t A_load_short(kloop_it.A_load(), A_load_plan, 1);
        tensor_config_t B_load_short(kloop_it.B_load(), B_load_plan, 1);

        k_loop_config_t k_loop_short {
                (int)lcm(A_load_short.tile[k_var], B_load_short.tile[k_var]), 0,
                0, kloop_it, A_load_short, B_load_short, A_prefetch_plan,
                B_prefetch_plan, C};
        gpu_assert(k_loop_short.warmup_k() == 0);

        if (problem.A.alignment) {
            assume(arg("lda") % (problem.A.alignment / problem.Ta_ext) == 0);
        }
        if (problem.B.alignment) {
            assume(arg("ldb") % (problem.B.alignment / problem.Tb_ext) == 0);
        }
        if (problem.C.alignment) {
            assume(arg("ldc") % (problem.C.alignment / problem.Tc_ext) == 0);
        }

        // TODO: This needs moved inside the following if statements
        assume(arg("lda") >= (64 / problem.Ta_ext));
        assume(arg("ldb") >= (64 / problem.Ta_ext));
        assume(arg("ldc") >= (64 / problem.Ta_ext));
        if_(kloop_it.is_inbounds(0), [&]() {
            if_(
                    k >= k_loop_main.warmup_k(),
                    [&]() { build_k_loop(k_loop_main); },
                    [&]() { build_k_loop(k_loop_short); });
            store_C();
        });

        return end_kernel();
    }

    struct k_loop_config_t {
        int k_blk;
        int warmup_prefetch_A; // Offset to A prefetch
        int warmup_prefetch_B; // Offset to B prefetch
        basic_iterator_t kloop_it;
        tensor_config_t A_load;
        tensor_config_t B_load;
        transform_t A_prefetch_plan;
        transform_t B_prefetch_plan;
        tensor_t C;

        int warmup_load_A() const {
            return A_load.copy_tile[k_var] - A_load.tile[k_var];
        }
        int warmup_load_B() const {
            return B_load.copy_tile[k_var] - B_load.tile[k_var];
        }
        int warmup_k() const {
            return std::max({warmup_load_A(), warmup_load_B(),
                    warmup_prefetch_A, warmup_prefetch_B});
        }
    };

    void build_k_loop(const k_loop_config_t &cfg) {
        auto k_blk = cfg.k_blk;
        auto kloop_it = cfg.kloop_it;
        auto &C = cfg.C;

        tensor_t A = def(cfg.A_load.copy_layout, "A_blk");
        tensor_t B = def(cfg.B_load.copy_layout, "B_blk");

        int mma_k_blk
                = std::min(cfg.A_load.tile[k_var], cfg.B_load.tile[k_var]);

        std::cout << "A: " << A.str();
        std::cout << "B: " << B.str();

        auto pipeline_idx = [&](int loop_idx, int warmup_size, int period) {
            return (loop_idx + warmup_size) % period;
        };

        int prefetch_A_blk
                = cfg.warmup_prefetch_A ? kloop_it.A_prefetch().tile[k_var] : 0;
        auto prefetch_A = [&](int k_unroll_idx) {
            if (cfg.warmup_prefetch_A == 0) return;
            int idx = pipeline_idx(
                    k_unroll_idx, cfg.warmup_prefetch_A, prefetch_A_blk);
            if (idx % prefetch_A_blk != 0) return;
            prefetch(kloop_it.A_prefetch(), cfg.A_prefetch_plan, {{k_var, 0}});
            kloop_it.inc_prefetch_A(prefetch_A_blk);
        };

        int load_A_blk = cfg.A_load.tile[k_var];
        auto load_A = [&](int k_unroll_idx) {
            int idx = pipeline_idx(k_unroll_idx, cfg.warmup_load_A(),
                    cfg.A_load.copy_tile[k_var]);
            if (idx % load_A_blk != 0) return;
            load(A.sub_tensor(cfg.A_load.layout, {{k_var, idx}}),
                    kloop_it.A_load(), cfg.A_load.transform, {{k_var, 0}});
            kloop_it.inc_load_A(load_A_blk);
        };

        int prefetch_B_blk
                = cfg.warmup_prefetch_B ? kloop_it.B_prefetch().tile[k_var] : 0;
        auto prefetch_B = [&](int k_unroll_idx) {
            if (cfg.warmup_prefetch_B == 0) return;
            int idx = pipeline_idx(
                    k_unroll_idx, cfg.warmup_prefetch_B, prefetch_B_blk);
            if (idx % prefetch_B_blk != 0) return;
            prefetch(kloop_it.B_prefetch(), cfg.B_prefetch_plan, {{k_var, 0}});
            kloop_it.inc_prefetch_B(prefetch_B_blk);
        };

        int load_B_blk = cfg.B_load.tile[k_var];
        auto load_B = [&](int k_unroll_idx) {
            int idx = pipeline_idx(k_unroll_idx, cfg.warmup_load_B(),
                    cfg.B_load.copy_tile[k_var]);
            if (idx % load_B_blk != 0) return;
            load(B.sub_tensor(cfg.B_load.layout, {{k_var, idx}}),
                    kloop_it.B_load(), cfg.B_load.transform, {{k_var, 0}});
            kloop_it.inc_load_B(load_B_blk);
        };

        int k_unroll_blk = [&]() {
            int ret = k_blk;
            for (auto v :
                    {prefetch_A_blk, load_A_blk, prefetch_B_blk, load_B_blk}) {
                ret = gcd(ret, v);
            }
            return ret;
        }();

        auto k_body = [&](int k_offset, bool do_prefetch_A, bool do_prefetch_B,
                              bool do_load_A, bool do_load_B, bool do_mma) {
            if (do_prefetch_A) { prefetch_A(k_offset); }

            if (do_prefetch_B) { prefetch_B(k_offset); }

            if (do_load_A) { load_A(k_offset); }

            if (do_load_B) { load_B(k_offset); }

            if (do_mma) {
                if (k_offset % mma_k_blk == 0) {
                    ir::tile_t tile = C.layout.int_dim_sizes();
                    tile[k_var] = mma_k_blk;
                    mma(C, A, B, tile, {{k_var, k_offset}}, strategy.systolic);
                }
            }
        };

        // Pipeline controls
        auto warmup = cfg.warmup_k();

        for (int k_unroll_idx = -warmup; k_unroll_idx < 0;
                k_unroll_idx += k_unroll_blk) {
            bool prefetch_A = k_unroll_idx + cfg.warmup_prefetch_A >= 0;
            bool prefetch_B = k_unroll_idx + cfg.warmup_prefetch_B >= 0;
            bool load_A = k_unroll_idx + cfg.warmup_load_A() >= 0;
            bool load_B = k_unroll_idx + cfg.warmup_load_B() >= 0;
            bool do_mma = false;
            k_body(k_unroll_idx, prefetch_A, prefetch_B, load_A, load_B,
                    do_mma);
        }

        while_(kloop_it.is_inbounds(warmup), [&]() {
            for (int k_unroll_idx = 0; k_unroll_idx < k_blk;
                    k_unroll_idx += k_unroll_blk) {
                k_body(k_unroll_idx, cfg.warmup_prefetch_A,
                        cfg.warmup_prefetch_B, true, true, true);
            }
            kloop_it.inc_kloop(k_blk);
        });

        auto tail_end = dnnl::impl::utils::rnd_up(warmup, k_blk);
        for (int k_unroll_idx = 0; k_unroll_idx < tail_end;
                k_unroll_idx += k_unroll_blk) {
            bool prefetch_A = k_unroll_idx + cfg.warmup_prefetch_A < tail_end;
            bool prefetch_B = k_unroll_idx + cfg.warmup_prefetch_B < tail_end;
            bool load_A = k_unroll_idx + cfg.warmup_load_A() < tail_end;
            bool load_B = k_unroll_idx + cfg.warmup_load_B() < tail_end;
            k_body(k_unroll_idx, prefetch_A, prefetch_B, load_A, load_B, true);
        }
    }

    const GEMMProblem &problem;
    const GEMMStrategy &strategy;
};

ir::stmt_t build_ir(const gemm_ir_desc_t &desc, ir::constraint_set_t cset) {
    ir::ir_context_t ctx(desc.compile_ctx().exec_config(), cset);

    ir::trace_start();
    auto stmt = gemm_ir(desc).build(desc.kernel_iface(), ctx);
    ir::trace_pass("build gemm_ir", stmt, ctx);

    stmt = ir::simplify(stmt, ctx);
    stmt = ir::inject_send(stmt, ctx);

    // TODO: This should be unnecessary as it could happen at codegen
    stmt = ir::fixup_if_conditions(stmt, ctx);
    stmt = ir::eliminate_common_subexprs(
            stmt, ctx, desc.strategy.GRFs * grf_size);
    return stmt;
}

} // namespace gemmstone
