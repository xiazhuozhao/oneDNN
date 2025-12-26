#ifndef CPU_RV64_JIT_RVV_NCHW_POOL_KERNEL_HPP
#define CPU_RV64_JIT_RVV_NCHW_POOL_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "cpu/cpu_pooling_pd.hpp"
#include "xbyak_riscv/xbyak_riscv.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct jit_pool_conf_t {
    int ndims;
    int mb, c;
    int id, ih, iw;
    int od, oh, ow;
    int kd, kh, kw;
    int stride_d, stride_h, stride_w;
    int pad_f, pad_t, pad_l;
    bool is_max;
    data_type_t dt;
};

struct jit_pool_call_t {
    const void *src;
    void *dst;
    size_t id_start;
    size_t ih_start;
    size_t src_d_stride;
    size_t src_h_stride;
    const void *indices;
};

struct jit_rvv_nchw_pool_kernel : public Xbyak_riscv::CodeGenerator {
    jit_rvv_nchw_pool_kernel(const jit_pool_conf_t &jpp)
        : Xbyak_riscv::CodeGenerator(), jpp_(jpp) {}

    void operator()(const jit_pool_call_t *args) const {
        auto func = getCode<void (*)(const jit_pool_call_t *)>();
        func(args);
    }

    status_t create_kernel() {
        generate();
        return getSize() > 0 ? status::success : status::runtime_error;
    }

private:
    jit_pool_conf_t jpp_;

    const Xbyak_riscv::Reg &reg_src_base = Xbyak_riscv::s1;
    const Xbyak_riscv::Reg &reg_dst = Xbyak_riscv::s2;
    const Xbyak_riscv::Reg &reg_id_start = Xbyak_riscv::s3;
    const Xbyak_riscv::Reg &reg_ih_start = Xbyak_riscv::s4;
    const Xbyak_riscv::Reg &reg_src_d_stride = Xbyak_riscv::s5;
    const Xbyak_riscv::Reg &reg_src_h_stride = Xbyak_riscv::s6;

    void generate();
    void generate_f16();

    void preamble() {
        using namespace Xbyak_riscv;
        addi(sp, sp, -64);
        sd(ra, sp, 56);
        sd(s1, sp, 48);
        sd(s2, sp, 40);
        sd(s3, sp, 32);
        sd(s4, sp, 24);
        sd(s5, sp, 16);
        sd(s6, sp, 8);
    }

    void postamble() {
        using namespace Xbyak_riscv;
        ld(s6, sp, 8);
        ld(s5, sp, 16);
        ld(s4, sp, 24);
        ld(s3, sp, 32);
        ld(s2, sp, 40);
        ld(s1, sp, 48);
        ld(ra, sp, 56);
        addi(sp, sp, 64);
        ret();
    }
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
