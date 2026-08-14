/*******************************************************************************
* Copyright 2018 Intel Corporation
* Copyright 2025 Institute of Software, Chinese Academy of Sciences
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
#include <cmath>
#include <cstddef>

#include "common/dnnl_thread.hpp"
#include "common/utils.hpp"

#include "cpu/rv64/gemm/rvv_gemm_utils_f32.hpp"
#include "cpu/rv64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace gemm_utils {

std::atomic<dim_t> rvv_gemm_f32_m_unroll {0};
std::atomic<dim_t> rvv_gemm_s8_m_unroll {0};

namespace {

using namespace Xbyak_riscv;

struct jit_rvv_sum_two_matrices_kernel_t : public jit_generator_t {
    struct call_params_t {
        const void *src;
        void *dst;
        dim_t ld_src;
        dim_t ld_dst;
        dim_t m;
        dim_t n;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_rvv_sum_two_matrices_kernel_t)

    explicit jit_rvv_sum_two_matrices_kernel_t(bool is_float)
        : jit_generator_t("rv64_sum_two_matrices_jit"), is_float_(is_float) {
        create_kernel();
    }

    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }

protected:
    void generate() override {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
#define SUM_OFF(field) static_cast<int32_t>(offsetof(call_params_t, field))

        const Reg reg_param = a0;
        const Reg reg_src_base = a1;
        const Reg reg_dst_base = a2;
        const Reg reg_ld_src_bytes = a3;
        const Reg reg_ld_dst_bytes = a4;
        const Reg reg_m = a5;
        const Reg reg_n = a6;
        const Reg reg_src = t0;
        const Reg reg_dst = t1;
        const Reg reg_remaining = t2;
        const Reg reg_vl = t3;
        const Reg reg_bytes = t4;

        const VReg v_src(0);
        const VReg v_dst(4);

        ld(reg_src_base, reg_param, SUM_OFF(src));
        ld(reg_dst_base, reg_param, SUM_OFF(dst));
        ld(reg_ld_src_bytes, reg_param, SUM_OFF(ld_src));
        ld(reg_ld_dst_bytes, reg_param, SUM_OFF(ld_dst));
        ld(reg_m, reg_param, SUM_OFF(m));
        ld(reg_n, reg_param, SUM_OFF(n));
        slli(reg_ld_src_bytes, reg_ld_src_bytes, 2);
        slli(reg_ld_dst_bytes, reg_ld_dst_bytes, 2);

        Label column_loop, vector_loop, next_column, done;
        L(column_loop);
        beqz(reg_n, done);
        mv(reg_src, reg_src_base);
        mv(reg_dst, reg_dst_base);
        mv(reg_remaining, reg_m);

        L(vector_loop);
        beqz(reg_remaining, next_column);
        vsetvli(reg_vl, reg_remaining, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);
        vle32_v(v_src, reg_src);
        vle32_v(v_dst, reg_dst);
        if (is_float_)
            vfadd_vv(v_dst, v_dst, v_src);
        else
            vadd_vv(v_dst, v_dst, v_src);
        vse32_v(v_dst, reg_dst);
        slli(reg_bytes, reg_vl, 2);
        add(reg_src, reg_src, reg_bytes);
        add(reg_dst, reg_dst, reg_bytes);
        sub(reg_remaining, reg_remaining, reg_vl);
        j_(vector_loop);

        L(next_column);
        add(reg_src_base, reg_src_base, reg_ld_src_bytes);
        add(reg_dst_base, reg_dst_base, reg_ld_dst_bytes);
        addi(reg_n, reg_n, -1);
        j_(column_loop);

        L(done);
        ret();
#undef SUM_OFF
#else
        ret();
#endif
    }

private:
    bool is_float_;
};

void run_sum_two_matrices(dim_t m, dim_t n, void *p_src, dim_t ld_src,
        void *p_dst, dim_t ld_dst, bool is_float) {
    constexpr dim_t jit_sum_two_matrices_min_elems = 8192;
    if (m * n < jit_sum_two_matrices_min_elems) {
        if (is_float) {
            auto *src = static_cast<float *>(p_src);
            auto *dst = static_cast<float *>(p_dst);
            for (dim_t j = 0; j < n; j++)
                for (dim_t i = 0; i < m; i++)
                    dst[i + j * ld_dst] += src[i + j * ld_src];
        } else {
            auto *src = static_cast<int32_t *>(p_src);
            auto *dst = static_cast<int32_t *>(p_dst);
            for (dim_t j = 0; j < n; j++)
                for (dim_t i = 0; i < m; i++)
                    dst[i + j * ld_dst] += src[i + j * ld_src];
        }
        return;
    }

    const jit_rvv_sum_two_matrices_kernel_t::call_params_t p {
            p_src, p_dst, ld_src, ld_dst, m, n};
    if (is_float) {
        static const jit_rvv_sum_two_matrices_kernel_t kernel(true);
        kernel(&p);
    } else {
        static const jit_rvv_sum_two_matrices_kernel_t kernel(false);
        kernel(&p);
    }
}

} // namespace

void sum_two_matrices(dim_t m, dim_t n, float *__restrict p_src, dim_t ld_src,
        float *__restrict p_dst, dim_t ld_dst) {
    run_sum_two_matrices(m, n, p_src, ld_src, p_dst, ld_dst, true);
}

void sum_two_matrices(dim_t m, dim_t n, int32_t *__restrict p_src, dim_t ld_src,
        int32_t *__restrict p_dst, dim_t ld_dst) {
    run_sum_two_matrices(m, n, p_src, ld_src, p_dst, ld_dst, false);
}

#define BM_NOCOPY_RVV 64
#define BN_NOCOPY_RVV 48
#define BK_NOCOPY_RVV 384
#define BN_LARGE_NOCOPY_RVV 192
#define BM_SMALL_NOCOPY_RVV 16
#define BN_SMALL_NOCOPY_RVV 1
#define BK_SMALL_NOCOPY_RVV 4
// Determine number of threads for each dimension of a 3-D partitioning
// algorithm based on input parameters
// m/n/k - First/second/third parameter for GEMM
// nthrs - total available number of threads
// nthrs_m/nthrs_n/nthrs_k - number of threads to use in each dimension
// BM/BN/BK - blocking values
void calc_nthr_nocopy_rvv(dim_t m, dim_t n, dim_t k, int nthrs, int *nthrs_m,
        int *nthrs_n, int *nthrs_k, dim_t *BM, dim_t *BN, dim_t *BK) {

    // Quick exit for single thread.
    if (nthrs == 1) {
        *nthrs_m = 1;
        *nthrs_n = 1;
        *nthrs_k = 1;

        *BM = m;
        *BN = n;
        *BK = k;
        return;
    }

    int nthr, nthr_m, nthr_n, nthr_k;
    dim_t MB, NB, KB;

    nthr = nthrs;
    nthr_m = static_cast<int>((m + BM_NOCOPY_RVV - 1) / BM_NOCOPY_RVV);
    nthr_n = static_cast<int>((n + BN_NOCOPY_RVV - 1) / BN_NOCOPY_RVV);
    nthr_k = 1;

    // Partition along K dimension
    //  - if threading allows having barriers (e.g. OMP)
    //  - if there is not enough parallelism along M or N
    if (dnnl_thr_syncable()) {
        int nthr_other = nthr_k = 1;
        while ((nthr_m * nthr_n * nthr_other < nthr)
                && (k / (nthr_other + 1) > BK_NOCOPY_RVV)) {
            nthr_other++;
            if ((nthr / nthr_other) * nthr_other > 0.9 * nthr)
                nthr_k = nthr_other;
        }
    }
    nthr /= nthr_k;

    if (nthr_m == 1) nthr_n = nthr;
    if (nthr_n == 1) nthr_m = nthr;

    // Simple partition reduction
    while (nthr_m * nthr_n > nthr)
        if (nthr_m > nthr_n)
            nthr_m--;
        else
            nthr_n--;
    while (nthr_m * nthr_n < nthr)
        if (nthr_m < nthr_n)
            nthr_m++;
        else
            nthr_n++;

    if ((nthr_m * nthr_n > nthr) && (nthr_m > 1) && (nthr_n > 1)) {

        if (nthr_m <= nthr_n) {
            nthr_m = (int)sqrt((double)nthr);
            if (nthr_m > (m + BM_SMALL_NOCOPY_RVV - 1) / BM_SMALL_NOCOPY_RVV)
                nthr_m = static_cast<int>(
                        (m + BM_SMALL_NOCOPY_RVV - 1) / BM_SMALL_NOCOPY_RVV);
            nthr_n = nthr / nthr_m;

            while ((nthr_m > 1) && (nthr_m * nthr_n != nthr)) {
                nthr_m--;
                nthr_n = nthr / nthr_m;
            }
        } else {
            nthr_n = (int)sqrt((double)nthr);
            if (nthr_n > (n + BN_SMALL_NOCOPY_RVV - 1) / BN_SMALL_NOCOPY_RVV)
                nthr_n = static_cast<int>(
                        (n + BN_SMALL_NOCOPY_RVV - 1) / BN_SMALL_NOCOPY_RVV);
            nthr_m = nthr / nthr_n;

            while ((nthr_n > 1) && (nthr_m * nthr_n != nthr)) {
                nthr_n--;
                nthr_m = nthr / nthr_n;
            }
        }
    }

    MB = (m + nthr_m - 1) / nthr_m + BM_SMALL_NOCOPY_RVV - 1;
    MB -= MB % BM_SMALL_NOCOPY_RVV;
    NB = (n + nthr_n - 1) / nthr_n + BN_SMALL_NOCOPY_RVV - 1;
    NB -= NB % BN_SMALL_NOCOPY_RVV;
    KB = (k + nthr_k - 1) / nthr_k + BK_SMALL_NOCOPY_RVV - 1;
    KB -= KB % BK_SMALL_NOCOPY_RVV;

    if (MB * nthr_m > m) nthr_m = static_cast<int>((m + MB - 1) / MB);
    if (NB * nthr_n > n) nthr_n = static_cast<int>((n + NB - 1) / NB);
    if (KB * nthr_k > k) nthr_k = static_cast<int>((k + KB - 1) / KB);

    *nthrs_m = nthr_m;
    *nthrs_n = nthr_n;
    *nthrs_k = nthr_k;

    *BM = MB;
    *BN = NB;
    *BK = KB;
}
#undef BM_NOCOPY_RVV
#undef BN_NOCOPY_RVV
#undef BK_NOCOPY_RVV
#undef BN_LARGE_NOCOPY_RVV
#undef BM_SMALL_NOCOPY_RVV
#undef BN_SMALL_NOCOPY_RVV
#undef BK_SMALL_NOCOPY_RVV

// Partition n values as equally as possible among nthr threads
// and set the offset (t_offset) and number of values (t_block) for ithr
// Assumption: 0 <= ithr < nthr
void partition_unit_diff(
        int ithr, int nthr, dim_t n, dim_t *t_offset, dim_t *t_block) {

    dim_t band = n / nthr;
    if (band == 0) band = 1;
    dim_t tail = n - band * nthr;
    if (tail < 0) tail = 0;

    if (ithr < tail) {
        band++;
        *t_offset = band * ithr;
        *t_block = band;
    } else {
        *t_offset = band * ithr + tail;
        *t_block = band;
    }

    if (*t_offset >= n) {
        *t_offset = 0;
        *t_block = 0;
    }

    if (*t_offset + *t_block > n) { *t_block = n - *t_offset; }
}

} // namespace gemm_utils
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
