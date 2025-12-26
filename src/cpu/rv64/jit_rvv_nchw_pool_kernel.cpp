#ifndef XBYAK_RISCV_V
#define XBYAK_RISCV_V 1
#endif

#include "cpu/rv64/jit_rvv_nchw_pool_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

#define GET_OFF(field) offsetof(jit_pool_call_t, field)

void jit_rvv_nchw_pool_kernel::generate() {
    preamble();
    ld(reg_src_base, a0, GET_OFF(src));
    ld(reg_dst, a0, GET_OFF(dst));
    ld(reg_id_start, a0, GET_OFF(id_start));
    ld(reg_ih_start, a0, GET_OFF(ih_start));
    ld(reg_src_d_stride, a0, GET_OFF(src_d_stride));
    ld(reg_src_h_stride, a0, GET_OFF(src_h_stride));

    if (jpp_.dt == data_type::f16) generate_f16();
    postamble();
}

void jit_rvv_nchw_pool_kernel::generate_f16() {
    auto T_REG_OW = t3;
    auto T_REG_VL = t4;
    auto T_REG_BASE_OW = t5;

    li(T_REG_OW, jpp_.ow);
    li(T_REG_BASE_OW, 0);

    Label l_ow_loop;
    L(l_ow_loop);
    {
        vsetvli(T_REG_VL, T_REG_OW, SEW::e16, LMUL::m4);

        if (jpp_.is_max) {
            li(t0, 0xfc00); // -Inf
            vmv_v_x(v24, t0);
        } else {
            vmv_v_i(v24, 0);
        }

        vid_v(v8);
        vadd_vx(v8, v8, T_REG_BASE_OW);
        if (jpp_.stride_w > 1) {
            li(t0, jpp_.stride_w);
            vmul_vx(v8, v8, t0);
        }
        int w_offset = -jpp_.pad_l;
        if (w_offset != 0) { li(t0, w_offset); vadd_vx(v8, v8, t0); }

        for (int kd = 0; kd < jpp_.kd; ++kd) {
            addi(t0, reg_id_start, kd);
            Label l_skip_kd;
            bltz(t0, l_skip_kd);
            li(t1, jpp_.id);
            bge(t0, t1, l_skip_kd);
            mul(t2, t0, reg_src_d_stride);

            for (int kh = 0; kh < jpp_.kh; ++kh) {
                addi(t0, reg_ih_start, kh);
                Label l_skip_kh;
                bltz(t0, l_skip_kh);
                li(t1, jpp_.ih);
                bge(t0, t1, l_skip_kh);
                mul(t1, t0, reg_src_h_stride);
                add(t1, t1, t2);
                add(t1, t1, reg_src_base);

                vmslt_vx(v0, v8, zero);
                vmnot_m(v0, v0);
                li(t2, jpp_.iw);
                vmslt_vx(v0, v8, t2, VM::masked);

                vsll_vi(v16, v8, 1);
                vluxei16_v(v20, t1, v16, VM::masked);

                if (jpp_.is_max) vfmax_vv(v24, v24, v20, VM::masked);
                else vfadd_vv(v24, v24, v20, VM::masked);
                L(l_skip_kh);
            }
            L(l_skip_kd);
        }
        vse16_v(v24, reg_dst);
        sub(T_REG_OW, T_REG_OW, T_REG_VL);
        add(T_REG_BASE_OW, T_REG_BASE_OW, T_REG_VL);
        slli(t0, T_REG_VL, 1);
        add(reg_dst, reg_dst, t0);
    }
    bnez(T_REG_OW, l_ow_loop);
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
