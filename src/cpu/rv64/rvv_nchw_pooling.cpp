/******************************************************************************
* Copyright 2023 Intel Corporation
* Copyright 2023-2025 KNS Group LLC (YADRO)
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

#include <algorithm>
#include <float.h>
#include <riscv_vector.h>

#include "common/dnnl_thread.hpp"
#include "common/stream.hpp"
#include "cpu/rv64/rvv_nchw_pooling.hpp"
#include "cpu/rv64/jit_rvv_nchw_pool_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

namespace {
void MaxPooling(const float *src, float *dst, const dim_t batch,
        const dim_t channels, const dim_t outD, const dim_t outH,
        const dim_t outW, const dim_t inD, const dim_t inH, const dim_t inW,
        const dim_t kerD, const dim_t kerH, const dim_t kerW,
        const dim_t strideD, const dim_t strideH, const dim_t strideW,
        const dim_t padFront, const dim_t padTop, const dim_t padLeft) {
    parallel_nd(batch, channels, outD, outH, outW,
            [&](dim_t mb, dim_t c, dim_t od, dim_t oh, dim_t ow) {
                const size_t dst_offset = mb * channels * outD * outH * outW
                        + c * outD * outH * outW + od * outH * outW
                        + oh * outW + ow;
                float max_val = -FLT_MAX;
                for (dim_t kd = 0; kd < kerD; ++kd) {
                    const dim_t id = od * strideD - padFront + kd;
                    if (id < 0 || id >= inD) continue;
                    for (dim_t kh = 0; kh < kerH; ++kh) {
                        const dim_t ih = oh * strideH - padTop + kh;
                        if (ih < 0 || ih >= inH) continue;
                        for (dim_t kw = 0; kw < kerW; ++kw) {
                            const dim_t iw = ow * strideW - padLeft + kw;
                            if (iw < 0 || iw >= inW) continue;

                            const size_t src_offset
                                    = mb * channels * inD * inH * inW
                                    + c * inD * inH * inW + id * inH * inW
                                    + ih * inW + iw;
                            max_val = std::max(max_val, src[src_offset]);
                        }
                    }
                }
                dst[dst_offset] = max_val;
            });
}

void AvgPoolingExcludePadding(const float *src, float *dst, const dim_t batch,
        const dim_t channels, const dim_t outD, const dim_t outH,
        const dim_t outW, const dim_t inD, const dim_t inH, const dim_t inW,
        const dim_t kerD, const dim_t kerH, const dim_t kerW,
        const dim_t strideD, const dim_t strideH, const dim_t strideW,
        const dim_t padFront, const dim_t padTop, const dim_t padLeft) {
    parallel_nd(batch, channels, outD, outH, outW,
            [&](dim_t mb, dim_t c, dim_t od, dim_t oh, dim_t ow) {
                const size_t dst_offset = mb * channels * outD * outH * outW
                        + c * outD * outH * outW + od * outH * outW
                        + oh * outW + ow;
                float sum = 0;
                int count = 0;
                for (dim_t kd = 0; kd < kerD; ++kd) {
                    const dim_t id = od * strideD - padFront + kd;
                    if (id < 0 || id >= inD) continue;
                    for (dim_t kh = 0; kh < kerH; ++kh) {
                        const dim_t ih = oh * strideH - padTop + kh;
                        if (ih < 0 || ih >= inH) continue;
                        for (dim_t kw = 0; kw < kerW; ++kw) {
                            const dim_t iw = ow * strideW - padLeft + kw;
                            if (iw < 0 || iw >= inW) continue;

                            const size_t src_offset
                                    = mb * channels * inD * inH * inW
                                    + c * inD * inH * inW + id * inH * inW
                                    + ih * inW + iw;
                            sum += src[src_offset];
                            count++;
                        }
                    }
                }
                dst[dst_offset] = (count > 0) ? sum / count : 0;
            });
}

void AvgPoolingIncludePadding(const float *src, float *dst, const dim_t batch,
        const dim_t channels, const dim_t outD, const dim_t outH,
        const dim_t outW, const dim_t inD, const dim_t inH, const dim_t inW,
        const dim_t kerD, const dim_t kerH, const dim_t kerW,
        const dim_t strideD, const dim_t strideH, const dim_t strideW,
        const dim_t padFront, const dim_t padTop, const dim_t padLeft) {
    parallel_nd(batch, channels, outD, outH, outW,
            [&](dim_t mb, dim_t c, dim_t od, dim_t oh, dim_t ow) {
                const size_t dst_offset = mb * channels * outD * outH * outW
                        + c * outD * outH * outW + od * outH * outW
                        + oh * outW + ow;
                float sum = 0;
                for (dim_t kd = 0; kd < kerD; ++kd) {
                    const dim_t id = od * strideD - padFront + kd;
                    if (id < 0 || id >= inD) continue;
                    for (dim_t kh = 0; kh < kerH; ++kh) {
                        const dim_t ih = oh * strideH - padTop + kh;
                        if (ih < 0 || ih >= inH) continue;
                        for (dim_t kw = 0; kw < kerW; ++kw) {
                            const dim_t iw = ow * strideW - padLeft + kw;
                            if (iw < 0 || iw >= inW) continue;

                            const size_t src_offset
                                    = mb * channels * inD * inH * inW
                                    + c * inD * inH * inW + id * inH * inW
                                    + ih * inW + iw;
                            sum += src[src_offset];
                        }
                    }
                }
                dst[dst_offset] = sum / (kerD * kerH * kerW);
            });
}
} // namespace

riscv_nchw_pooling_fwd_t::riscv_nchw_pooling_fwd_t(const pd_t *apd)
    : primitive_t(apd) {
    
    if (pd()->src_md()->data_type == data_type::f16) {
        jit_pool_conf_t jpp;
        jpp.dt = data_type::f16;
        jpp.ndims = pd()->ndims();
        jpp.mb = pd()->MB();
        jpp.c = pd()->OC();
        
        jpp.id = (jpp.ndims >= 5) ? pd()->ID() : 1;
        jpp.ih = pd()->IH();
        jpp.iw = pd()->IW();
        jpp.od = (jpp.ndims >= 5) ? pd()->OD() : 1;
        jpp.oh = pd()->OH();
        jpp.ow = pd()->OW();
        
        jpp.kd = (jpp.ndims >= 5) ? pd()->KD() : 1;
        jpp.kh = pd()->KH();
        jpp.kw = pd()->KW();
        
        jpp.stride_d = (jpp.ndims >= 5) ? pd()->KSD() : 1;
        jpp.stride_h = pd()->KSH();
        jpp.stride_w = pd()->KSW();
        
        jpp.pad_f = (jpp.ndims >= 5) ? pd()->padFront() : 0;
        jpp.pad_t = pd()->padT();
        jpp.pad_l = pd()->padL();
        
        jpp.is_max = (pd()->desc()->alg_kind == alg_kind::pooling_max);

        kernel_ = std::make_shared<jit_rvv_nchw_pool_kernel>(jpp);
        if (kernel_) {
            kernel_->create_kernel();
        }
    }
}

status_t riscv_nchw_pooling_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    
    auto src_ptr = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto dst_ptr = CTX_OUT_MEM(void *, DNNL_ARG_DST);

    const auto dt = pd()->src_md()->data_type;

    if (dt == data_type::f32) {
        const float *src = static_cast<const float *>(src_ptr);
        float *dst = static_cast<float *>(dst_ptr);

        const dim_t MB = pd()->MB();
        const dim_t C = pd()->OC();
        const dim_t OD = pd()->OD();
        const dim_t OH = pd()->OH();
        const dim_t OW = pd()->OW();
        const dim_t ID = pd()->ID();
        const dim_t IH = pd()->IH();
        const dim_t IW = pd()->IW();
        const dim_t KD = pd()->KD();
        const dim_t KH = pd()->KH();
        const dim_t KW = pd()->KW();
        const dim_t SD = pd()->KSD();
        const dim_t SH = pd()->KSH();
        const dim_t SW = pd()->KSW();
        const dim_t padF = pd()->padFront();
        const dim_t padT = pd()->padT();
        const dim_t padL = pd()->padL();

        const auto alg = pd()->desc()->alg_kind;

        if (alg == alg_kind::pooling_max) {
            MaxPooling(src, dst, MB, C, OD, OH, OW, ID, IH, IW, KD, KH, KW, SD,
                    SH, SW, padF, padT, padL);
        } else if (alg == alg_kind::pooling_avg_exclude_padding) {
            AvgPoolingExcludePadding(src, dst, MB, C, OD, OH, OW, ID, IH, IW,
                    KD, KH, KW, SD, SH, SW, padF, padT, padL);
        } else if (alg == alg_kind::pooling_avg_include_padding) {
            AvgPoolingIncludePadding(src, dst, MB, C, OD, OH, OW, ID, IH, IW,
                    KD, KH, KW, SD, SH, SW, padF, padT, padL);
        } else {
            return status::unimplemented;
        }
    } 
    else if (dt == data_type::f16) {
        if (!kernel_) return status::runtime_error;

        const dim_t MB = pd()->MB();
        const dim_t C = pd()->OC();
        const dim_t OD = (pd()->ndims() >= 5) ? pd()->OD() : 1;
        const dim_t OH = pd()->OH();
        
        const dim_t ID = pd()->ID();
        const dim_t IH = pd()->IH();
        const dim_t IW = pd()->IW();
        
        const size_t src_stride_h = (size_t)IW * 2;
        const size_t src_stride_d = (size_t)IH * src_stride_h;
        const size_t src_stride_c = (size_t)ID * src_stride_d;
        const size_t src_stride_n = (size_t)C * src_stride_c;

        const size_t dst_stride_h = (size_t)pd()->OW() * 2;
        const size_t dst_stride_d = (size_t)pd()->OH() * dst_stride_h;
        const size_t dst_stride_c = (size_t)pd()->OD() * dst_stride_d;
        const size_t dst_stride_n = (size_t)C * dst_stride_c;

        parallel_nd(MB, C, OD, OH, [&](dim_t mb, dim_t c, dim_t od, dim_t oh) {
            jit_pool_call_t args;

            args.src = (const char*)src_ptr + mb * src_stride_n + c * src_stride_c;
            args.dst = (char*)dst_ptr + mb * dst_stride_n + c * dst_stride_c 
                       + od * dst_stride_d + oh * dst_stride_h;

            args.id_start = (size_t)((long)od * (pd()->ndims() >= 5 ? pd()->KSD() : 0) - pd()->padFront());
            args.ih_start = (size_t)((long)oh * pd()->KSH() - pd()->padT());

            args.src_d_stride = src_stride_d;
            args.src_h_stride = src_stride_h;
            args.indices = nullptr;

            // 执行 JIT Kernel
            (*kernel_)(&args);
        });
    } else {
        return status::unimplemented;
    }

    return status::success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
