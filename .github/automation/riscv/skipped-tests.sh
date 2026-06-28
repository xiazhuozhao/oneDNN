#!/usr/bin/env bash

# *******************************************************************************
# Copyright 2025 Arm Limited and affiliates.
# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *******************************************************************************

# Tests to skip for RISC-V architecture.

set -eo pipefail

# Skip these tests that will fail on CI for the RISC-V architecture.
SKIPPED_TESTS="cpu-matmul-coo-cpp|cpu-matmul-csr-cpp|test_sum"

if [[ "${ONEDNN_RISCV_PROFILE:-}" == "no_vector_extensions_benchmark" ]]; then
    # Conservative no-vector-extensions benchmark profile: keep the first CI
    # run well below the GitHub job timeout until we have real timing data
    # from a machine without vector extensions.
    SKIPPED_TESTS+="|test_graph_unit_dnnl_sdp_decomp_cpu"
    SKIPPED_TESTS+="|test_benchdnn_modeC_zeropad_ci_cpu"
    SKIPPED_TESTS+="|test_graph_unit_dnnl_mqa_decomp_cpu"
    SKIPPED_TESTS+="|cpu-graph-sdpa-cpp"
    SKIPPED_TESTS+="|test_benchdnn_modeC_deconv_ci_cpu"
    SKIPPED_TESTS+="|cpu-graph-gated-mlp-int4-cpp"
    SKIPPED_TESTS+="|test_gemm_u8s8s32"
    SKIPPED_TESTS+="|test_benchdnn_modeC_lnorm_ci_cpu"
    SKIPPED_TESTS+="|test_benchdnn_modeC_conv_ci_cpu"
    SKIPPED_TESTS+="|test_convolution_eltwise_forward_f32"
    SKIPPED_TESTS+="|test_gemm_s8s8s32"
    SKIPPED_TESTS+="|test_convolution_backward_weights_f32"
    SKIPPED_TESTS+="|test_pooling_forward"
    SKIPPED_TESTS+="|test_convolution_forward_f32"
    SKIPPED_TESTS+="|test_benchdnn_modeC_rnn_ci_cpu"
    SKIPPED_TESTS+="|test_convolution_backward_data_f32"
    SKIPPED_TESTS+="|test_gemm_f32"
    SKIPPED_TESTS+="|test_pooling_backward"
    SKIPPED_TESTS+="|cpu-graph-gqa-training-cpp"
    SKIPPED_TESTS+="|test_benchdnn_modeC_pool_ci_cpu"
    SKIPPED_TESTS+="|test_benchdnn_modeC_bnorm_ci_cpu"
    SKIPPED_TESTS+="|test_benchdnn_modeC_concat_ci_cpu"
    SKIPPED_TESTS+="|test_convolution_eltwise_forward_x8s8f32s32"
    SKIPPED_TESTS+="|test_benchdnn_modeC_binary_different_dt_ci_cpu"
    SKIPPED_TESTS+="|cpu-cnn-inference-f32-cpp"
    SKIPPED_TESTS+="|test_benchdnn_modeC_lstm_ci_cpu"
    SKIPPED_TESTS+="|test_graph_unit_dnnl_convolution_cpu"
    SKIPPED_TESTS+="|test_benchdnn_modeC_softmax_ci_cpu"
    SKIPPED_TESTS+="|cpu-performance-profiling-cpp"
    SKIPPED_TESTS+="|test_benchdnn_modeC_reorder_ci_cpu"
    SKIPPED_TESTS+="|test_lrn"
    SKIPPED_TESTS+="|test_benchdnn_modeC_binary_ci_cpu"
    SKIPPED_TESTS+="|test_benchdnn_modeC_gru_ci_cpu"
    SKIPPED_TESTS+="|cpu-cnn-training-f32-c"
    SKIPPED_TESTS+="|cpu-rnn-training-f32-cpp"
    SKIPPED_TESTS+="|api-c"
fi

if [[ "$ONEDNN_TEST_SET" == "SMOKE" ]]; then
    # Skip time-consuming tests (QEMU is slow)
    SKIPPED_TESTS+="|cpu-cnn-training-f32-cpp"
    SKIPPED_TESTS+="|cpu-cnn-inference-f32-cpp"
    SKIPPED_TESTS+="|cpu-cnn-training-f32-c"
    SKIPPED_TESTS+="|cpu-graph-gated-mlp-int4-cpp"
    SKIPPED_TESTS+="|cpu-performance-profiling-cpp"
    SKIPPED_TESTS+="|cpu-rnn-training-f32-cpp"
    SKIPPED_TESTS+="|test_convolution_backward_data_f32"
    SKIPPED_TESTS+="|test_convolution_backward_weights_f32"
    SKIPPED_TESTS+="|test_convolution_eltwise_forward_f32"
    SKIPPED_TESTS+="|test_convolution_eltwise_forward_x8s8f32s32"
    SKIPPED_TESTS+="|test_convolution_forward_f32"
    SKIPPED_TESTS+="|test_pooling_backward"
    SKIPPED_TESTS+="|test_pooling_forward"
    SKIPPED_TESTS+="|test_gemm_f32"
    SKIPPED_TESTS+="|test_gemm_s8s8s32"
    SKIPPED_TESTS+="|test_gemm_u8s8s32"
    SKIPPED_TESTS+="|test_graph_unit_dnnl_mqa_decomp_cpu"
    SKIPPED_TESTS+="|test_graph_unit_dnnl_sdp_decomp_cpu"
    SKIPPED_TESTS+="|cpu-graph-sdpa-cpp"
elif [[ "$ONEDNN_TEST_SET" == "CI" ]]; then
    SKIPPED_TESTS+="|test_benchdnn_modeC_matmul_sparse_ci_cpu"
    SKIPPED_TESTS+="|test_benchdnn_modeC_graph_ci_cpu"
    SKIPPED_TESTS+="|test_benchdnn_modeC_matmul_ci_cpu"
    SKIPPED_TESTS+="|test_benchdnn_modeC_ip_ci_cpu"
fi

echo "$SKIPPED_TESTS"
