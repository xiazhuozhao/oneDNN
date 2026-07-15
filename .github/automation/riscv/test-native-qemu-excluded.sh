#!/usr/bin/env bash

# *******************************************************************************
# Copyright 2026 Institute of Software, Chinese Academy of Sciences
# SPDX-License-Identifier: Apache-2.0
# *******************************************************************************

# Run exactly one CI test excluded from QEMU execution on native RISC-V hardware.

set -u -o pipefail

readonly test_name="${1:?usage: $0 <ctest-name>}"
readonly log_dir="${ONEDNN_TEST_LOG_DIR:-native-test-logs}"

case "${test_name}" in
    test_benchdnn_modeC_matmul_sparse_ci_cpu \
            | test_benchdnn_modeC_graph_ci_cpu \
            | test_benchdnn_modeC_matmul_ci_cpu \
            | test_benchdnn_modeC_ip_ci_cpu \
            | cpu-matmul-coo-cpp \
            | cpu-matmul-csr-cpp \
            | test_sum \
            | cpu-graph-sdpa-cpp) ;;
    *)
        echo "Unsupported native QEMU-excluded test: ${test_name}" >&2
        exit 2
        ;;
esac

# These variables make a native binary run through QEMU when inherited from
# the cross-compilation workflow environment.
unset QEMU_LD_PREFIX QEMU_CPU

mkdir -p "${log_dir}"

timeout --signal=TERM --kill-after=5m 6h \
        ctest --no-tests=error --output-on-failure -R "^${test_name}$" \
        2>&1 | tee "${log_dir}/${test_name}.log"
