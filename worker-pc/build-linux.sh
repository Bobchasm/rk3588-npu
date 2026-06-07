#!/bin/bash
set -e

ROOT_DIR="$( cd "$( dirname "$0" )" && pwd )"
BUILD_DIR="${ROOT_DIR}/build"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake "${ROOT_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build . -j"$(nproc)"

echo ""
echo "worker-pc build done."
echo "artifacts:"
echo "  ${BUILD_DIR}/qwen2_pc_demo"
echo "  ${BUILD_DIR}/qwen2_pc_chat"
echo "  ${BUILD_DIR}/worker_pc_rpc_server"
