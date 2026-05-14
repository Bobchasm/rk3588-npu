#!/bin/bash
set -e

# ---------- 检查交叉编译器 ----------
if [[ -z "${GCC_COMPILER}" ]]; then
    # 尝试默认路径
    if command -v aarch64-linux-gnu-g++ &>/dev/null; then
        export GCC_COMPILER=aarch64-linux-gnu
    else
        echo "错误：请设置 GCC_COMPILER 环境变量"
        echo "示例：export GCC_COMPILER=aarch64-linux-gnu"
        echo "或：   export GCC_COMPILER=~/opt/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu"
        exit 1
    fi
fi

export CC="${GCC_COMPILER}-gcc"
export CXX="${GCC_COMPILER}-g++"
echo "使用编译器: $CXX"

# ---------- 构建目录 ----------
ROOT_DIR="$( cd "$( dirname "$0" )" && pwd )"
BUILD_DIR="${ROOT_DIR}/build/aarch64"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake "${ROOT_DIR}" \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}"

make -j$(nproc)
make install

echo ""
echo "构建完成！产物在：${ROOT_DIR}/install/"
echo ""
echo "传到板子并运行："
echo "  scp ${ROOT_DIR}/install/qwen2_demo board_ip:/path/"
echo "  scp ${ROOT_DIR}/install/librknnrt.so board_ip:/path/"
echo "  scp -r /path/to/model/Qwen1.5B board_ip:/path/"
echo "  # 在板子上："
echo "  export LD_LIBRARY_PATH=/path:\$LD_LIBRARY_PATH"
echo "  ./qwen2_demo /path/Qwen1.5B <token_id1> <token_id2> ..."
