#!/bin/bash
set -e

# ---------- 检查交叉编译器 ----------
if [[ -z "${GCC_COMPILER}" ]]; then
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
INSTALL_DIR="${ROOT_DIR}/install"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake "${ROOT_DIR}" \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}"

make -j"$(nproc)"
make install

echo ""
echo "构建完成！产物在：${INSTALL_DIR}/rkq"
echo ""
echo "传到板子并运行："
echo "  scp ${INSTALL_DIR}/rkq board_ip:/path/"
echo "  # 在板子上："
echo "  chmod +x /path/rkq"
echo "  /path/rkq status"
