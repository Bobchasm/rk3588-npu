# bindings

`bindings/` 负责 Python 与 C++ 推理执行器之间的桥接。

设计目标：

- 不把 Python 逻辑混入 `worker/` 或 `worker-pc/`
- 对 Ray 层暴露统一接口
- 为后续 `pc` / `rk3588-` 两套底层实现保留扩展空间

当前状态：

- 已实现 `worker-pc` 的 pybind11 模块：`pc_engine`
- 统一 Python 工厂接口：`runtime.create_engine(target=...)`
- 已补 `rk3588` 模块骨架：`rk3588_engine`
- `rk3588` 版本需要在板端或对应 aarch64 交叉环境中实际编译验证

## 目录

```text
bindings/
├── CMakeLists.txt
├── python/runtime/
│   ├── __init__.py
│   └── engine.py
└── src/
    ├── pc_engine_module.cpp
    └── rk3588_engine_module.cpp
```

## 构建前提

- Python 开发头文件
- `pybind11`

安装 `pybind11`：

```bash
pip install pybind11
```

## 构建

```bash
cmake -S bindings -B bindings/build
cmake --build bindings/build -j4
```

默认只构建 `pc_engine`。

如果要构建 `rk3588_engine`，需要额外启用：

```bash
cmake -S bindings -B bindings/build-rk3588 -DBUILD_RK3588_PYTHON=ON
cmake --build bindings/build-rk3588 -j4
```

注意：

- `rk3588_engine` 依赖 `worker/` 下的 RKNN/NPU 代码
- 一般应在板端环境或 aarch64 交叉编译环境中构建
- `pc_engine` 和 `rk3588_engine` 的 `device` 语义不同
  - `pc`：`cpu` / `gpu` / `auto`
  - `rk3588`：`npu` / `cpu` / `npu_single` / `npu_sharded` / `auto`

生成的 Python 模块会输出到：

```text
bindings/python/runtime/
```
