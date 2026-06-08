# worker-pc

`worker-pc` 是给普通 PC 准备的 worker 节点实现，现支持纯cpu方案和gpu方案

## 当前状态

- `cpu`：可用，是真正的执行后端。
- `gpu`：已接入 CUDA/cuBLAS 线性层，并补上了 prefill/decode 两条 attention 的 GPU fast path 与设备端 KV cache 镜像；若运行机上 GPU 或 CUDA 不可用，会自动退回 CPU。
- `auto`：启动时自动探测 GPU，有可用 CUDA 设备就走 `gpu`，否则走 `cpu`。

## 构建

```bash
cd worker-pc
chmod +x build-linux.sh
./build-linux.sh
```

构建完成后会生成：

- `build/qwen2_pc_demo`
- `build/qwen2_pc_chat`
- `build/worker_pc_rpc_server`
- `build/worker_pc_gpu_probe`

## 本地 demo

```bash
./build/qwen2_pc_demo --device cpu /path/to/model_dir 151644 8948 198
./build/qwen2_pc_demo --device gpu /path/to/model_dir 151644 8948 198
./build/qwen2_pc_demo --device auto /path/to/model_dir 151644 8948 198
```

## GPU 自检

```bash
./build/worker_pc_gpu_probe
```

典型输出会包含：

- `cuda_build=on`
- `device_count=1`
- GPU 型号、算力版本、显存大小
- `auto_resolved=gpu`

## 常驻 chat

`auto` 可换成 `cpu` / `gpu`

```bash
./build/qwen2_pc_chat --device auto /path/to/model_dir
```

## RPC 服务

```bash
./build/worker_pc_rpc_server --device auto /path/to/model_dir 0.0.0.0:5001
```

然后你就可以让调度器把这个 PC 节点当成一个普通 worker 去访问。

## CUDA 说明

- 当前 GPU 路径已经覆盖：
  - `Linear / GEMM`
  - `prefill attention`
  - `decode attention`
  - `device-side KV cache mirror`
- 当前仍留在 CPU 的部分主要是：
  - `RoPE / RMSNorm / embedding / FFN 激活`
- 所以它已经不是“只有矩阵乘在 GPU”，而是 decode 主干已进入混合 CPU/GPU 执行。
- 若你在 WSL2 下使用 NVIDIA 显卡，通常可以直接复用 Windows 宿主机 GPU，不需要单独直通一块卡给虚拟机。
- 构建成功后可以用下面命令确认是否真的链到了 CUDA：

```bash
ldd ./build/qwen2_pc_demo | grep -E 'cudart|cublas'
```
