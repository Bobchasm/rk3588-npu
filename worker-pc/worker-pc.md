# worker-pc

`worker-pc` 是给普通 PC 准备的 worker 节点实现，目标是让 PC 在当前项目里作为一个可调度节点参与分布式实验。

## 目标

- 先提供一个 **CPU 可运行** 的 PC worker。
- 保持与现有 `worker/` 相似的分层结构，方便后续把模型按 block 切到不同节点。
- 通过 `device` 抽象为后续 GPU 后端预留接口，避免 CPU / GPU 路径和模型主流程耦死。

## 当前状态

- `cpu`：可用，是真正的执行后端。
- `gpu`：当前是占位后端，接口已分离，但暂时退回 CPU 计算。
- `auto`：当前解析后默认落到 CPU。

## 构建

```bash
cd worker-pc
chmod +x build-linux.sh
./build-linux.sh
```

## 本地 demo

```bash
./build/qwen2_pc_demo --device cpu /path/to/model_dir 151644 8948 198
```

## 常驻 chat

```bash
./build/qwen2_pc_chat --device cpu /path/to/model_dir
```

## RPC 服务

```bash
./build/worker_pc_rpc_server --device cpu /path/to/model_dir 0.0.0.0:5001
```

然后你就可以让调度器把这个 PC 节点当成一个普通 worker 去访问。

## 目录说明

- `include/backend/`
  - 设备抽象与 CPU/GPU 线性层后端
- `include/model/`
  - 模型组织、KV cache、权重加载
- `include/network/`
  - RPC server 与服务层
- `src/api/`
  - `LLMEngine`
- `src/model/`
  - `Qwen2Model`
- `src/network/`
  - RPC 请求处理

## 后续扩展建议

- 将当前 `GpuLinear` 替换为真正 CUDA / Vulkan / ROCm 后端。
- 对齐分布式 stage 执行接口，使 `worker-pc` 可以直接承担 head / middle / tail 节点中的任意一种角色。
