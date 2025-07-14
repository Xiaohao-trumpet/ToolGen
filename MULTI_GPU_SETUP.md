# 多GPU配置说明

## 问题描述

当你设置 `CUDA_VISIBLE_DEVICES=0,1,2` 时，模型仍然只在第0号GPU上运行，这是因为原始的ToolGen实现没有使用Hugging Face的 `device_map` 功能来在多GPU间分配模型。

## 解决方案

我已经修改了以下文件来支持多GPU：

### 1. 修改的文件

- `evaluation/toolbench/inference/LLM/toolgen.py` - ToolGen模型类
- `evaluation/toolbench/inference/LLM/toolgen_atomic.py` - ToolGenAtomic模型类  
- `evaluation/toolbench/inference/Downstream_tasks/rapidapi_multithread.py` - 模型加载逻辑
- `scripts/inference/inference_toolgen_pipeline_virtual.sh` - 添加调试信息

### 2. 主要修改内容

#### ToolGen和ToolGenAtomic类的修改：

1. **添加多GPU参数**：
   ```python
   def __init__(self, ..., num_gpus: int=1, max_gpu_memory: str=None):
   ```

2. **多GPU模型加载**：
   ```python
   if device == "cuda" and num_gpus > 1:
       # 自动计算每个GPU的内存分配
       available_gpu_memory = get_gpu_memory(num_gpus)
       max_memory = {
           i: str(int(available_gpu_memory[i] * 0.85)) + "GiB"
           for i in range(num_gpus)
       }
       
       self.model = AutoModelForCausalLM.from_pretrained(
           model_name_or_path,
           torch_dtype=torch.bfloat16,
           device_map="auto",  # 关键：自动分配设备
           max_memory=max_memory,
           low_cpu_mem_usage=True
       )
   ```

3. **智能设备检测**：
   ```python
   # 多GPU情况下，模型已经在正确的设备上
   if hasattr(self.model, 'device'):
       device = self.model.device
   else:
       device = next(self.model.parameters()).device
   ```

#### Pipeline Runner的修改：

自动检测GPU数量并传递给模型：
```python
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
backbone_model = ToolGen(
    model_name_or_path=args.model_path,
    template=args.template,
    indexing=args.indexing,
    num_gpus=num_gpus  # 传递GPU数量
)
```

## 使用方法

### 1. 运行你的脚本

```bash
bash scripts/inference/inference_toolgen_pipeline_virtual.sh
```

脚本现在会显示GPU配置信息，确认多GPU被正确识别。

### 2. 验证多GPU使用

运行后，你应该能看到类似这样的输出：
```
=== GPU配置信息 ===
CUDA_VISIBLE_DEVICES: 0,1,2
CUDA available: True
GPU count: 3
GPU 0: NVIDIA GeForce RTX 4090
GPU 1: NVIDIA GeForce RTX 4090  
GPU 2: NVIDIA GeForce RTX 4090
==================
```

### 3. 检查模型设备分布

模型加载后，你可以通过以下方式检查模型是否分布在多个GPU上：

```python
# 检查模型设备映射
if hasattr(model.model, 'hf_device_map'):
    for module_name, device in model.model.hf_device_map.items():
        print(f"{module_name}: {device}")
```

你应该看到模型的不同层被分配到不同的GPU上，例如：
```
model.embed_tokens: cuda:0
model.layers.0: cuda:0
model.layers.1: cuda:1
model.layers.2: cuda:1
...
model.norm: cuda:2
lm_head: cuda:2
```

## 内存管理

- 系统会自动检测每个GPU的可用内存
- 默认使用每个GPU 85%的可用内存
- 你可以通过 `max_gpu_memory` 参数自定义每个GPU的内存限制

## 注意事项

1. **模型大小**：确保你的模型大小适合多GPU分配
2. **内存平衡**：如果GPU内存不同，系统会自动平衡分配
3. **性能**：多GPU可能会增加一些通信开销，但对于大模型通常能显著提升性能

## 故障排除

如果仍然只在单GPU上运行：

1. 检查CUDA_VISIBLE_DEVICES设置
2. 确认torch.cuda.device_count()返回正确的GPU数量
3. 检查模型是否有hf_device_map属性
4. 查看GPU内存使用情况：`nvidia-smi`

## 测试脚本

我还创建了一个测试脚本 `test_multi_gpu.py`，你可以运行它来验证多GPU配置：

```bash
python test_multi_gpu.py
``` 