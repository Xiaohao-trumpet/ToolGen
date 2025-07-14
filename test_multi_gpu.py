#!/usr/bin/env python3
"""
测试多GPU配置的脚本
"""

import torch
import os
from evaluation.toolbench.inference.LLM.toolgen import ToolGen
from evaluation.toolbench.inference.LLM.toolgen_atomic import ToolGenAtomic

def test_multi_gpu():
    print("=== 多GPU配置测试 ===")
    
    # 检查CUDA是否可用
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    
    # 设置环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
    
    # 测试ToolGen模型
    print("\n=== 测试ToolGen模型 ===")
    try:
        model = ToolGen(
            model_name_or_path="reasonwang/ToolGen-Llama-3-8B",
            template="llama-3",
            indexing="Atomic",
            num_gpus=3
        )
        print("ToolGen模型加载成功")
        
        # 检查模型设备分布
        if hasattr(model.model, 'hf_device_map'):
            print("模型设备映射:")
            for module_name, device in model.model.hf_device_map.items():
                print(f"  {module_name}: {device}")
        else:
            print("模型设备:", next(model.model.parameters()).device)
            
    except Exception as e:
        print(f"ToolGen模型加载失败: {e}")
    
    # 测试ToolGenAtomic模型
    print("\n=== 测试ToolGenAtomic模型 ===")
    try:
        model_atomic = ToolGenAtomic(
            model_name_or_path="reasonwang/ToolGen-Llama-3-8B",
            template="llama-3",
            num_gpus=3
        )
        print("ToolGenAtomic模型加载成功")
        
        # 检查模型设备分布
        if hasattr(model_atomic.model, 'hf_device_map'):
            print("模型设备映射:")
            for module_name, device in model_atomic.model.hf_device_map.items():
                print(f"  {module_name}: {device}")
        else:
            print("模型设备:", next(model_atomic.model.parameters()).device)
            
    except Exception as e:
        print(f"ToolGenAtomic模型加载失败: {e}")

if __name__ == "__main__":
    test_multi_gpu() 