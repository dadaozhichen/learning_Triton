#  Learning Triton

欢迎来到 **Learning Triton** 仓库！这是一个专注于学习和实践 **Triton** 编程语言的小型实战项目。

> **Triton** 是由 OpenAI 开源的编程语言和编译器。它让开发者能够用类似 Python 的语法轻松编写高度优化的 GPU 内核，在保持开发效率的同时，获得媲美手写 CUDA 的极致性能。

##  项目结构

本仓库目前包含以下核心文件：

```text
.
├── add.py        # 向量加法算子 (Element-wise Addition)
├── softmax.py    # Softmax 算子实现与融合
└── Readme.md     # 项目说明文档
```


## 效果测试：
| 文件名 | 算子名称 | 核心知识点 | 状态 | 误差 
| :--- | :--- | :--- | :---: |:---:|
| `add.py` | 向量加法 | Block 并行、指针操作、Mask 防越界 | ✅ 已完成 | 0 
| `softmax.py` | Softmax 融合 | 归约操作 (Reduction)、内存优化 | ✅ 已完成 | 1e-10 
| `matmul.py` | 矩阵乘法 | Tiling、Shared Memory 模拟 | 🚧 计划中 | 