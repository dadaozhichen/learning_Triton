#  Learning Triton

欢迎来到 **Learning Triton** 仓库！这是一个专注于学习和实践 **Triton** 编程语言的小型实战项目。

> **Triton** 是由 OpenAI 开源的编程语言和编译器。它让开发者能够用类似 Python 的语法轻松编写高度优化的 GPU 内核，在保持开发效率的同时，获得媲美手写 CUDA 的极致性能。

##  项目结构

本仓库目前包含以下核心文件：

```text
.
├── add.py        # 向量加法算子 (Element-wise Addition)
├── softmax.py    # Softmax 算子实现与融合
├── matmul.py     # 矩阵乘法算子 (Matrix Multiplication)
└── Readme.md     # 项目说明文档
```


## 效果测试：

### ✅ 已完成算子
| 文件名 | 算子名称 | 核心知识点 | 误差 |
| :--- | :--- | :--- |:---:|
| `add.py` | 向量加法 | Block 并行、指针操作、Mask 防越界 | 0 |
| `softmax.py` | Softmax 融合 | 归约操作 (Reduction)、内存优化 | $10^{-10}$ |
| `matmul.py` | 矩阵乘法 | Tiling、`tl.dot`、2D 网格 / FP16 | $10^{-4}$ |

###   ️ 进阶路线图

#### 第一阶段：实际模型常见算子
| 计划算子 | 核心新知识点 | 阶段 |
| :--- | :--- | :---: |
| `layer_norm.py` | LayerNorm / RMSNorm | 两阶段归约 (mean→var)、`tl.where`、融合梯度 | 一 |
| `flash_attn.py` | Flash Attention | Online safe softmax + `tl.dot` 融合、分块双层循环 | 一 |
| `cross_entropy.py` | Cross Entropy (含梯度) | Reduction + 多输出 (loss & dX) | 一 |
| `rope.py` | RoPE 旋转位置编码 | `tl.cos`/`tl.sin`、slice 与复杂索引 | 一 |
| `gelu.py` | GELU / SwiGLU 融合激活 | 激活函数融合、element-wise 优化 | 一 |

#### 第二阶段：高性能进阶技巧
| 文件名 | 核心知识点 | 阶段 |
| :--- | :--- | :---: |
| `persistent_matmul.py` | 持久化 Kernel | 少量 program 循环处理多个 block，减少 launch overhead | 二 |
| `grouped_reduce.py` | 分组归约 / `tl.atomic_add` | 原子操作、多 program 并行累加 | 二 |
| `pipeline_matmul.py` | 软件流水线 | `tl.load` 与 `tl.dot` 交错重叠，隐藏访存延迟 | 二 |
| `fp8_matmul.py` | FP8 混合精度 | `input_dtype=tl.float8e5`、Hopper 特性 | 二 |

#### 第三阶段：工程化与性能调优
| 文件名 | 核心知识点 | 阶段 |
| :--- | :--- | :---: |
| `autotune_demo.py` | Autotune 自动调优 | `@triton.autotune`、`num_stages`/`num_warps`、`key` | 三 |
| `heuristics_demo.py` | Heuristics 启发式调度 | `@triton.heuristics`、动态常量推导 | 三 |
| `varlen_attn.py` | Varlen Attention | 变长序列 batch、累加偏移量 | 三 |