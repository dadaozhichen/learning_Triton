import torch

import triton
import triton.language as tl
from triton.language import constexpr


@triton.jit
def add_kernel(x_ptr,  # 第一个输入向量的指针。
               y_ptr,  # 第二个输入向量的指针。
               output_ptr,  # 输出向量的指针。
               n_elements,  # 向量的大小。
               BLOCK_SIZE: tl.constexpr,  # 每个程序应该处理的元素数量。
               # 注意：`constexpr` 可以作为形状值使用。
               ):
    pid = tl.program_id(axis=0)

    block_start = BLOCK_SIZE*pid

    offset = block_start+tl.arange(0,BLOCK_SIZE)

    mask = offset < n_elements

    x = tl.load(x_ptr+offset,mask=mask)
    y = tl.load(y_ptr+offset,mask=mask)

    output = x+y 

    tl.store(output_ptr+offset,output,mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)

    assert x.is_cuda and y.is_cuda and output.is_cuda

    n_elements = output.numel()

    grid = lambda meta:(triton.cdiv(n_elements,meta['BLOCK_SIZE']),)

    add_kernel[grid](x,y,output,n_elements,BLOCK_SIZE=1024)
    return output

torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'在torch和triton之间的最大差异是 '
      f'{torch.max(torch.abs(output_torch - output_triton))}')