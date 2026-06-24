import triton
import triton.language as tl

from triton.language import constexpr
import torch 
@triton.jit()
def softmax_kernel(x_ptr,
                   output_ptr,
                   M,
                   N,
                   BLOCK_SIZE:constexpr):
    pid = tl.program_id(axis=0)

    row_start = pid*N
    row_max = float('-inf')
    for col_start in range(0,N,BLOCK_SIZE):
        offset = col_start+tl.arange(0,BLOCK_SIZE)
        mask = offset < N 
        x = tl.load(x_ptr+row_start+offset,mask=mask,other=-float("inf"))
        row_max = tl.maximum(row_max,tl.max(x))

    sum_exp =0.0
    for col_start in range(0,N,BLOCK_SIZE):
        offset = col_start+tl.arange(0,BLOCK_SIZE)
        mask = offset < N 
        x = tl.load(x_ptr+row_start+offset,mask=mask,other=-float("inf"))
        exp_x = tl.exp(x-row_max)
        sum_exp +=tl.sum(exp_x)
    
    for col_start in range(0,N,BLOCK_SIZE):
        offset = col_start+tl.arange(0,BLOCK_SIZE)
        mask = offset < N 
        x = tl.load(x_ptr+row_start+offset,mask=mask,other=-float("inf"))
        exp_x = tl.exp(x-row_max)
        output = exp_x/sum_exp
        tl.store(output_ptr+row_start+offset,output,mask=mask)



def softmax(x:torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    M,N = x.shape
    grid = lambda meta:(M,)

    softmax_kernel[grid](x,output,M,N,BLOCK_SIZE=2048)

    return output


size = (90,91212)
device = torch.device('cuda')
x = torch.randn(size,device=device)

y1 = torch.softmax(x,dim=1)
y2 = softmax(x)

print(torch.max(torch.abs(y1-y2)))