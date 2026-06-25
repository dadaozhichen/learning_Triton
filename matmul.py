import triton
import triton.language as tl
from triton.language import constexpr
import torch


@triton.jit()
def matmul_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    N,  # X 的行数
    M,  # Y 的列数
    K,  # X 的列数 = Y 的行数
    BLOCK_SIZE: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    row_start,col_start = pid0*BLOCK_SIZE,pid1*BLOCK_SIZE

    row_offset = row_start + tl.arange(0,BLOCK_SIZE)
    col_offset = col_start + tl.arange(0,BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,BLOCK_SIZE),dtype=tl.float32)

    for k in range(0,K,BLOCK_SIZE):
        k_offset = k + tl.arange(0,BLOCK_SIZE)

        mask_x = (row_offset[:,None] < N) & (k_offset[None,:] < K)
        x = tl.load(x_ptr+row_offset[:,None]*K+k_offset[None,:],mask=mask_x)

        mask_y = (col_offset[None,:] < M) & (k_offset[:,None] < K)
        y = tl.load(y_ptr+k_offset[:,None]*M+col_offset[None,:],mask=mask_y)

        acc+=tl.dot(x,y)
    
    mask = (row_offset[:,None] < N) & (col_offset[None,:] < M)
    tl.store(output_ptr+row_offset[:,None]*M+col_offset[None,:],acc,mask=mask)


def matmul(x: torch.Tensor, y: torch.Tensor):
    N,K = x.shape
    K1,M = y.shape

    assert K1==K ,"x 和 y 维度并不匹配"

    output = torch.empty((N,M),dtype=x.dtype,device=x.device)

    assert x.is_cuda and y.is_cuda and output.is_cuda

    grid = lambda meta:(triton.cdiv(N,meta["BLOCK_SIZE"]),triton.cdiv(M,meta["BLOCK_SIZE"]))

    matmul_kernel[grid](x,y,output,N,M,K,BLOCK_SIZE=32)

    return output


device = torch.device("cuda")
# 测试
torch.manual_seed(0)
N, K, M = 1280, 640, 2560
x = torch.randn(N, K, device=device)
y = torch.randn(K, M, device=device)

output_torch = torch.matmul(x, y)
output_triton = matmul(x, y)

diff = torch.max(torch.abs(output_torch - output_triton))
relative_diff = diff / torch.max(torch.abs(output_torch))
print(f"fp32 测试 - 最大绝对误差: {diff:.6f}, 相对误差: {relative_diff:.6f}")
print(f"测试 {'通过 ✅' if relative_diff < 1e-2 else '失败 ❌'}")

# 更大规模测试
N2, K2, M2 = 5120, 2560, 5120

x2 = torch.randn(N2, K2, device=device, dtype=torch.float16)
y2 = torch.randn(K2, M2, device=device, dtype=torch.float16)

output_torch2 = torch.matmul(x2, y2)
output_triton2 = matmul(x2, y2)

diff2 = torch.max(torch.abs(output_torch2.float() - output_triton2.float()))
print(f"fp16 大规模测试 - 最大误差: {diff2}")
print(f"测试 {'通过 ✅' if diff2 < 3e-1 else '失败 ❌'}")
