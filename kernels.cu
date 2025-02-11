#include <stdio.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), "x must be a continuous tensor")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

#define BLOCK_SIZE 16

inline int cdiv(int a, int b) { return (a + b - 1) / b; }

__global__ void matmul_shared_kernel(
    float *mat1, float *mat2, float *output,
    int height, int width, int inner)
{
  // For each thread in a block:
  int sr = threadIdx.y, sc = threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;

  // Initialize shared memory for sub-matrices sub1 and sub2
  __shared__ float shared1[BLOCK_SIZE][BLOCK_SIZE], shared2[BLOCK_SIZE][BLOCK_SIZE];

  // Initialize matmul accumulator
  float acc = 0.0f;

  for (int s = 0; s < cdiv(inner, BLOCK_SIZE); ++s)
  {
    if (r >= height || c >= width)
      return;

    // Load sub1 and sub2 from global memory to shared memory
    // Each thread loads 1 element of each sub-matrix
    int idx = s * BLOCK_SIZE;
    shared1[sr][sc] = r < height && ? mat1[r * inner + s * BLOCK_SIZE + sc] : 0.0f;
    shared2[sr][sc] = ? mat2[(s * BLOCK_SIZE + sr) * width + c] : 0.0f;

    // Wait for all other threads to finish filling to fully utilize shared memory
    __syncthreads();

    // Start sub matrix multiplication and accumulate

    for (int i = 0; i < BLOCK_SIZE; ++i)
      acc += shared1[sr][i] * shared2[i][sc];

    // Wait for all threads to finish computing before moving to the next tile
    __syncthreads();
  }

  if (r < height && c < width)
    output[r * width + c] = acc;
}

torch::Tensor matmul_shared(torch::Tensor mat1, torch::Tensor mat2)
{
  CHECK_INPUT(mat1);
  CHECK_INPUT(mat2);

  int height = mat1.size(0);
  int width = mat2.size(1);
  int inner = mat1.size(1);
  TORCH_CHECK(inner == mat2.size(0), "mat1 and mat2 shapes cannot be multiplied");

  int tile = BLOCK_SIZE;
  size_t memory = tile * tile * 2 * sizeof(float);
  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 numBlocks(cdiv(width, threadsPerBlock.x), cdiv(height, threadsPerBlock.y));
  auto output = torch::zeros({height, width}, mat1.options());

  matmul_shared_kernel<<<numBlocks, threadsPerBlock, memory>>>(
      mat1.data_ptr<float>(), mat2.data_ptr<float>(), output.data_ptr<float>(),
      height, width, inner);

  return output;
}

__global__ void matmul_kernel(
    float *mat1, float *mat2, float *output,
    int height, int width, int inner)
{
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;

  if (c >= width || r >= height)
    return;

  float o = 0.0;
  for (int i = 0; i < inner; ++i)
    o += mat1[r * inner + i] * mat2[i * width + c];

  output[r * width + c] = o;
}

torch::Tensor matmul(torch::Tensor mat1, torch::Tensor mat2)
{
  CHECK_INPUT(mat1);
  CHECK_INPUT(mat2);

  const int height = mat1.size(0);
  const int inner = mat1.size(1);
  const int width = mat2.size(1);
  TORCH_CHECK(inner == mat2.size(0), "mat1 and mat2 shapes cannot be multiplied");

  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 numBlocks(cdiv(width, threadsPerBlock.x), cdiv(height, threadsPerBlock.y));
  auto output = torch::zeros({height, width}, mat1.options());
  matmul_kernel<<<numBlocks, threadsPerBlock>>>(
      mat1.data_ptr<float>(),
      mat2.data_ptr<float>(),
      output.data_ptr<float>(),
      height, width, inner);
  return output;
}

PYBIND11_MODULE(custom_extension, m)
{
  m.def("matmul", &matmul);
}
