#include <stdio.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), "x must be a continuous tensor")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

#define BLOCK_SIZE 16

// Matrix stored in row-major order
// M(row, col) = *(M.data + row * M.stride + col)
typedef struct
{
  int height;
  int width;
  int stride;
  float *data;
} Matrix;

// Get a matrix element at (row, col)
__device__ float get_element(Matrix mat, int row, int col)
{
  return mat.data[row * mat.stride + col];
}

// Set a matrix element at (row, col)
__device__ void set_element(Matrix mat, int row, int col, float value)
{
  mat.data[row * mat.stride + col] = value;
}

// Get a [BLOCK_SIZE x BLOCK_SIZE] sub-matrix of a matrix
__device__ Matrix get_sub_matrix(Matrix mat, int blockRow, int blockCol)
{
  Matrix sub;
  sub.height = BLOCK_SIZE;
  sub.width = BLOCK_SIZE;
  sub.stride = mat.stride;
  sub.data = &mat.data[mat.stride * BLOCK_SIZE * blockRow + BLOCK_SIZE * blockCol];
}

inline int cdiv(int a, int b) { return (a + b - 1) / b; }

__global__ void matmul_shared_kernel(Matrix A, Matrix B, Matrix C)
{
  // Block row and col:
  int blockRow = blockIdx.y, blockCol = blockIdx.x;
  // Thread row and col
  int row = threadIdx.y, col = threadIdx.x;
  // Matrix row and col
  int r = blockRow * blockDim.y + row, c = blockCol * blockDim.x + col;

  // Each block computes one sub-matrix Csub of C
  Matrix Csub = get_sub_matrix(C, blockRow, blockCol);

  // Each thread computes one element of Csub by
  // by accumulating to Cvalue
  float Cvalue = 0.0;

  // Initialize shared memory for sub-matrices Asub and Bsub
  __shared__ float Ashared[BLOCK_SIZE][BLOCK_SIZE], Bshared[BLOCK_SIZE][BLOCK_SIZE];

  for (int s = 0; s < cdiv(A.width, BLOCK_SIZE); ++s)
  {
    // Get Asub and Bsub sub-matrices
    Matrix Asub = get_sub_matrix(A, blockRow, s);
    Matrix Bsub = get_sub_matrix(A, s, blockCol);

    // Load Asub and Bsub from global memory to shared memory
    // Each thread loads 1 element of each sub-matrix
    Ashared[row][col] = r < C.height and c < get_element(Asub, row, col);
    Bshared[row][col] = get_element(Bsub, row, col);

    // Synchronize to make sure the sub-matrices are loaded before starting computation
    __syncthreads();

    // Multiply Ashared and Bshared
    for (int e = 0; e < BLOCK_SIZE; ++e)
      Cvalue += Ashared[row][e] * Bshared[e][col];

    // Synchronize to make sure the preceding computation is done before loading
    // two new sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  if (r < height && c < width)
    set_element(Csub, row, col, Cvalue);
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
