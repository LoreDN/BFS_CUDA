#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/time.h>
#include <time.h>
#include <vector>

#define MAX_FRONTIER_SIZE 128
#define DIM_GRID 1024
#define DIM_BLOCK 128

#define CHECK(call)                                                                 \
  {                                                                                 \
    const cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                       \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                           \
    }                                                                               \
  }

#define CHECK_KERNELCALL()                                                          \
  {                                                                                 \
    const cudaError_t err = cudaGetLastError();                                     \
    if (err != cudaSuccess) {                                                       \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                           \
    }                                                                               \
  }

void read_matrix(std::vector<int> &row_ptr,
                 std::vector<int> &col_ind,
                 std::vector<float> &values,
                 const std::string &filename,
                 int &num_rows,
                 int &num_cols,
                 int &num_vals);

__device__ void insertIntoFrontier(int val, int *frontier, int *frontier_size) {
  frontier[*frontier_size] = val;
  *frontier_size           = *frontier_size + 1;
}

__device__ inline void swap(int **ptr1, int **ptr2) {
  int *tmp = *ptr1;
  *ptr1    = *ptr2;
  *ptr2    = tmp;
}


// BFS algorithm optimized for GPU
__global__ void BFS_gpu(const int *source_ptr, const int *rowPointers, const int *destinations, int *distances)
{
  // initialize frontiers
  __shared__ int currentFrontier[MAX_FRONTIER_SIZE];
  __shared__ int currentFrontierSize;
  __shared__ int previousFrontier[MAX_FRONTIER_SIZE];
  __shared__ int previousFrontierSize;
  
  // initialize block's previous frontier from source
  if (threadIdx.x == 0)
  {
    currentFrontierSize = 0;
    previousFrontierSize = 0;
    const int source = *source_ptr;
    insertIntoFrontier(source, previousFrontier, &previousFrontierSize);
    distances[source] = 0;
  }

  __syncthreads();

  // BFS with parallel vertices
  while(previousFrontierSize > 0)       // while there are new vertices to visit
  {
    // visit all vertices on the previus frontier
    if(blockIdx.x < previousFrontierSize)
    {
      int currentVertex = previousFrontier[blockIdx.x];
      int row_start = rowPointers[currentVertex];
      int row_end = rowPointers[currentVertex + 1];

      // check all outgoing edges
      for(int row_i = row_start + threadIdx.x; row_i < row_end; row_i += DIM_BLOCK)      // parallelize over all outgoing edges even if they are more than the block size
      {
        if(distances[destinations[row_i]] == -1)
        {
          // this vertex has not been visited yet
          insertIntoFrontier(destinations[row_i], currentFrontier, &currentFrontierSize);
          distances[destinations[row_i]] = distances[currentVertex] +1;
        }
      }
    }
    
    // wait for all vertices to be visited
    __syncthreads();
    
    // swap to the next frontier
    if(threadIdx.x == 0)
    {
      swap((int**)&currentFrontier, (int**)&previousFrontier);
      previousFrontierSize = currentFrontierSize;
      currentFrontierSize  = 0;
    }

    // synchronize with the swap
    __syncthreads();

  }
}


int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: ./exec matrix_file source\n");
    return 0;
  }

  // host variables allocation
  std::vector<int> host_row_ptr;
  std::vector<int> host_col_ind;
  std::vector<float> values;
  int num_rows, num_cols, num_vals;

  const std::string filename{argv[1]};
  // The node starts from 1 but array starts from 0
  const int host_source = atoi(argv[2]) - 1;

  read_matrix(host_row_ptr, host_col_ind, values, filename, num_rows, num_cols, num_vals);

  // Initialize dist to -1
  std::vector<int> host_dist(num_vals);
  for (int i = 0; i < num_vals; i++) { host_dist[i] = -1; }

  // gpu variables allocation
  int *gpu_row_ptr;
  int *gpu_col_ind;
  int *gpu_dist;
  int *gpu_source;

  // gpu memory allocation
  CHECK(cudaMalloc(&gpu_source, sizeof(int)));
  CHECK(cudaMalloc(&gpu_row_ptr, host_row_ptr.size() * sizeof(int)));
  CHECK(cudaMalloc(&gpu_col_ind, host_col_ind.size() * sizeof(int)));
  CHECK(cudaMalloc(&gpu_dist, num_vals * sizeof(int)));

  // Copy data from host to device
  CHECK(cudaMemcpy(gpu_source, &host_source, sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(gpu_row_ptr, host_row_ptr.data(), host_row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(gpu_col_ind, host_col_ind.data(), host_col_ind.size() * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(gpu_dist, host_dist.data(), num_vals * sizeof(int), cudaMemcpyHostToDevice));

  // Call the gpu_kernel function
  BFS_gpu<<<DIM_GRID,DIM_BLOCK>>>(gpu_source, gpu_row_ptr, gpu_col_ind, gpu_dist);
  CHECK_KERNELCALL();

  // gpu memory free
  CHECK(cudaFree(gpu_source));
  CHECK(cudaFree(gpu_row_ptr));
  CHECK(cudaFree(gpu_col_ind));
  CHECK(cudaFree(gpu_dist));
  
  return EXIT_SUCCESS;
}

// Reads a sparse matrix and represents it using CSR (Compressed Sparse Row) format
void read_matrix(std::vector<int> &row_ptr,
                 std::vector<int> &col_ind,
                 std::vector<float> &values,
                 const std::string &filename,
                 int &num_rows,
                 int &num_cols,
                 int &num_vals) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "File cannot be opened!\n";
    throw std::runtime_error("File cannot be opened");
  }

  // Get number of rows, columns, and non-zero values
  file >> num_rows >> num_cols >> num_vals;

  row_ptr.resize(num_rows + 1);
  col_ind.resize(num_vals);
  values.resize(num_vals);

  // Collect occurrences of each row for determining the indices of row_ptr
  std::vector<int> row_occurrences(num_rows, 0);

  int row, column;
  float value;
  while (file >> row >> column >> value) {
    // Subtract 1 from row and column indices to match C format
    row--;
    column--;

    row_occurrences[row]++;
  }

  // Set row_ptr
  int index = 0;
  for (int i = 0; i < num_rows; i++) {
    row_ptr[i] = index;
    index += row_occurrences[i];
  }
  row_ptr[num_rows] = num_vals;

  // Reset the file stream to read again from the beginning
  file.clear();
  file.seekg(0, std::ios::beg);

  // Read the first line again to skip it
  file >> num_rows >> num_cols >> num_vals;

  std::fill(col_ind.begin(), col_ind.end(), -1);

  int i = 0;
  while (file >> row >> column >> value) {
    row--;
    column--;

    // Find the correct index (i + row_ptr[row]) using both row information and an index i
    while (col_ind[i + row_ptr[row]] != -1) { i++; }
    col_ind[i + row_ptr[row]] = column;
    values[i + row_ptr[row]]  = value;
    i                         = 0;
  }
}
