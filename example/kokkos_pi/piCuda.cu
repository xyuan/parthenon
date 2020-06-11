
//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================

// The goal of this program is to provide a mocked up mesh block using
// raw CUDA.  The mesh block here provides minimal functionality
// namely, Add() and Get() for variables and a variable class that has
// a host and device pointer.
//
// Three variants are provided here:
//   1: naive() Launches one kernel per meshBlock on default stream
//   2: stream() Launches one kernel per meshBlock on NSTREAMS streams
//   3: naiveOneKernel() Launches one kernel for entire mesh on default stream
//
// Metrics calculated are GFlops and Write GByte/s (since we only write, don't read)
// Results are printed in a markdown table.
//
// Compile time flags:
//   DO_UVM: uses UVM for allocation
//   DO_SELECTIVE_PROFILE: does profiling only around 'interesting' parts of code
//
// Below is a bash function to assist in compiling.  Set OPT=-g for debug code.

/**
 * A bash function to compile UVM and NO UVM versions of this file for the Volta
 **

    function nvcompile() {
        if [ -z "${OPT}" ] ; then export OPT=" -O3 -g"; fi;

        export nvFlags=" ${OPT}"
        nvFlags+=" -arch=sm_70"
        nvFlags+=" -gencode=arch=compute_70,code=sm_70"
        nvFlags+=" -gencode=arch=compute_70,code=compute_70"

        echo "nvcc -o a_uvm -DDO_UVM ${nvFlags}  ../example/kokkos_pi/piCuda.cu"
        nvcc -o a_uvm -DDO_UVM ${nvFlags}  ../example/kokkos_pi/piCuda.cu

        echo "nvcc -o a_nouvm  ${nvFlags}  ../example/kokkos_pi/piCuda.cu"
        nvcc -o a_nouvm  ${nvFlags}  ../example/kokkos_pi/piCuda.cu
    }

**/

#include <iostream>
#include <memory>
#include <vector>

#ifdef DO_SELECTIVE_PROFILE
#include "cudaProfiler.h"
#include "cuda_profiler_api.h"
#endif

#ifdef DO_UVM
#define DEVICE_MALLOC cudaMallocManaged
#else
#define DEVICE_MALLOC cudaMalloc
#endif

#ifndef NGHOST
#define NGHOST 2
#endif

#ifndef Real
typedef double Real;
#endif

// Simple cuda test.
// Plan is to:
// 1: allocate blocks of size nBlock + 2 * NGHOST on each side of which
// 2: mesh is nMesh * nMesh * nMesh blocks big
// 3: GPU: Compute InOrOut on each block
// 4: GPU not timed: Compute pi contribution for each block
// 5: GPU not timed: Compute pi

// global event timers
static cudaEvent_t start_, stop_;

// global streams
constexpr int NSTREAMS = 8;
cudaStream_t streams[NSTREAMS];

// global cuda launch parameters
// for only internal computation
static dim3 block_(1, 1, 1);
static dim3 grid_(1, 1, 1);

// for computations including ghosts
static dim3 blockAll_(1, 1, 1);
static dim3 gridAll_(1, 1, 1);

static __host__ Real calcGops(const int &nops, const Real &t, const int &n_block3,
                              const int &n_mesh3, const int &n_iter) {
  // a simple compute giga-ops function that takes into account block
  // size, mesh size, and number of iterations.
  Real nCopies = static_cast<Real>(n_mesh3) * static_cast<Real>(n_block3);
  return nCopies * static_cast<Real>(nops * n_iter) / t / 1.0e9;
}

__device__ int fromBlockKJI(const int &k, const int &j, const int &i, const int &nSide) {
  // returns global index from local (k,j,i)
  // iops = 8
  return i + NGHOST + (j + NGHOST) * nSide + (k + NGHOST) * nSide * nSide;
}

__device__ void toKJI(int &k, int &j, int &i, const int &idx, const int &nSide) {
  // computes mesh k,j,i from index using nSide as the length of each
  // of the axes.
  // iops = 9
  const int nSide2 = nSide * nSide;
  k = idx / nSide2;
  j = (idx - k * nSide2) / nSide;
  i = (idx - k * nSide2 - j * nSide);
  return;
}

__global__ void varZero(Real *ptr, const size_t N) {
  int cidx = blockIdx.x * blockDim.x + threadIdx.x;
  if (cidx < N) {
    ptr[cidx] = 0.0;
  }
}

class Variable {
 public:
  std::string name;
  size_t bytes;
  Real *device_ptr;
  Real *host_ptr;
  Variable(const std::string name_, const int &nTotal) : name(name_) {
    // compute size
    bytes = nTotal * sizeof(Real);
    // create variable
    DEVICE_MALLOC((void **)(&device_ptr), nTotal * sizeof(Real));
#ifdef DO_UVM
    host_ptr = device_ptr;
#else
    host_ptr = new Real[nTotal];
#endif
    // zero out variable
    cudaMemset(device_ptr, 0, bytes);
#ifndef DO_UVM
    memset(host_ptr, 0, bytes);
#endif
  }
  ~Variable() {
    if (device_ptr) cudaFree(device_ptr);
    device_ptr = nullptr;
#ifndef DO_UVM
    if (host_ptr) free(host_ptr);
    host_ptr = nullptr;
#endif
  }
  __host__ void zeroDevice() {
    varZero<<<gridAll_, blockAll_>>>(device_ptr, bytes / sizeof(Real));
  }

#ifdef DO_UVM
  // copy functions do nothing
  __host__ void ToHost() {}
  __host__ void ToHost(int istart, int iend) {}
  __host__ void ToDevice() {}
  __host__ void ToDevice(int istart, int iend) {}
#else
  // we need to transfer if not using UVM
  __host__ void ToHost() {
    // copy all data from device to host
    cudaMemcpy(host_ptr, device_ptr, bytes, cudaMemcpyDeviceToHost);
  }
  __host__ void ToHost(int istart, int iend) {
    // copy index range data from device to host
    const size_t myBytes = (iend - istart + 1) * sizeof(Real);
    cudaMemcpy(host_ptr + istart, device_ptr + istart, myBytes, cudaMemcpyDeviceToHost);
  }
  __host__ void ToDevice() {
    // copy all data from host to device
    cudaMemcpy(device_ptr, host_ptr, bytes, cudaMemcpyHostToDevice);
  }
  __host__ void ToDevice(int istart, int iend) {
    // copy index range data from host to device
    const size_t myBytes = (iend - istart + 1) * sizeof(Real);
    cudaMemcpy(device_ptr + istart, host_ptr + istart, myBytes, cudaMemcpyHostToDevice);
  }
#endif
};

class MeshBlock {
  // A minimal 3D mesh block with nBlock cells on each side.
 public:
  const int id;        // ID of current mesh block
  const int nBlock;    // Number of internal cells on each side of block
  const int nSide;     // Number of cells on each side of block
  const int iS;        // internal start
  const int iE;        // internal end
  const int tE;        // total end.  total start = 0
  const int nInternal; // Total number of 'true' or internal cells for block
  const int nTotal;    // Total number of cells including ghosts
  MeshBlock *next;     // pointer to next mesh block
  MeshBlock *prev;     // pointer to previous mesh block

  // Constructor, takes one argument: number of cells on each side of 3D block
  MeshBlock(const int id_, const int n_block)
      : id(id_), nBlock(n_block), nSide(n_block + 2 * NGHOST), iS(NGHOST),
        iE(n_block + NGHOST), tE(n_block + 2 * NGHOST),
        nInternal(n_block * n_block * n_block),
        nTotal((n_block + 2 * NGHOST) * (n_block + 2 * NGHOST) * (n_block + 2 * NGHOST)),
        next(nullptr) {
    myVars_.clear();
  }

  // Add an array to the block
  // do nothing if it exists
  void Add(std::string name) {
    // only cell based variables for now
    for (auto &var : myVars_) {
      if (var->name.compare(name) == 0) {
        // error print message
        printf("Variable already exists\n");
        return;
      }
    }
    // otherwise allocate and return
    myVars_.push_back(std::make_shared<Variable>(name, nTotal));
    return;
  }

  // Get an array from block container.
  // Allocate if it doesn't exist
  std::shared_ptr<Variable> Get(std::string name) {
    for (auto &var : myVars_) {
      if (var->name.compare(name) == 0) {
        return var;
      }
    }
    printf("Variable doesn't exists, adding\n");
    Add(name);

    // return the added variable
    return Get(name);
  }

 private:
  std::vector<std::shared_ptr<Variable>> myVars_;
};

MeshBlock *setupMesh(const int &n_block, const int &n_mesh) {
  // Set up our mesh.
  MeshBlock *firstBlock = nullptr;
  MeshBlock *lastBlock = nullptr;

  int idx = 0; // an index into Block coordinate array
  for (int k_mesh = 0; k_mesh < n_mesh; k_mesh++) {
    for (int j_mesh = 0; j_mesh < n_mesh; j_mesh++) {
      for (int i_mesh = 0; i_mesh < n_mesh; i_mesh++, idx++) {
        // get a new meshblock and insert into chain
        auto *pmb = new MeshBlock(idx, n_block);
        if (lastBlock) {
          lastBlock->next = pmb;
          pmb->prev = lastBlock;
        } else {
          firstBlock = pmb;
        }

        // Add variable for in_or_out
        pmb->Add(std::string("in_or_out"));

        // Reset lastBlock pointer and repeat
        lastBlock = pmb;
      }
    }
  }
  // ensure that we are done with all GPU ops
  cudaEventRecord(start_);
  cudaEventSynchronize(start_);
  return firstBlock;
}

void deleteMesh(MeshBlock *firstBlock) {
  // deletes the mesh chain starting with firstBlock
  auto pmb = firstBlock;
  while (pmb) {
    auto next = pmb->next;
    delete pmb;
    pmb = next;
  }
}

// The result struct contains results of different tests
typedef struct result_t {
  std::string name;  // The name of this test
  Real pi;           // the value of pi calculated
  Real t;            // time taken to run test
  int iops;          // number of integer ops
  int fops;          // number of floating point ops
  size_t bytesRead;  // number of bytes transferred
  size_t bytesWrite; // number of bytes transferred
} result_t;

__global__ void blockSum(int nBlock, Real *inOrOut, Real *pi) {
  // sums up block into variable pi
  // could use a reduce kernel here, but not performance critical
  // iops = 16;  fops = 1
  const int nSide = nBlock + 2 * NGHOST;                   // iops = 2
  const int kcell = blockIdx.z * blockDim.z + threadIdx.z; // iops = 2
  const int jcell = blockIdx.y * blockDim.y + threadIdx.y; // iops = 2
  const int icell = blockIdx.x * blockDim.x + threadIdx.x; // iops = 2
  int cidx = fromBlockKJI(kcell, jcell, icell, nSide);     // iops = 8
  atomicAdd(pi, inOrOut[cidx]);                            // fops = 1
}

Real getPi(const int nBlock, int nMesh, Real dxyz, MeshBlock *firstBlock) {
  // Computes the value of Pi from the inOrOut arrays in mesh blocks.
  Real pi;
  auto pmb = firstBlock;
  Real *d_pi;

  // allocate space for pi on GPU and zero it
  DEVICE_MALLOC((void **)(&d_pi), sizeof(Real));
  varZero<<<1, 1>>>(d_pi, 1);
  cudaDeviceSynchronize();

  // now sum the blocks
  pmb = firstBlock;
  while (pmb) {
    auto var = pmb->Get(std::string("in_or_out"));
    blockSum<<<grid_, block_>>>(nBlock, var->device_ptr, d_pi);
    pmb = pmb->next;
  }
  cudaDeviceSynchronize();

  // pull back value of pi
  cudaMemcpy(&pi, d_pi, sizeof(Real), cudaMemcpyDeviceToHost);

  // scale the sum
  pi = 6.0 * pi * dxyz * dxyz * dxyz;

  // return the value of pi computed
  return pi;
}

__global__ void calculateInOrOut(int nBlock, int nMesh, int blockID, Real dxyz,
                                 Real *inOrOut) {
  // Kernel to calculate if cell center is in or out of sphere
  // Called once  per block.
  // Metrics: iops = 31;  fops = 11; write = 1
  int kgrid, jgrid, igrid, cidx;
  const int kcell = blockIdx.z * blockDim.z + threadIdx.z; // iops = 2
  const int jcell = blockIdx.y * blockDim.y + threadIdx.y; // iops = 2
  const int icell = blockIdx.x * blockDim.x + threadIdx.x; // iops = 2

  cidx = fromBlockKJI(kcell, jcell, icell, nBlock + 2 * NGHOST); // iops = 8+2
  toKJI(kgrid, jgrid, igrid, blockID, nMesh);                    // iops = 9

  Real x = dxyz * (Real)(igrid * nBlock + icell + 0.5); // iops = 2 fops = 2
  Real y = dxyz * (Real)(jgrid * nBlock + jcell + 0.5); // iops = 2 fops = 2
  Real z = dxyz * (Real)(kgrid * nBlock + kcell + 0.5); // iops = 2 fops = 2
  Real r2 = x * x + y * y + z * z;                      // iops = 0 fops = 5

  inOrOut[cidx] = (r2 < 1.0 ? 1.0 : 0.0);
}

__global__ void oneKernel(const int nBlock, const int nMesh, Real dxyz,
                          Real **allVariables) {
  // Kernel to calculate if cell center is in or out of sphere
  // Called once  per mesh.
  // Metrics: iops = 36;  fops = 11; write = 1
  Real *inOrOut = allVariables[blockIdx.x];
  int kgrid, jgrid, igrid;
  int kcell, jcell, icell;

  // compute cell index
  int internalIndex = blockIdx.y * 512 + threadIdx.x;                      // iops = 2
  toKJI(kcell, jcell, icell, internalIndex, nBlock);                       // iops = 9
  const int cidx = fromBlockKJI(kcell, jcell, icell, nBlock + 2 * NGHOST); // iops = 8+2

  // compute grid index
  toKJI(kgrid, jgrid, igrid, blockIdx.x, nMesh); // iops = 9

  // compute radius
  Real x = dxyz * (Real)(igrid * nBlock + icell + 0.5); // iops = 2 fops = 2
  Real y = dxyz * (Real)(jgrid * nBlock + jcell + 0.5); // iops = 2 fops = 2
  Real z = dxyz * (Real)(kgrid * nBlock + kcell + 0.5); // iops = 2 fops = 2
  Real r2 = x * x + y * y + z * z;                      // iops = 0 fops = 5

  inOrOut[cidx] = (r2 < 1.0 ? 1.0 : 0.0);
}

__global__ void naiveInOrOut(int nBlock, int nMesh, int blockID, Real dxyz,
                             Real *inOrOut) {
  // Note: Identical to calculateInOrOut above.  Called a different name here
  //       so that I can differentiate it in the profiler

  // Kernel to calculate if cell center is in or out of sphere
  // Called once  per block.
  // Metrics: iops = 31;  fops = 11; write = 1
  int kgrid, jgrid, igrid, cidx;
  const int kcell = blockIdx.z * blockDim.z + threadIdx.z; // iops = 2
  const int jcell = blockIdx.y * blockDim.y + threadIdx.y; // iops = 2
  const int icell = blockIdx.x * blockDim.x + threadIdx.x; // iops = 2

  cidx = fromBlockKJI(kcell, jcell, icell, nBlock + 2 * NGHOST); // iops = 8+2
  toKJI(kgrid, jgrid, igrid, blockID, nMesh);                    // iops = 9

  Real x = dxyz * (Real)(igrid * nBlock + icell + 0.5); // iops = 2 fops = 2
  Real y = dxyz * (Real)(jgrid * nBlock + jcell + 0.5); // iops = 2 fops = 2
  Real z = dxyz * (Real)(kgrid * nBlock + kcell + 0.5); // iops = 2 fops = 2
  Real r2 = x * x + y * y + z * z;                      // iops = 0 fops = 5

  inOrOut[cidx] = (r2 < 1.0 ? 1.0 : 0.0);
}

result_t stream(const int n_block, const int n_mesh, const int n_iter,
                const char *name = "stream", MeshBlock *firstBlock_in = nullptr) {
  // Uses cuda streams to reduce latency.
  // Number of streams can be adjusted with NSTREAMS above

  // Create mesh if required
  MeshBlock *firstBlock;
  if (!firstBlock_in) {
    firstBlock = setupMesh(n_block, n_mesh);
  } else {
    firstBlock = firstBlock_in;
  }

  // Compute cell size
  Real dxyz = (Real)1.0 / (Real)(n_mesh * n_block);

  // Ensure all work is finished before timing
  cudaDeviceSynchronize();

#ifdef DO_SELECTIVE_PROFILE
  cudaProfilerStart();
#endif

  // Call kernel to compute In Or Out
  cudaEventRecord(start_, streams[0]);
  int iStream = 0;
  auto pmb = firstBlock;
  while (pmb) {
    if (iStream >= NSTREAMS) iStream = 0;
    auto var = pmb->Get("in_or_out");
    calculateInOrOut<<<grid_, block_, 0, streams[iStream]>>>(n_block, n_mesh, pmb->id,
                                                             dxyz, var->device_ptr);
    iStream = iStream + 1;
    pmb = pmb->next;
  }

  // Wait for all streams to finish
  for (auto &s : streams) {
    cudaStreamSynchronize(s);
  }

  // Finish timing
  cudaDeviceSynchronize();
  cudaEventRecord(stop_, streams[0]);
  cudaEventSynchronize(stop_);

#ifdef DO_SELECTIVE_PROFILE
  cudaProfilerStop();
#endif

  // Compute the result and return it
  result_t myResult;
  float milliseconds = 1.0;
  cudaEventElapsedTime(&milliseconds, start_, stop_);
  myResult.name = std::string(name);
  myResult.pi = getPi(n_block, n_mesh, dxyz, firstBlock);
  myResult.t = milliseconds * 0.001;
  myResult.iops = 31;
  myResult.fops = 11;
  myResult.bytesRead = 0;
  myResult.bytesWrite = 1;

  // clean up if required
  if (!firstBlock_in) {
    deleteMesh(firstBlock);
  }

  // return result
  return myResult;
}

result_t naive(const int n_block, const int n_mesh, const int n_iter,
               const char *name = "naive", MeshBlock *firstBlock_in = nullptr) {
  Real dxyz = (Real)1.0 / (Real)(n_mesh * n_block);
  // Uses a naive CUDA kernel on a per-block basis

  // create a mesh if required
  MeshBlock *firstBlock;
  if (!firstBlock_in) {
    firstBlock = setupMesh(n_block, n_mesh);
  } else {
    firstBlock = firstBlock_in;
  }

#ifdef DO_SELECTIVE_PROFILE
  cudaProfilerStart();
#endif

  // Ensure all work is finished before timing
  cudaDeviceSynchronize();

  // Use a cuda kernel to calculate in or out
  cudaEventRecord(start_);
  auto pmb = firstBlock;
  while (pmb) {
    auto var = pmb->Get("in_or_out");
    naiveInOrOut<<<grid_, block_>>>(n_block, n_mesh, pmb->id, dxyz, var->device_ptr);
    pmb = pmb->next;
  }

  // Wait for work to finish
  cudaDeviceSynchronize();
  cudaEventRecord(stop_);
  cudaEventSynchronize(stop_);

#ifdef DO_SELECTIVE_PROFILE
  cudaProfilerStop();
#endif

  // compute result
  result_t myResult;
  float milliseconds = 1.0;
  cudaEventElapsedTime(&milliseconds, start_, stop_);
  myResult.name = std::string(name);
  myResult.pi = getPi(n_block, n_mesh, dxyz, firstBlock);
  myResult.t = milliseconds * 0.001;
  myResult.iops = 31;
  myResult.fops = 11;
  myResult.bytesRead = 0;
  myResult.bytesWrite = 1;

  // clean up if required
  if (!firstBlock_in) deleteMesh(firstBlock);

  // return result
  return myResult;
}
result_t naiveOneKernel(const int n_block, const int n_mesh, const int n_iter,
                        const char *name = "naiveOneKernel",
                        MeshBlock *firstBlock_in = nullptr) {
  // Uses a single kernel launch for the entire mesh.

  // create a mesh if required

  MeshBlock *firstBlock;
  if (!firstBlock_in) {
    firstBlock = setupMesh(n_block, n_mesh);
  } else {
    firstBlock = firstBlock_in;
  }
  Real dxyz = (Real)1.0 / (Real)(n_mesh * n_block);

#ifdef DO_SELECTIVE_PROFILE
  cudaProfilerStart();
#endif

  // mark the stream and ensure all work is done before timing starts
  cudaEventRecord(start_);
  cudaEventSynchronize(start_);

  // start computing
  cudaEventRecord(start_);

  // allocate space to hold all pointers on host and device
  const int n_mesh3 = n_mesh * n_mesh * n_mesh;
  dim3 oneGrid(n_mesh3);

  int n3 = n_block * n_block * n_block;
  while (n3 > 512) {
    n3 = n3 / 2;
    oneGrid.y = 2 * oneGrid.y;
  }

  // allocate space on host
  double **allVars;
  double **d_allVars;

  // allocate the space on device and host to hold pointers
  DEVICE_MALLOC((void **)(&d_allVars), n_mesh3 * sizeof(Real *));
  allVars = (double **)calloc(n_mesh3, sizeof(Real *));

  //  collect up all the pointers on host
  auto pmb = firstBlock;
  for (int idx = 0; idx < n_mesh3; idx++) {
    auto var = pmb->Get("in_or_out");
    allVars[idx] = var->device_ptr;
    pmb = pmb->next;
  }

  // push the pointers to the device
  cudaMemcpy(d_allVars, allVars, n_mesh3 * sizeof(Real *), cudaMemcpyHostToDevice);

  // Run the kernel to compute in or out
  oneKernel<<<oneGrid, n3>>>(n_block, n_mesh, dxyz, d_allVars);

  // Release host and device memory.
  // Not that cudaFree is automatic synchronization
  free(allVars);
  cudaFree(d_allVars);

  cudaEventRecord(stop_);
  cudaEventSynchronize(stop_);

#ifdef DO_SELECTIVE_PROFILE
  cudaProfilerStop();
#endif

  // Return the result
  result_t myResult;
  float milliseconds = 1.0;
  cudaEventElapsedTime(&milliseconds, start_, stop_);
  myResult.name = std::string(name);
  myResult.pi = getPi(n_block, n_mesh, dxyz, firstBlock);
  myResult.t = milliseconds * 0.001;
  myResult.iops = 36;
  myResult.fops = 11;
  myResult.bytesRead = 0;
  myResult.bytesWrite = 1;

  // clean up if required
  if (!firstBlock_in) deleteMesh(firstBlock);

  // Return the result
  return myResult;
}

static void usage(std::string program) {
  // Minimal help on usage
  std::cout << std::endl
            << "    Usage: " << program << " n_block n_mesh " << std::endl
            << std::endl
            << "             n_block = size of each mesh block on each axis" << std::endl
            << "              n_mesh = number mesh blocks along each axis" << std::endl
            << std::endl;
}

void setLimits(int n, dim3 &myBlock, dim3 &myGrid) {
  // compute block and grid vectors used in cuda launches for a given
  // block size
  myGrid = dim3(1, 1, 1);
  while (n > 8) {
    n = n / 2;
    myGrid.x = myGrid.x * 2;
    myGrid.y = myGrid.y * 2;
    myGrid.z = myGrid.z * 2;
  }
  myBlock.x = myBlock.y = myBlock.z = n;
  return;
}

int main(int argc, char **argv) {

  // argument check
  if (argc != 3) {
    std::cout << "argc=" << argc << std::endl;
    usage(argv[0]);
    return 1;
  }

  // Read command line input
  std::size_t pos;
  const int n_block = std::stoi(argv[1], &pos);
  const int n_mesh = std::stoi(argv[2], &pos);
  const int n_iter = 1;

  // Ensure that n_block and n_mesh are powers of 2
  if ((n_block == 0) || (n_block & (n_block - 1)) || (n_mesh == 0) ||
      (n_mesh & (n_mesh - 1))) {
    std::cout << std::endl
              << "    ERROR: n_block & n_mesh must be non-zero powers of 2" << std::endl
              << std::endl;
    return 2;
  }

#ifdef DO_SELECTIVE_PROFILE
  cudaProfilerStop();
#endif

  // Create global timing events
  cudaEventCreate(&stop_);
  cudaEventCreate(&start_);

  // initialize stream
  for (auto &s : streams) {
    cudaStreamCreate(&s);
    // wait until stream created
    cudaEventRecord(start_, s);
    cudaEventSynchronize(start_);
  }

  // set iteration limits for 'real' and 'all' cells in a block
  setLimits(n_block, block_, grid_);
  setLimits((n_block + 2 * NGHOST), blockAll_, gridAll_);

  // creates a mesh that we can test reuse with - not currently used
  //  MeshBlock *reuseBlock = setupMesh(n_block, n_mesh);

  // Run tests and capture answers in a result vector
  // A result vector
  std::vector<struct result_t> results;

  // note that we don't appear to need to discard first run timing in
  // raw cuda but do it anyway.
  results.push_back(stream(n_block, n_mesh, n_iter, "discard"));
  results.push_back(stream(n_block, n_mesh, n_iter, "stream-1"));
  results.push_back(naive(n_block, n_mesh, n_iter, "naive-1"));
  results.push_back(naiveOneKernel(n_block, n_mesh, n_iter, "naiveOneKernelWholeMesh"));

  // The next four use a pre-defined block, which provides a bit more performance
  /* results.push_back( */
  /*     stream(n_block, n_mesh, n_iter, "reuse-stream-1", reuseBlock)); */
  /* results.push_back(naive(n_block, n_mesh, n_iter, "reuse-naive-1",
   * reuseBlock)); */
  /* results.push_back( */
  /*     stream(n_block, n_mesh, n_iter, "reuse-stream-2", reuseBlock)); */
  /* results.push_back(naive(n_block, n_mesh, n_iter, "reuse-naive-2",
   * reuseBlock)); */

  // print all results in a markdown friendlyt format
  const int64_t n_block3 = n_block * n_block * n_block;
  const int64_t n_mesh3 = n_mesh * n_mesh * n_mesh;

  const int64_t iterBlockMesh = static_cast<int64_t>(n_iter) *
                                static_cast<int64_t>(n_mesh3) *
                                static_cast<int64_t>(n_block3);
  /* Markdown print */
  printf("\n|name|t(s)|cps|GFlops|Write GB/s|π|\n");
  printf("|:---|:---|:---|:---|:---|:---|\n");
  for (auto &test : results) {
    Real cps = static_cast<Real>(iterBlockMesh) / test.t;
    Real compRate = calcGops(test.fops, test.t, n_block3, n_mesh3, n_iter);
    Real writeRate = calcGops(8 * test.bytesWrite, test.t, n_block3, n_mesh3, n_iter);
    printf("|%s|%.8lf|%10g|%.2lf|%.2lf|%.14lf|\n", test.name.c_str(), test.t, cps,
           compRate, writeRate, test.pi);
  }

  // release streams
  for (auto &s : streams) {
    cudaStreamDestroy(s);
  }

  // Destroy events we created
  cudaEventDestroy(stop_);
  cudaEventDestroy(start_);

  // release the mesh if needed
  //  if (reuseBlock) deleteMesh(reuseBlock);

  return 0;
}
