
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

// This is a simple example that uses par_for() to compute whether cell
// centers sit within a unit sphere or not.  Adding up all the
// cells that lie within a unit sphere gives us a way to compute pi.
//
// Note: The goal is to use different methods of iterating through
// mesh blocks to see what works best for different architectures.
// While this code could be sped up by checking the innermost and
// outermost points of a mesh block, that would defeat the purpose of
// this program, so please do not make that change.
//
// Since the mesh infrastructure is not yet usable on GPUs, we create
// mesh blocks and chain them manually.  The cell coordinates are
// computed based on the origin of the mesh block and given cell
// sizes.  Once we have a canonical method of using a mesh on the GPU,
// this code will be changed to reflect that.
//
// Usage: examples/kokkos_pi/kokkos-pi N_Block N_Mesh N_iter
//          N_Block = size of each mesh block on each edge
//           N_Mesh = Number of mesh blocks along each axis
//           N_Iter = Number of timing iterations to run
//         [Radius] = Optional: Radius of sphere (size of cube).
//                    Defaults to 1.0
//
// The unit sphere is actually a unit octant that sits within a unit
// square which runs from (0,0,0) to (1,1,1).  Hence, in the perfect
// case, the sum of the interior would be pi/6. Our unit cube has
// N_Mesh*N_Block cells that span from [0,1] which gives us a
// dimension of 1.0/(N_Mesh*N_Block) for each side of the cell and the
// rest can be computed accordingly.  The coordinates of each cell
// within the block can be computed as:
//      (x0+dx*i_grid,y0+dy*j_grid,z0+dx*k_grid).
//
// We plan to explore using a flat range and a MDRange within par_for
// and using flat range and MDRange in Kokkos
//

#include <stdio.h>

#include "Kokkos_Core.hpp"
#include "cudaProfiler.h"
#include <iostream>
#include <string>
#include <vector>

// Get most commonly used parthenon package includes
#include "parthenon/package.hpp"
using namespace parthenon::package::prelude;
// using parthenon::LoopPatternMDRange;
// using parthenon::LoopPatternFlatRange;
// using parthenon::LoopPatternSimdFor;

using View2D = Kokkos::View<Real **, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>;

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

// simple giga-ops calculator
static double calcGops(const int &nops, const double &t, const int &n_block3,
                       const int &n_mesh3, const int &n_iter) {
  return (static_cast<Real>(nops * n_iter) / t / 1.0e9 * static_cast<Real>(n_block3) *
          static_cast<Real>(n_mesh3));
}

// Test wrapper to run a function multiple times
template <typename PerfFunc>
static double kernel_timer_wrapper(const int n_burn, const int n_perf,
                                   PerfFunc perf_func) {
  // Initialize the timer and test
  Kokkos::Timer timer;
  Kokkos::fence();
  timer.reset();
  //  cuProfilerStart();
  for (int i_run = 0; i_run < n_burn + n_perf; i_run++) {
    if (i_run == n_burn) {
      // Burn in time is over, start timing
      Kokkos::fence();
      timer.reset();
    }

    // Run the function timing performance
    perf_func();
  }

  // Time it
  Kokkos::fence();
  double perf_time = timer.seconds();
  //  cuProfilerStop();

  return perf_time;
}

static void usage(std::string program) {
  std::cout << std::endl
            << "    Usage: " << program << " n_block n_mesh n_iter" << std::endl
            << std::endl
            << "             n_block = size of each mesh block on each axis" << std::endl
            << "              n_mesh = number mesh blocks along each axis" << std::endl
            << "              n_iter = number of iterations to time" << std::endl
            << "            [Radius] = Optional: Radius of sphere" << std::endl
            << "                                 Defaults to 1.0" << std::endl
            << std::endl;
}

static double sumArray(MeshBlock *firstBlock, const int &n_block) {
  // This policy is over one block
  const int n_block2 = n_block * n_block;
  const int n_block3 = n_block * n_block * n_block;
  auto policyBlock = Kokkos::RangePolicy<>(Kokkos::DefaultExecutionSpace(), 0, n_block3,
                                           Kokkos::ChunkSize(512));
  double theSum = 0.0;
  // reduce the sum on the device
  // I'm pretty sure I can do this better, but not worried about performance for this
  MeshBlock *pmb = firstBlock;
  while (pmb) {
    Container<Real> &base = pmb->real_containers.Get();
    auto inOrOut = base.PackVariables({Metadata::Independent});
    double oneSum = 0.0;
    Kokkos::parallel_reduce(
        "sumArrayReduce", policyBlock,
        KOKKOS_LAMBDA(const int &idx, double &mySum) {
          const int k_grid = idx / n_block2;
          const int j_grid = (idx - k_grid * n_block2) / n_block;
          const int i_grid = idx - k_grid * n_block2 - j_grid * n_block;
          mySum += inOrOut(0, k_grid + NGHOST, j_grid + NGHOST, i_grid + NGHOST);
        },
        oneSum);
    Kokkos::fence();
    theSum += oneSum;
    pmb = pmb->next;
  }
  // calculate Pi
  return theSum;
}

static MeshBlock *setupMesh(const int &n_block, const int &n_mesh, const double &radius,
                            View2D &xyz, const int NG = 0) {
  // *** Kludge warning ***
  // Since our mesh is not GPU friendly we set up a hacked up
  // collection of mesh blocks.  The hope is that when our mesh is
  // up to par we will replace this code with the mesh
  // infrastructure.

  const Real dxyzCell = radius / static_cast<Real>(n_mesh * n_block);
  auto h_xyz = Kokkos::create_mirror_view(xyz);

  // Set up our mesh.
  Metadata myMetadata({Metadata::Independent, Metadata::Cell});
  MeshBlock *firstBlock = nullptr;
  MeshBlock *lastBlock = nullptr;

  // compute an offset due to ghost cells
  double delta = dxyzCell * static_cast<Real>(NG);

  int idx = 0; // an index into Block coordinate array
  for (int k_mesh = 0; k_mesh < n_mesh; k_mesh++) {
    for (int j_mesh = 0; j_mesh < n_mesh; j_mesh++) {
      for (int i_mesh = 0; i_mesh < n_mesh; i_mesh++, idx++) {
        // get a new meshblock and insert into chain
        auto *pmb = new MeshBlock(n_block, 3);
        if (lastBlock) {
          lastBlock->next = pmb;
          pmb->prev = lastBlock;
        } else {
          firstBlock = pmb;
        }
        // set coordinates of first cell center
        h_xyz(0, idx) = dxyzCell * (static_cast<Real>(i_mesh * n_block) + 0.5) - delta;
        h_xyz(1, idx) = dxyzCell * (static_cast<Real>(j_mesh * n_block) + 0.5) - delta;
        h_xyz(2, idx) = dxyzCell * (static_cast<Real>(k_mesh * n_block) + 0.5) - delta;
        // Add variable for in_or_out
        Container<Real> &base = pmb->real_containers.Get();
        base.setBlock(pmb);
        base.Add("in_or_out", myMetadata);
        // repoint lastBlock for next iteration
        lastBlock = pmb;
      }
    }
  }
  // copy our coordinates over to Device and wait for completion
  Kokkos::deep_copy(xyz, h_xyz);
  Kokkos::fence();

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

#define NLOOP 1

template <typename T>
KOKKOS_INLINE_FUNCTION void doit(const int &idx, const int &iMesh, const int &n_block,
                                 const int &n_block2, const Real &radius2,
                                 const Real &dxyzCell, const View2D &xyz, T &inOrOut) {
  const int k_grid = idx / n_block2;                                   // iops = 1
  const int j_grid = (idx - k_grid * n_block2) / n_block;              // iops = 3
  const int i_grid = idx - k_grid * n_block2 - j_grid * n_block;       // iops = 4
  const Real x = xyz(0, iMesh) + dxyzCell * static_cast<Real>(i_grid); // fops = 2
  const Real y = xyz(1, iMesh) + dxyzCell * static_cast<Real>(j_grid); // fops = 2
  const Real z = xyz(2, iMesh) + dxyzCell * static_cast<Real>(k_grid); // fops = 2
#if (NLOOP > 1)
  for (int i = 0; i < NLOOP; i++) {
    const Real myR2 = x * x + y * y + z * z + static_cast<Real>(i); // fops = NLOOP*6
    inOrOut(0, k_grid + NGHOST, j_grid + NGHOST, i_grid + NGHOST) =
        (myR2 < radius2 + static_cast<Real>(i) ? 1.0
                                               : 0.0); // fops = NLOOP*1 iops = NLOOP*3
  }
#else
  const Real myR2 = x * x + y * y + z * z; // fops = 5
  inOrOut(0, k_grid + NGHOST, j_grid + NGHOST, i_grid + NGHOST) =
      (myR2 < radius2 ? 1.0 : 0.0); // fops = 0 iops = 3
#endif
}

result_t naiveKokkos(int n_block, int n_mesh, int n_iter, double radius) {
  // creates a mesh and runs a basic Kokkos implementation for looping through blocks.

  // Setup auxilliary variables
  const int n_block2 = n_block * n_block;
  const int n_block3 = n_block * n_block * n_block;
  const int n_mesh3 = n_mesh * n_mesh * n_mesh;
  const double radius2 = radius * radius;
  const double radius3 = radius * radius * radius;
  const Real dxyzCell = radius / static_cast<Real>(n_mesh * n_block);
  const Real dVol = radius3 / static_cast<Real>(n_mesh3) / static_cast<Real>(n_block3);

  // allocate space for origin coordinates and set up the mesh
  View2D xyz("xyzBlocks", 3, n_mesh3);
  MeshBlock *firstBlock = setupMesh(n_block, n_mesh, radius, xyz);

  // first A  naive Kokkos loop over the mesh
  // This policy is over one block
  auto policyBlock = Kokkos::RangePolicy<>(Kokkos::DefaultExecutionSpace(), 0, n_block3,
                                           Kokkos::ChunkSize(512));

  MeshBlock *pStart = firstBlock;
  double time_basic = kernel_timer_wrapper(0, n_iter, [&]() {
    MeshBlock *pmb = pStart;
    for (int iMesh = 0; iMesh < n_mesh3; iMesh++, pmb = pmb->next) {
      Container<Real> &base = pmb->real_containers.Get();
      auto inOrOut = base.PackVariables({Metadata::Independent});
      // iops = 8  fops = 11
      Kokkos::parallel_for(
          "Compute In Or Out", policyBlock, KOKKOS_LAMBDA(const int &idx) {
            doit(idx, iMesh, n_block, n_block2, radius2, dxyzCell, xyz, inOrOut);
          });
    }
  });
  Kokkos::fence();

  // formulate result struct
  constexpr int niops = 11;
#if NLOOP > 1
  constexpr int nfops = 6 + NLOOP * 7;
#else
  constexpr int nfops = 11;
#endif
  auto r = result_t{"Naive_Kokkos",
                    (6.0 * sumArray(firstBlock, n_block) * dVol / radius3),
                    time_basic,
                    niops,
                    nfops,
                    0,
                    1};

  // Clean up the mesh
  deleteMesh(firstBlock);

  return r;
}

result_t cudaStream(int n_block, int n_mesh, int n_iter, double radius) {
  // creates a mesh and runs a 3D Kokkos implementation for looping through blocks.
  // Setup auxilliary variables
  const int n_block2 = n_block * n_block;
  const int n_block3 = n_block * n_block * n_block;
  const int n_mesh3 = n_mesh * n_mesh * n_mesh;
  const double radius2 = radius * radius;
  const double radius3 = radius * radius * radius;
  const Real dxyzCell = radius / static_cast<Real>(n_mesh * n_block);
  const Real dVol = radius3 / static_cast<Real>(n_mesh3) / static_cast<Real>(n_block3);
  constexpr int NSTREAMS = 8;

  cudaStream_t s[NSTREAMS];
  Kokkos::Cuda c[NSTREAMS];
  for (int i = 0; i < NSTREAMS; i++) {
    cudaStreamCreate(&(s[i]));
    c[i] = Kokkos::Cuda(s[i]);
  }
  // allocate space for origin coordinates and set up the mesh
  View2D xyz("xyzBlocks", 3, n_mesh3);
  MeshBlock *firstBlock = setupMesh(n_block, n_mesh, radius, xyz);

  // first A  naive Kokkos loop over the mesh
  // This policy is over one block
  auto policyBlock = Kokkos::RangePolicy<>(Kokkos::DefaultExecutionSpace(), 0, n_block3,
                                           Kokkos::ChunkSize(512));
  //  auto p0 = Kokkos::RangePolicy<Kokkos::Cuda>(cuda0, n_block3);

  MeshBlock *pStart = firstBlock;
  double time_basic = kernel_timer_wrapper(0, n_iter, [&]() {
    MeshBlock *pmb = pStart;
    int iMeshLimit = n_mesh3 - n_mesh3 % NSTREAMS;
    for (int iMesh = 0; iMesh < iMeshLimit;) {
      for (int iStream = 0; iStream < NSTREAMS; iStream++, iMesh++, pmb = pmb->next) {
        Container<Real> &base = pmb->real_containers.Get();
        auto inOrOut = base.PackVariables({Metadata::Independent});
        // iops = 11  fops = 11
        Kokkos::parallel_for(
            "Compute In Or Out",
            Kokkos::RangePolicy<Kokkos::Cuda>(c[iStream], 0, n_block3),
            KOKKOS_LAMBDA(const int &idx) {
              doit(idx, iMesh, n_block, n_block2, radius2, dxyzCell, xyz, inOrOut);
            });
      }
    }
    for (int iStream = 0, iMesh = iMeshLimit; iMesh < n_mesh3;
         iStream++, iMesh++, pmb = pmb->next) {
      Container<Real> &base = pmb->real_containers.Get();
      auto inOrOut = base.PackVariables({Metadata::Independent});
      // iops = 8  fops = 11
      Kokkos::parallel_for(
          "Compute In Or Out", Kokkos::RangePolicy<Kokkos::Cuda>(c[iStream], 0, n_block3),
          KOKKOS_LAMBDA(const int &idx) {
            doit(idx, iMesh, n_block, n_block2, radius2, dxyzCell, xyz, inOrOut);
          });
    }
  });
  Kokkos::fence();

  // formulate result struct
  constexpr int niops = 11;
#if NLOOP > 1
  constexpr int nfops = 6 + NLOOP * 7;
#else
  constexpr int nfops = 11;
#endif
  auto r = result_t{"CUDA_Streams",
                    (6.0 * sumArray(firstBlock, n_block) * dVol / radius3),
                    time_basic,
                    niops,
                    nfops,
                    0,
                    1};

  // Clean up the mesh
  deleteMesh(firstBlock);

  return r;
}

result_t naiveParFor(int n_block, int n_mesh, int n_iter, double radius) {
  // creates a mesh and runs a basic par_for implementation for looping through blocks.

  // Setup auxilliary variables
  const int n_block3 = n_block * n_block * n_block;
  const int n_mesh3 = n_mesh * n_mesh * n_mesh;
  const double radius2 = radius * radius;
  const double radius3 = radius * radius * radius;
  const Real dxyzCell = radius / static_cast<Real>(n_mesh * n_block);
  const Real dVol = radius3 / static_cast<Real>(n_mesh3) / static_cast<Real>(n_block3);

  // allocate space for origin coordinates and set up the mesh
  View2D xyz("xyzBlocks", 3, n_mesh3);
  MeshBlock *firstBlock = setupMesh(n_block, n_mesh, radius, xyz, NGHOST);

  MeshBlock *pStart = firstBlock;
  double time_basic = kernel_timer_wrapper(0, n_iter, [&]() {
    MeshBlock *pmb = pStart;
    for (int iMesh = 0; iMesh < n_mesh3; iMesh++, pmb = pmb->next) {
      Container<Real> &base = pmb->real_containers.Get();
      auto inOrOut = base.PackVariables({Metadata::Independent});
      // iops = 0  fops = 11
      par_for(
          "par_for in or out", DevExecSpace(), 0, inOrOut.GetDim(4) - 1, NGHOST,
          inOrOut.GetDim(3) - NGHOST - 1, NGHOST, inOrOut.GetDim(2) - NGHOST - 1, NGHOST,
          inOrOut.GetDim(1) - NGHOST - 1,
          KOKKOS_LAMBDA(const int l, const int k_grid, const int j_grid,
                        const int i_grid) {
            const Real x =
                xyz(0, iMesh) + dxyzCell * static_cast<Real>(i_grid); // fops = 2
            const Real y =
                xyz(1, iMesh) + dxyzCell * static_cast<Real>(j_grid); // fops = 2
            const Real z =
                xyz(2, iMesh) + dxyzCell * static_cast<Real>(k_grid); // fops = 2

#if NLOOP > 1
            for (int i = 0; i < NLOOP; i++) {
              const Real myR2 =
                  x * x + y * y + z * z + static_cast<Real>(i); // fops = NLOOP*6
              inOrOut(0, k_grid, j_grid, i_grid) =
                  (myR2 < radius2 + static_cast<Real>(i)
                       ? 1.0
                       : 0.0); // fops = NLOOP*1 iops = NLOOP*3
            }
#else
	    const Real myR2 = x * x + y * y + z * z;                  // fops = 5
	    inOrOut(l, k_grid, j_grid, i_grid) = (myR2 < radius2 ? 1.0 : 0.0);
#endif
          });
    }
  });
  Kokkos::fence();

  // formulate result struct
  constexpr int niops = 11;
#if NLOOP > 1
  constexpr int nfops = 6 + NLOOP * 7;
#else
  constexpr int nfops = 11;
#endif
  auto r = result_t{"Naive_ParFor",
                    (6.0 * sumArray(firstBlock, n_block) * dVol / radius3),
                    time_basic,
                    niops,
                    nfops,
                    0,
                    1};

  // Clean up the mesh
  deleteMesh(firstBlock);

  return r;
}

int main(int argc, char *argv[]) {
  //  cuProfilerStop();
  Kokkos::initialize(argc, argv);
  do {
    // ensure we have correct number of arguments
    if (!(argc == 4 || argc == 5)) {
      std::cout << "argc=" << argc << std::endl;
      usage(argv[0]);
      break;
    }

    std::size_t pos;
    Real radius = 1.0;

    // Read command line input
    const int n_block = std::stoi(argv[1], &pos);
    const int n_mesh = std::stoi(argv[2], &pos);
    const int n_iter = std::stoi(argv[3], &pos);
    if (argc >= 5) {
      radius = static_cast<Real>(std::stod(argv[4], &pos));
    } else {
      radius = 1.0;
    }

    // Run tests
    // A result vector
    std::vector<struct result_t> results;

    // discard first loop timing
    (void)naiveParFor(n_block, n_mesh, n_iter, radius);

    // Run Naive Kokkos Implementation
    results.push_back(cudaStream(n_block, n_mesh, n_iter, radius));
    results.push_back(naiveKokkos(n_block, n_mesh, n_iter, radius));
    results.push_back(naiveParFor(n_block, n_mesh, n_iter, radius));

    // print all results
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
      printf("|%s|%.8lf|%10g|%.4lf|%.4lf|%.14lf|\n", test.name.c_str(), test.t, cps,
             compRate, writeRate, test.pi);
    }

  } while (0);
  Kokkos::finalize();
}
