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

#include "parthenon_manager.hpp"

#include <string>
#include <utility>

#include <Kokkos_Core.hpp>

#include "driver/driver.hpp"
#include "globals.hpp"
#include "interface/update.hpp"
#include "mesh/domain.hpp"
#include "outputs/parthenon_hdf5.hpp"
#include "refinement/refinement.hpp"

namespace parthenon {

ParthenonStatus ParthenonManager::ParthenonInit(int argc, char *argv[]) {
  // initialize MPI
#ifdef MPI_PARALLEL
#ifdef OPENMP_PARALLEL
  int mpiprv;
  if (MPI_SUCCESS != MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpiprv)) {
    std::cout << "### FATAL ERROR in ParthenonInit" << std::endl
              << "MPI Initialization failed." << std::endl;
    return ParthenonStatus::error;
  }
  if (mpiprv != MPI_THREAD_MULTIPLE) {
    std::cout << "### FATAL ERROR in ParthenonInit" << std::endl
              << "MPI_THREAD_MULTIPLE must be supported for the hybrid parallelzation. "
              << MPI_THREAD_MULTIPLE << " : " << mpiprv << std::endl;
    // MPI_Finalize();
    return ParthenonStatus::error;
  }
#else  // no OpenMP
  if (MPI_SUCCESS != MPI_Init(&argc, &argv)) {
    std::cout << "### FATAL ERROR in ParthenonInit" << std::endl
              << "MPI Initialization failed." << std::endl;
    return ParthenonStatus::error;
  }
#endif // OPENMP_PARALLEL
  // Get process id (rank) in MPI_COMM_WORLD
  if (MPI_SUCCESS != MPI_Comm_rank(MPI_COMM_WORLD, &(Globals::my_rank))) {
    std::cout << "### FATAL ERROR in ParthenonInit" << std::endl
              << "MPI_Comm_rank failed." << std::endl;
    // MPI_Finalize();
    return ParthenonStatus::error;
  }

  // Get total number of MPI processes (ranks)
  if (MPI_SUCCESS != MPI_Comm_size(MPI_COMM_WORLD, &Globals::nranks)) {
    std::cout << "### FATAL ERROR in main" << std::endl
              << "MPI_Comm_size failed." << std::endl;
    // MPI_Finalize();
    return ParthenonStatus::error;
  }
#else  // no MPI
  Globals::my_rank = 0;
  Globals::nranks = 1;
#endif // MPI_PARALLEL

  Kokkos::initialize(argc, argv);

  // parse the input arguments
  ArgStatus arg_status = arg.parse(argc, argv);
  if (arg_status == ArgStatus::error) {
    return ParthenonStatus::error;
  } else if (arg_status == ArgStatus::complete) {
    return ParthenonStatus::complete;
  }

  // Set up the signal handler
  SignalHandler::SignalHandlerInit();
  if (Globals::my_rank == 0 && arg.wtlim > 0) SignalHandler::SetWallTimeAlarm(arg.wtlim);

  // Populate the ParameterInput object
  if (arg.input_filename != nullptr) {
    pinput = std::make_unique<ParameterInput>(arg.input_filename);
  } else if (arg.res_flag != 0) {
    // Open restart file
    restartReader = std::make_unique<RestartReader>(arg.restart_filename);

    pinput = std::make_unique<ParameterInput>();
    
    // Load input stream
    std::string inputString = restartReader->ReadAttrString("Input", "File");
    std::istringstream is(inputString);
    pinput->LoadFromStream(is);

  }
  pinput->ModifyFromCmdline(argc, argv);

  // read in/set up application specific properties
  auto properties = ProcessProperties(pinput);
  // set up all the packages in the application
  auto packages = ProcessPackages(pinput);
  // always add the Refinement package
  packages["ParthenonRefinement"] = Refinement::Initialize(pinput.get());

  if (arg.res_flag == 0) {
    pmesh = std::make_unique<Mesh>(pinput.get(), properties, packages, arg.mesh_flag);
  } else {
    // Read simulation time and cycle from restart file and set in input
    Real tNow = restartReader->GetAttr<Real>("Info", "Time");
    pinput->SetPrecise("parthenon/time", "start_time", tNow);

    Real dt = restartReader->GetAttr<Real>("Info", "dt");
    pinput->SetPrecise("parthenon/time", "dt", dt);

    int ncycle = restartReader->GetAttr<int32_t>("Info", "NCycle");
    pinput->SetInteger("parthenon/time", "ncycle", ncycle);

    // Read Mesh from restart file and create meshblocks
    pmesh = std::make_unique<Mesh>(pinput.get(), *restartReader, properties, packages);

    // Read package data from restart file
    RestartPackages(*pmesh, *restartReader);
  }

  // add root_level to all max_level
  for (auto const &ph : packages) {
    for (auto &amr : ph.second->amr_criteria) {
      amr->max_level += pmesh->GetRootLevel();
    }
  }

  SetFillDerivedFunctions();

  pmesh->Initialize(Restart(), pinput.get());

  ChangeRunDir(arg.prundir);

  return ParthenonStatus::ok;
}

ParthenonStatus ParthenonManager::ParthenonFinalize() {
  pmesh.reset();
  Kokkos::finalize();
#ifdef MPI_PARALLEL
  MPI_Finalize();
#endif
  return ParthenonStatus::complete;
}

void __attribute__((weak)) ParthenonManager::SetFillDerivedFunctions() {
  FillDerivedVariables::SetFillDerivedFunctions(nullptr, nullptr);
}

Properties_t __attribute__((weak))
ParthenonManager::ProcessProperties(std::unique_ptr<ParameterInput> &pin) {
  // In practice, this function should almost always be replaced by a version
  // that sets relevant things for the application.
  Properties_t props;
  return props;
}

Packages_t __attribute__((weak))
ParthenonManager::ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  // In practice, this function should almost always be replaced by a version
  // that sets relevant things for the application.
  Packages_t packages;
  return packages;
}

void ParthenonManager::RestartPackages(Mesh &rm, RestartReader &resfile) {
  // Restart packages with information for blocks in ids from the restart file
  // Assumption: blocks are contiguous in restart file, may have to revisit this.
  const IndexDomain interior = IndexDomain::interior;
  auto &packages = rm.packages;
  // Get block list and temp array size
  int nb = rm.GetNumMeshBlocksThisRank(Globals::my_rank);
  int nbs = rm.pblock->gid;
  int nbe = nbs + nb - 1;
  IndexRange myBlocks{nbs, nbe};

  // Get an iterator on block 0 for variable listing
  IndexRange out_ib = rm.pblock->cellbounds.GetBoundsI(interior);
  IndexRange out_jb = rm.pblock->cellbounds.GetBoundsJ(interior);
  IndexRange out_kb = rm.pblock->cellbounds.GetBoundsK(interior);

  size_t nCells = static_cast<size_t>(out_ib.e - out_ib.s + 1) *
                  static_cast<size_t>(out_jb.e - out_jb.s + 1) *
                  static_cast<size_t>(out_kb.e - out_kb.s + 1);
  // Get list of variables, assumed same for all blocks
  auto ciX = ContainerIterator<Real>(
      rm.pblock->real_containers.Get(),
      {parthenon::Metadata::Independent, parthenon::Metadata::Restart}, true);

  // Allocate space based on largest vector
  hsize_t vlen = 1;
  for (auto &v : ciX.vars) {
    if (v->GetDim(4) > vlen) {
      vlen = v->GetDim(4);
    }
  }
  Real *tmp = new Real[static_cast<size_t>(nb) * nCells * vlen];
  std::cout << "SIZES:" << nb << ":" << vlen << ":"
            << static_cast<size_t>(nb) * nCells * vlen << std::endl;
  for (auto &v : ciX.vars) {
    const hsize_t v4 = v->GetDim(4);
    const std::string vName = v->label();

    std::cout << "Var:" << vName << ":" << v4 << std::endl;
    // Read relevant data from the hdf file
    int stat = resfile.ReadBlocks(vName.c_str(), myBlocks, tmp, v4);
    if (stat < 0) {
      std::cout << " WARNING: Variable " << v->label() << " Not found in restart file";
      continue;
    }

    auto pmb = rm.pblock;
    hsize_t index = 0;
    while (pmb != nullptr) {
      // std::cout << pmb->gid << ":" << pmb->real_containers.Get() << std::endl;
      auto cX = ContainerIterator<Real>(
          pmb->real_containers.Get(),
          {parthenon::Metadata::Independent, parthenon::Metadata::Restart}, true);
      for (auto &v : cX.vars) {
        if (vName.compare(v->label()) == 0) {
          auto v_h = (*v).data.GetHostMirrorAndCopy();
          UNLOADVARIABLEONE(index, tmp, v_h, out_ib.s, out_ib.e, out_jb.s, out_jb.e,
                            out_kb.s, out_kb.e, v4);
          break;
        }
      }
      pmb = pmb->next;
    }
  }
  delete[] tmp;
}
} // namespace parthenon
