# Building Parthenon on Various Systems

*IMPORTANT: We try our best to keep the instructions up-to-date.
However, Parthenon itself, dependencies, and environments constantly changes so that the instruction may not work any more.
If you come across a disfunctional setup, please report it by open an issue or propose an updated description in a pull request*

## General list of cmake options:

   |           Option             | Default  | Type   | Description |
   | ---------------------------: | :------- | :----- | :---------- |
   |            PARTHENON\_NGHOST | 2        | String | Number of ghost cells |
   | PARTHENON\_SINGLE\_PRECISION | OFF      | Option | Enable single precision mode if requested |
   |     PARTHENON\_DISABLE\_HDF5 | OFF      | Option | HDF5 is enabled by default if found, set this to True to disable HDF5 |
   |      PARTHENON\_DISABLE\_MPI | OFF      | Option | MPI is enabled by default if found, set this to True to disable MPI |
   |   PARTHENON\_DISABLE\_OPENMP | OFF      | Option | OpenMP is enabled by default if found, set this to True to disable OpenMP |
   |   ENABLE\_COMPILER\_WARNINGS | OFF      | Option | Enable compiler warnings |
   |        TEST\_ERROR\_CHECKING | OFF      | Option | Enables the error checking unit test. This test will FAIL |
   |    TEST\_INTEL\_OPTIMIZATION | OFF      | Option | Test intel optimization and vectorization |
   |    CHECK\_REGISTRY\_PRESSURE | OFF      | Option | Check the registry pressure for Kokkos CUDA kernels |
   |               BUILD\_TESTING | ON       | Option | Multi-testing enablement |
   | PARTHENON\_DISABLE\_EXAMPLES | OFF      | Option | Toggle building of examples, if regression tests are on, drivers needed by the tests will still be built |
   |   ENABLE\_INTEGRATION\_TESTS | ${BUILD\_TESTING} | Option | Enable integration tests |
   |    ENABLE\_REGRESSION\_TESTS | ${BUILD\_TESTING} | Option | Enable regression tests |
   |      NUM\_MPI\_PROC\_TESTING | 4        | String | Number of MPI ranks used for MPI-enabled regression tests |
   |  REGRESSION\_GOLD\_STANDARD\_VER | #     | Int    | Version of current gold standard file used in regression tests. Default is set to latest version matching the source. |
   | REGRESSION\_GOLD\_STANDARD\_HASH | SHA512=... | String | Hash value of gold standard file to be downloaded. Used to ensure that the download is not corrupted. |
   | REGRESSION\_GOLD\_STANDARD\_SYNC | ON    | Option | Create `gold_standard` target to download gold standard files |
   |          ENABLE\_UNIT\_TESTS | ${BUILD\_TESTING} | Option | Enable unit tests |
   |               CODE\_COVERAGE | OFF      | Option | Builds with code coverage flags |
   |       CMAKE\_INSTALL\_PREFIX | machine specific | String | Optional path for library installation |
   |                 Kokkos\_ROOT | unset    | String | Path to a Kokkos source directory (containing CMakeLists.txt) |
   |  PARTHENON\_IMPORT\_KOKKOS | ON/OFF   | Option | If ON, attempt to link to an external Kokkos library. If OFF, build Kokkos from source and package with Parthenon |
   |          BUILD\_SHARED\_LIBS | OFF      | Option | If installing Parthenon, whether to build as shared rather than static |

### NB: CMake options prefixed with *PARTHENON\_* modify behavior.

## Installing Parthenon

An alternative to building Parthenon alongside a custom app (as in the examples)
is to first build Parthenon separately as a library and then link to it
when building the app. Parthenon can be built as either a static (default) or a shared library.

To build Parthenon as a library, provide a `CMAKE_INSTALL_PREFIX` path
to the desired install location to the Parthenon cmake call. To build a shared rather
than a static library, also set `BUILD_SHARED_LIBS=ON`. Then build and install
(note that `--build` and `--install` require CMake 3.15 or greater).

### Building as a static library

```bash
cmake -DCMAKE_INSTALL_PREFIX="$your_install_dir" $parthenon_source_dir
cmake --build . --parallel
cmake --install .
```

### Building as a shared library

```bash
cmake -DCMAKE_INSTALL_PREFIX="$your_install_dir" -DBUILD_SHARED_LIBS=ON $parthenon_source_dir
cmake --build . --parallel
cmake --install .
```

When building Parthenon, Kokkos will also be built from source if it exists in
`parthenon/external` or at a provided `Kokkos_ROOT` by default. If installing
Parthenon, this will also install Kokkos in the same directory. If
`PARTHENON_IMPORT_KOKKOS=ON` is provided or no Kokkos/CMakeLists.txt is found,
the build system will attempt to find a Kokkos installation in the current PATH.

A cmake target, `lib*/cmake/parthenon/parthenonConfig.cmake` is created during
installation. To link to parthenon, one can either specify the include files and
libraries directly or call `find_package(parthenon)` from cmake.

### Linking an app with *make*

The below example makefile can be used to compile the *calculate\_pi* example by
linking to a prior library installation of Parthenon. Note that library
flags must be appropriate for the Parthenon installation; it is not enough to
simply provide *-lparthenon*.

```bash
PARTHENON_INSTALL=/path/to/your/parthenon/install
KOKKOS_INSTALL=/path/to/your/Kokkos/install
CC=g++
CCFLAGS = -g -std=c++14 -L${PARTHENON_INSTALL}/lib \
 -I${PARTHENON_INSTALL}/include/ \
 -I${KOKKOS_INSTALL}/include/ -L${KOKKOS_INSTALL}/lib
LIB_FLAGS = -Wl,-rpath,${PARTHENON_INSTALL}/lib -lparthenon \
 -Wl,-rpath,${KOKKOS_INSTALL}/lib -lmpi -lkokkoscore -lhdf5 -ldl \
 -lkokkoscontainers -lz -lpthread -lgomp -lmpi_cxx
CC_COMPILE = $(CC) $(CCFLAGS) -c
CC_LOAD = $(CC) $(CCFLAGS)
.cpp.o:
  $(CC_COMPILE) $*.cpp
EXE = pi_example
all: $(EXE)
SRC = calculate_pi.cpp pi_driver.cpp
OBJ = calculate_pi.o pi_driver.o
INC = calculate_pi.hpp pi_driver.hpp
$(OBJ): $(INC) makefile
$(EXE): $(OBJ) $(INC) makefile
  $(CC_LOAD) $(OBJ) $(LIB_FLAGS) -o $(EXE)
clean:
  $(RM) $(OBJ) $(EXE)
```

### Linking an app with *cmake*
The below example `CMakeLists.txt` can be used to compile the *calculate_pi* example with a separate Parthenon installation through *cmake*'s `find_package()` routine.
```cmake
cmake_minimum_required(VERSION 3.11)

project(parthenon_linking_example)
set(Kokkos_CXX_STANDARD "c++14")
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(parthenon REQUIRED PATHS "/path/to/parthenon/install")
add_executable(
  pi-example
  pi_driver.cpp
  pi_driver.hpp
  calculate_pi.cpp
  calculate_pi.hpp
  )
target_link_libraries(pi-example PRIVATE Parthenon::parthenon)
```

## System specific instructions

Common first step: Obtain the Parthenon source including external dependencies (mostly Kokkos)

```bash
# Clone parthenon, with submodules
git clone --recursive https://github.com/lanl/parthenon.git
export PARTHENON_ROOT=$(pwd)/parthenon
```
We set the latter variable for easier reference in out-of-source builds.

### Default machine configurations

To make the default configuration on widely used systems easier, Parthenon provides machine configuration files that contain default options.
Defaults options include, but are not limited to setting
- the compiler (e.g., `nvcc_wrapper` for Cuda builds), or
- paths to non default package locations (e.g., for a custom HDF5 install), or 
- custom MPI related commands used in the Parthenon test suite (e.g., the launch command).

The machine configurations shipped with Parthenon are located in [`PARTHENON_ROOT/cmake/machinecfg`](../cmake/machinecfg) and are named by the machine name.
In order to use them either
- set the `MACHINE_CFG` environment variable to the appropriate file, or
- set the `MACHINE_CFG` CMake variable to the appropriate file.
In addition, you can set the `MACHINE_VARIANT` CMake variable to pick a specific configuration, e.g., one with Cuda and MPI enabled.

We suggest to inspect the corresponding file for available options on a specific machine.

In general, a typical workflow is expected to create your own machine file, e.g., on your develop system.
We suggest to start with a copy of a machine file that matches closely with your target machine.
Custom machine files should not be pushed to the main repository.

### Ubuntu 20.04 LTS

The following procedure has been tested for an Ubuntu 20.04 LTS system:

```bash
# install dependencies
# openmpi is installed implicitly by the hdf5 install
sudo apt-get update
install cmake build-essentials libhdf5-openmpi-dev

# make a bin directory
mkdir bin
cd bin
# configure and build
cmake ..
cmake -j --build .
# run unit and regression tests
ctest -LE performance
# run performance tests
ctest -L performance
```

### OLCF Summit (Power9+Volta)

Last verified 28 Aug 2020.

#### Common environment

```bash
# setup environment
$ module restore system
$ module load cuda gcc cmake/3.14.2 python hdf5

# on 28 Aug 2020 that results the following version
$ module list

Currently Loaded Modules:
  1) hsi/5.0.2.p5    4) darshan-runtime/3.1.7   7) gcc/6.4.0                       10) hdf5/1.10.4
  2) xalt/1.2.0      5) DefApps                 8) cmake/3.14.2                    11) python/3.6.6-anaconda3-5.3.0
  3) lsf-tools/2.0   6) cuda/10.1.243           9) spectrum-mpi/10.3.1.2-20200121
```

### LLNL RZAnsel (Power9+Volta)

Last verified 02 Sept 2020.

#### Common environment

```bash
# setup environment
$ module restore system
$ module load cuda gcc/7.3.1 

# on 02 Sept 2020 that results the following version
$ module list

Currently Loaded Modules:
  1) StdEnv (S)   2) cuda/10.1.243   3) gcc/7.3.1   4) spectrum-mpi/rolling-release

  Where:
   S:  Module is Sticky, requires --force to unload or purge
```

#### Cuda with MPI

```bash
# configure and build. Make sure to build in an directory on the GPFS filesystem if you want to run the regression tests because the home directory is not writeable from the compute nodes (which will result in the regression tests failing)
$ mkdir build-cuda-mpi && cd build-cuda-mpi
# note that we do not specify the mpicxx wrapper in the following as cmake automatically extracts the required include and linker options
$ cmake -DPARTHENON_DISABLE_HDF5=On -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_OPENMP=True -DKokkos_ARCH_POWER9=True -DKokkos_ENABLE_CUDA=True -DKokkos_ARCH_VOLTA70=True -DCMAKE_CXX_COMPILER=${PWD}/../external/Kokkos/bin/nvcc_wrapper ..
$ make -j

# The following commands are exepected to be run within job (interactive or scheduled)

# Make sure that GPUs are assigned round robin to MPI processes
$ export KOKKOS_NUM_DEVICES=4

# run all MPI regression tests
$ ctest -L regression -LE mpi-no

# manually run a simulation (here using 1 node with 4 GPUs and 1 MPI processes per GPU and a total of 2 processes (ranks))
# note the `-M "-gpu"` which is required to enable Cuda aware MPI
# also note the `--kokkos-num-devices=1` that ensures that each process on a node uses a different GPU
$ jsrun -p 2 -g 1 -c 20 -M "-gpu" ./example/advection/advection-example -i ../example/advection/parthinput.advection parthenon/time/nlim=10 parthenon/mesh/nx1=128 parthenon/mesh/nx2=64 parthenon/mesh/nx3=64 parthenon/meshblock/nx1=32 parthenon/meshblock/nx2=32 parthenon/meshblock/nx3=32 --kokkos-num-devices=1 | tee 2.out
```

#### Cuda without MPI

```bash
# configure and build
$ mkdir build-cuda && cd build-cuda
$ cmake -DCMAKE_BUILD_TYPE=Release -DMACHINE_CFG=${PARTHENON_ROOT}/cmake/machinecfg/Summit.cma
ke -DMACHINE_VARIANT=cuda -DPARTHENON_DISABLE_MPI=On ${PARTHENON_ROOT}
$ make -j10

# run unit tests (assumes running within a job, e.g., via `bsub -W 1:30 -nnodes 1 -P PROJECTID -Is /bin/bash`)
# - jsrun is required as the test would otherwise be executed on the scheduler node rather than on a compute node
# - "off" is required as otherwise the implicit PAMI initialization would fail
$ jsrun -n 1 -g 1 --smpiargs="off" ctest -L unit

# run convergence test
$ jsrun -n 1 -g 1 --smpiargs="off" ctest -R regression_test:advection_performance
```

