#=========================================================================================
# (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
#=========================================================================================

# Adding regression tests
#
# To add additional regression tests to ctest, three things must be added to 
# this file:
# 
# 1. there must exist within the regression/test_suites directory a folder associated with the test
#   * For the calculate_pi example the folder is called 'calculate_pi', This
#     folder must be added to the cmake variable TEST_DIRS as shown below. 
#
# 2. the correct arguments to pass to the 'run_test.py' script must be specfied
#   * This arguments must be added to the CMAKE variable TEST_ARGS as shown below
#
# 3. the number of mpi processors that should be used when testing the mpi implementation 
#   * This argument needs to be added to the TEST_PROCS variable
#
# NOTE: each new test must append exactly 1 item to each of the CMAKE variables
#
# Add additional regression tests to ctest below this line by calling
#
# list(APPEND TEST_DIRS name_of_folder )   
# list(APPEND TEST_PROCS ${NUM_MPI_PROC_TESTING})
# list(APPEND TEST_ARGS args_to_pass_to_run_test.py )
#
# Here name_of_folder should be the test folder located in parthenon/tst/regression/test_suites
# it is the parameter that is passed to run_test.py with the --test_dir argument

# Performance regression test
list(APPEND TEST_DIRS advection_performance)
list(APPEND TEST_PROCS "1")
list(APPEND TEST_ARGS "--driver ${PROJECT_BINARY_DIR}/example/advection/advection-example \
--driver_input ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/advection_performance/parthinput.advection_performance \
--num_steps 5")

if (ENABLE_HDF5)

  # h5py is needed for restart and hdff5 test
  list(APPEND REQUIRED_PYTHON_MODULES h5py) 

  # Restart
  list(APPEND TEST_DIRS restart)
  list(APPEND TEST_PROCS ${NUM_MPI_PROC_TESTING})
  list(APPEND TEST_ARGS "--driver ${PROJECT_BINARY_DIR}/example/advection/advection-example \
    --driver_input ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/restart/parthinput.restart \
    --num_steps 2")
  
  # Calculate pi example
  list(APPEND TEST_DIRS calculate_pi)
  list(APPEND TEST_PROCS ${NUM_MPI_PROC_TESTING})
  list(APPEND TEST_ARGS "--driver ${PROJECT_BINARY_DIR}/example/calculate_pi/pi-example \
  --driver_input ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/calculate_pi/parthinput.regression")

  # Advection test 
  list(APPEND TEST_DIRS advection_convergence)
  list(APPEND TEST_PROCS "1")
  list(APPEND TEST_ARGS "--driver ${PROJECT_BINARY_DIR}/example/advection/advection-example \
  --driver_input ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/advection_convergence/parthinput.advection \
  --num_steps 25")

  list(APPEND TEST_DIRS output_hdf5)
  list(APPEND TEST_PROCS ${NUM_MPI_PROC_TESTING})
  list(APPEND TEST_ARGS "--driver ${PROJECT_BINARY_DIR}/example/advection/advection-example \
    --driver_input ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/output_hdf5/parthinput.advection \
    --num_steps 4")
endif()

# Any external modules that are required by python can be added to REQUIRED_PYTHON_MODULES 
# list variable, before including TestSetup.cmake. 
list(APPEND REQUIRED_PYTHON_MODULES matplotlib numpy) 

# Include test setup functions, and check for python interpreter and modules
#  setup_test
#  setup_test_coverage
#  setup_test_mpi
#  setup_test_mpi_coverage
include(${PROJECT_SOURCE_DIR}/cmake/TestSetup.cmake)

list(LENGTH TEST_DIRS len)
math(EXPR end_index "${len} - 1")

foreach( index RANGE ${end_index})
  list(GET TEST_DIRS ${index} TEST_DIR)
  list(GET TEST_PROCS ${index} TEST_PROC)
  list(GET TEST_ARGS ${index} TEST_ARG)
  setup_test(${TEST_DIR} ${TEST_ARG})
  setup_test_coverage(${TEST_DIR} ${TEST_ARG})
  setup_test_mpi(${TEST_PROC} ${TEST_DIR} ${TEST_ARG})
  setup_test_mpi_coverage(${TEST_PROC} ${TEST_DIR} ${TEST_ARG})
endforeach()

# manually add label to performance regression test (to selectively enable/disable)
set_tests_properties(regression_test:advection_performance PROPERTIES LABELS "regression;mpi-no;perf-reg" )
if( MPI_FOUND )
  set_tests_properties(regression_mpi_test:advection_performance PROPERTIES LABELS "regression;mpi-yes;perf-reg" )
endif()
