//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
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
#ifndef CONFIG_HPP_
#define CONFIG_HPP_
//! \file config.hpp.in
//  \brief Template file for config.hpp.  When the configure.py script is run, a new
//  config.hpp file will be created (overwriting the last) from this template. This new
//  file contains Athena++ specific cpp macros and definitions set by configure.

#define COORDINATE_TYPE UniformCartesian

//----------------------------------------------------------------------------------------
// macros which define physics and algorithms

// configure.py dict(definitions) string values:
// problem generator
#define PROBLEM_GENERATOR ""

// use single precision floating-point values (binary32)? default=0 (false; use binary64)
#define SINGLE_PRECISION_ENABLED 0

// configure.py dict(definitions) Boolean string macros:
// (these options have the latter (false) option as defaults, unless noted otherwise)

// MPI parallelization (MPI_PARALLEL or NOT_MPI_PARALLEL)
#define MPI_PARALLEL

// OpenMP parallelization (OPENMP_PARALLEL or NOT_OPENMP_PARALLEL)
#define OPENMP_PARALLEL

// HDF5 output (HDF5OUTPUT or NO_HDF5OUTPUT)
#define HDF5OUTPUT

// Default loop patterns for MeshBlock par_for() wrappers,
// see kokkos_abstraction.hpp for available tags.
// Kokkos tight loop layout
#define DEFAULT_LOOP_PATTERN parthenon::loop_pattern_flatrange_tag

// Kokkos nested loop layout
#define DEFAULT_OUTER_LOOP_PATTERN parthenon::outer_loop_pattern_teams_tag
#define DEFAULT_INNER_LOOP_PATTERN parthenon::inner_loop_pattern_tvr_tag

// try/throw/catch C++ exception handling (ENABLE_EXCEPTIONS or DISABLE_EXCEPTIONS)
// (enabled by default)
#define ENABLE_EXCEPTIONS

// compiler options
#define COMPILED_WITH ""
#define COMPILER_COMMAND "<not-implemented>"
#define COMPILED_WITH_OPTIONS "<not-implemented>" // NOLINT

//----------------------------------------------------------------------------------------
// macros associated with numerical algorithm (rarely modified)

#define NFIELD 0
#define NWAVE 5
#define NGHOST 2
#define MAX_NSTAGE 5     // maximum number of stages per cycle for time-integrator
#define MAX_NREGISTER 3  // maximum number of (u, b) register pairs for time-integrator

//----------------------------------------------------------------------------------------
// general purpose macros (never modified)

// all constants specified to 17 total digits of precision = max_digits10 for "double"
#define SQRT2 1.4142135623730951
#define ONE_OVER_SQRT2 0.70710678118654752
#define ONE_3RD 0.33333333333333333
#define TWO_3RD 0.66666666666666667
#define TINY_NUMBER 1.0e-20
#define HUGE_NUMBER 1.0e+36
#define SQR(x) ( (x) * (x) )
#define SIGN(x) ( ((x) < 0.0) ? -1.0 : 1.0 )

#endif // CONFIG_HPP_