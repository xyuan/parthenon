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

#include <iostream>

#include <catch2/catch.hpp>

#include "coordinates/coordinates.hpp"
#include "defs.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "parameter_input.hpp"

using parthenon::Coordinates_t;
using parthenon::IndexDomain;
using parthenon::Mesh;
using parthenon::MeshBlock;
using parthenon::MeshRefinement;
using parthenon::ParameterInput;
using parthenon::Real;

constexpr int NSIDE = 16;
constexpr int NDIM = 3;
constexpr Real XMIN = -1;
constexpr Real XMAX = 1;

TEST_CASE("Prolongation from coarse to fine cells for cell-centered variables",
          "[CellVariable][MeshRefinement][Coverage]") {
  GIVEN("meshblock, mesh, input, meshrefinement, and cellvariable objects") {
    ParameterInput input;
    Mesh mesh(NDIM);
    MeshBlock mb(NSIDE, NDIM);
    mb.pmy_mesh = &mesh;
    mb.block_size.x1min = XMIN;
    mb.block_size.x1max = XMAX;
    mb.block_size.x2min = XMIN;
    mb.block_size.x2max = XMAX;
    mb.block_size.x3min = XMIN;
    mb.block_size.x3max = XMAX;
    mb.block_size.nx1 = mb.cellbounds.ncellsi(IndexDomain::interior);
    mb.block_size.nx2 = mb.cellbounds.ncellsj(IndexDomain::interior);
    mb.block_size.nx3 = mb.cellbounds.ncellsk(IndexDomain::interior);
    mb.coords = Coordinates_t(mb.block_size, &input);
    MeshRefinement refinement(&mb, &input);
  }
}
