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

#include <array>
#include <iostream>

#include <catch2/catch.hpp>

#include "config.hpp"
#include "coordinates/coordinates.hpp"
#include "defs.hpp"
#include "interface/metadata.hpp"
#include "interface/variable.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"

using parthenon::CellVariable;
using parthenon::Coordinates_t;
using parthenon::IndexDomain;
using parthenon::IndexShape;
using parthenon::Mesh;
using parthenon::MeshBlock;
using parthenon::MeshRefinement;
using parthenon::Metadata;
using parthenon::Real;

constexpr int NSIDE = 16;
constexpr int NDIM = 3;
constexpr Real XMIN = -1;
constexpr Real XMAX = 1;

TEST_CASE("Prolongation from coarse to fine cells for cell-centered variables",
          "[CellVariable][MeshRefinement][Coverage]") {
  GIVEN("meshblock, meshrefinement, and cellvariable objects") {
    MeshBlock mb(NSIDE, NDIM);
    mb.c_cellbounds = IndexShape(NSIDE / 2, NSIDE / 2, NSIDE / 2, NGHOST);
    mb.coords = Coordinates_t(mb.block_size);
    MeshRefinement refinement(&mb, mb.coords);

    const std::array<int,6> shape = {mb.cellbounds.ncellsi(IndexDomain::entire),
                                     mb.cellbounds.ncellsj(IndexDomain::entire),
                                     mb.cellbounds.ncellsk(IndexDomain::entire),
                                     1,
                                     1,
                                     1};
    Metadata m({Metadata::Cell});
    CellVariable<Real> v("var", shape, m);
    v.AllocateCoarseCells(mb.c_cellbounds);
  }
}
