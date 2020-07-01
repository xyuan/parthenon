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

#include <cmath>
#include <string>

#include <parthenon/package.hpp>

#include "poisson_package.hpp"

using namespace parthenon::package::prelude;

namespace poisson {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("poisson_package");

  Real K = pin->GetOrAddReal("poisson", "K", 1);
  pkg->AddParam<>("K", K);

  Real w = pin->GetOrAddReal("poisson", "weight", 1);
  pkg->AddParam<>("weight", w);

  int subcycles = pin->GetOrAddInteger("poisson", "subcycles", 1);
  pkg->AddParam<>("subcycles", subcycles);

  std::string field_name = "field";
  Metadata m({Metadata::Cell, Metadata::Independent, Metadata::FillGhost});
  pkg->AddField(field_name, m);

  field_name = "potential";
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
  pkg->AddField(field_name, m);

  field_name = "residual";
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
  pkg->AddField(field_name, m);

  field_name = "inv_diagonal";
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
  pkg->AddField(field_name, m);

  pkg->FillDerived = ComputeResidualAndDiagonal;

  return pkg;
}

// TODO(JMM): Refactor to reduced repeated code
Real GetL1Residual(std::shared_ptr<Container<Real>> &rc) {
  MeshBlock *pmb = rc->pmy_block;
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  ParArrayND<Real> res = rc->Get("residual").data;

  Real max;
  Kokkos::parallel_reduce(
      "Poisson Get Residual",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>(pmb->exec_space, {kb.s, jb.s, ib.s},
                                             {kb.e + 1, jb.e + 1, ib.e + 1}),
      KOKKOS_LAMBDA(int k, int j, int i, Real &lmax) {
        lmax = fmax(fabs(res(k, j, i)), lmax);
      },
      Kokkos::Max<Real>(max));
  return max;
}

void ComputeResidualAndDiagonal(std::shared_ptr<Container<Real>> &rc) {
  MeshBlock *pmb = rc->pmy_block;
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto &coords = pmb->coords;
  auto pkg = pmb->packages["poisson_package"];
  int ndim = pmb->pmy_mesh->ndim;
  ParArrayND<Real> phi = rc->Get("field").data;
  ParArrayND<Real> rho = rc->Get("potential").data;
  ParArrayND<Real> res = rc->Get("residual").data;
  ParArrayND<Real> Dinv = rc->Get("inv_diagonal").data;
  Real K = pkg->Param<Real>("K");

  pmb->par_for(
      "ComputeResidualAndDiagonal", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real dx = coords.Dx(X1DIR, k, j, i);
        Real dy = coords.Dx(X2DIR, k, j, i);
        Real dz = coords.Dx(X3DIR, k, j, i);
        Real dx2 = dx * dx;
        Real dy2 = dy * dy;
        Real dz2 = dz * dz;
        Real ds2;
        if (ndim >= 3) {
          ds2 = (dx2 * dy2 * dz2) / (dx2 * dy2 + dx2 * dz2 + dy2 * dz2);
        } else if (ndim >= 2) {
          ds2 = (dx2 * dy2) / (dx2 + dy2);
        } else {
          ds2 = dx2;
        }
        Dinv(k, j, i) = -ds2 / 2;
        res(k, j, i) = K * rho(k, j, i);
        res(k, j, i) -= (phi(k, j, i + 1) + phi(k, j, i - 1) - 2 * phi(k, j, i)) / dx2;
        if (ndim >= 2) {
          res(k, j, i) -= (phi(k, j + 1, i) + phi(k, j - 1, i) - 2 * phi(k, j, i)) / dy2;
        }
        if (ndim >= 3) {
          res(k, j, i) -= (phi(k + 1, j, i) + phi(k - 1, j, i) - 2 * phi(k, j, i)) / dz2;
        }
      });
  return;
}

TaskStatus Smooth(std::shared_ptr<Container<Real>> &rc_in,
                  std::shared_ptr<Container<Real>> &rc_out) {
  MeshBlock *pmb = rc_in->pmy_block;
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto pkg = pmb->packages["poisson_package"];
  ParArrayND<Real> in = rc_in->Get("field").data;
  ParArrayND<Real> out = rc_out->Get("field").data;
  ParArrayND<Real> res = rc_in->Get("residual").data;
  ParArrayND<Real> Dinv = rc_in->Get("inv_diagonal").data;
  Real w = pkg->Param<Real>("weight");
  int nsub = pkg->Param<int>("subcycles");

  pmb->par_for(
      "JacobiSmooth", 1, nsub, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int n, const int k, const int j, const int i) {
        out(k, j, i) = in(k, j, i) + w * Dinv(k, j, i) * res(k, j, i);
      });
  return TaskStatus::complete;
}

} // namespace poisson
