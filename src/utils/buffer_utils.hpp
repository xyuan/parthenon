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
#ifndef UTILS_BUFFER_UTILS_HPP_
#define UTILS_BUFFER_UTILS_HPP_
//! \file buffer_utils.hpp
//  \brief prototypes of utility functions to pack/unpack buffers

#include "Kokkos_Macros.hpp"
#include "defs.hpp"
#include "parthenon_arrays.hpp"

namespace parthenon {
namespace BufferUtility {
template <typename TSrc, typename TBuf>
KOKKOS_INLINE_FUNCTION
void PackData(TSrc &src, TBuf &buf, int sn, int en, int si, int ei, int sj, int ej,
              int sk, int ek, int &offset, MeshBlock *pmb) {
  for (int n = sn; n <= en; ++n) {
    for (int k = sk; k <= ek; k++) {
      for (int j = sj; j <= ej; j++) {
#pragma omp simd
        for (int i = si; i <= ei; i++)
          buf(offset++) = src(n, k, j, i);
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn template <typename TSrc, typename TDes> void PackData(ParArrayND<T> &src,
//! ParArray1D<T> &buf,
//                      int si, int ei, int sj, int ej, int sk, int ek, int &offset,
//                      MeshBlock *pmb)
//  \brief pack a 3D ParArray into a one-dimensional buffer

template <typename TSrc, typename TBuf>
KOKKOS_INLINE_FUNCTION
void PackData(TSrc &src, TBuf &buf, int si, int ei, int sj, int ej, int sk, int ek,
              int &offset, MeshBlock *pmb) {
  for (int k = sk; k <= ek; k++) {
    for (int j = sj; j <= ej; j++) {
#pragma omp simd
      for (int i = si; i <= ei; i++)
        buf(offset++) = src(k, j, i);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn template <typename TSrc, typename TDes> void UnpackData(ParArray1D<T> &buf,
//! ParArray4D<T> &dst,
//                        int sn, int en, int si, int ei, int sj, int ej, int sk, int ek,
//                        int &offset, MeshBlock *pmb)
//  \brief unpack a one-dimensional buffer into a ParArray4D

template <typename TBuf, typename TDes>
KOKKOS_INLINE_FUNCTION
void UnpackData(TBuf &buf, TDes &dst, int sn, int en, int si, int ei, int sj, int ej,
                int sk, int ek, int &offset, MeshBlock *pmb) {
  for (int n = sn; n <= en; ++n) {
    for (int k = sk; k <= ek; ++k) {
      for (int j = sj; j <= ej; ++j) {
#pragma omp simd
        for (int i = si; i <= ei; ++i)
          dst(n, k, j, i) = buf(offset++);
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn template <typename TSrc, typename TDes> void UnpackData(ParArray1D<T> &buf,
//! ParArray3D<T> &dst,
//                        int si, int ei, int sj, int ej, int sk, int ek, int &offset,
//                        MeshBlock *pmb)
//  \brief unpack a one-dimensional buffer into a 3D ParArray

template <typename TBuf, typename TDes>
KOKKOS_INLINE_FUNCTION
void UnpackData(TBuf &buf, TDes &dst, int si, int ei, int sj, int ej, int sk, int ek,
                int &offset, MeshBlock *pmb) {
  for (int k = sk; k <= ek; ++k) {
    for (int j = sj; j <= ej; ++j) {
#pragma omp simd
      for (int i = si; i <= ei; ++i)
        dst(k, j, i) = buf(offset++);
    }
  }
  return;
}
// // 2x templated and overloaded functions
// // 4D
// template <typename TSrc, typename TBuf>
// void PackData(TSrc &src, TBuf &buf, int sn, int en, int si, int ei,
//               int sj, int ej, int sk, int ek, int &offset, MeshBlock *pmb);
// // 3D
// template <typename TSrc, typename TBuf>
// void PackData(TSrc &src, TBuf &buf, int si, int ei, int sj, int ej,
//               int sk, int ek, int &offset, MeshBlock *pmb);
// // 4D
// template <typename TBuf, typename TDes>
// void UnpackData(TBuf &buf, TDes &dst, int sn, int en, int si, int ei,
//                 int sj, int ej, int sk, int ek, int &offset, MeshBlock *pmb);
// // 3D
// template <typename TBuf, typename TDes>
// void UnpackData(TBuf &buf, TDes &dst, int si, int ei, int sj, int ej,
//                 int sk, int ek, int &offset, MeshBlock *pmb);

} // namespace BufferUtility
} // namespace parthenon

#endif // UTILS_BUFFER_UTILS_HPP_
