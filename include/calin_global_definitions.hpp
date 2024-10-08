/*

   calin/calin_global_definitions.hpp -- Stephen Fegan -- 2015-04-16

   Definitions of types and macros that apply to the full package

   Copyright 2015, Stephen Fegan <sfegan@llr.in2p3.fr>
   Laboratoire Leprince-Ringuet, CNRS/IN2P3, Ecole Polytechnique, Institut Polytechnique de Paris

   This file is part of "calin"

   "calin" is free software: you can redistribute it and/or modify it
   under the terms of the GNU General Public License version 2 or
   later, as published by the Free Software Foundation.

   "calin" is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.

*/

#pragma once

#include <google/protobuf/repeated_field.h>
#include <vector>
#include <Eigen/Core>

#define CALIN_NEW_ALIGN 32

namespace calin {

#ifndef SWIG
#define CALIN_TYPEALIAS(A,B...) using A = B
#else
#define CALIN_TYPEALIAS(A,B...) typedef B A
#endif

//#define CALIN_USE_EIGEN_REF

CALIN_TYPEALIAS(VecRef, Eigen::VectorXd&);
CALIN_TYPEALIAS(MatRef, Eigen::MatrixXd&);
CALIN_TYPEALIAS(ConstVecRef, const Eigen::VectorXd&);
CALIN_TYPEALIAS(ConstMatRef, const Eigen::MatrixXd&);

CALIN_TYPEALIAS(IntVecRef, Eigen::VectorXi&);
CALIN_TYPEALIAS(IntMatRef, Eigen::MatrixXi&);
CALIN_TYPEALIAS(ConstIntVecRef, const Eigen::VectorXi&);
CALIN_TYPEALIAS(ConstIntMatRef, const Eigen::MatrixXi&);

inline std::vector<double> eigen_to_stdvec(const Eigen::VectorXd& x)
{
  return std::vector<double>(x.data(), x.data()+x.size());
}

inline Eigen::VectorXd std_to_eigenvec(const std::vector<double> &x)
{
  return Eigen::Map<const Eigen::VectorXd>(&x.front(), x.size());
}

inline std::vector<int> eigen_to_stdvec(const Eigen::VectorXi& x)
{
  return std::vector<int>(x.data(), x.data()+x.size());
}

inline std::vector<unsigned> eigen_to_stdvec_unsigned(const Eigen::VectorXi& x)
{
  return std::vector<unsigned>(x.data(), x.data()+x.size());
}

inline Eigen::VectorXi std_to_eigenvec(const std::vector<int> &x)
{
  return Eigen::Map<const Eigen::VectorXi>(&x.front(), x.size());
}

inline Eigen::VectorXi std_to_eigenvec_unsigned(const std::vector<unsigned> &x)
{
  return Eigen::Map<const Eigen::VectorXi>(reinterpret_cast<const int*>(&x.front()), x.size());
}

template<typename T> inline 
std::vector<T> protobuf_to_stdvec(const google::protobuf::RepeatedField<T>& x)
{
  return std::vector<T>(x.begin(), x.end());
}

template<typename T> inline 
std::vector<T> protobuf_to_stdvec(const google::protobuf::RepeatedField<T>* x)
{
  return std::vector<T>(x->begin(), x->end());
}

template<typename T> inline void stdvec_to_existing_protobuf(
  google::protobuf::RepeatedField<T>& y, const std::vector<int> &x)
{
  y.Resize(x.size());
  std::copy(x.begin(), x.end(), y.begin());
}

template<typename T> inline void stdvec_to_existing_protobuf(
  google::protobuf::RepeatedField<T>* y, const std::vector<int> &x)
{
  y->Resize(x.size());
  std::copy(x.begin(), x.end(), y->begin());
}

template<typename T, typename V> inline void vec_to_xyz(T* xyz, const V& v)
{
  xyz->set_x(v[0]);
  xyz->set_y(v[1]);
  xyz->set_z(v[2]);
}

template<typename T, typename V> inline T vec_to_xyz(const V& v)
{
  T xyz;
  xyz.set_x(v[0]);
  xyz.set_y(v[1]);
  xyz.set_z(v[2]);
  return xyz;
}

template<typename T> inline Eigen::Vector3d xyz_to_eigenvec(const T& xyz)
{
  return { xyz.x(), xyz.y(), xyz.z() };
}

}; // namespace calin
