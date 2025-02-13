/*

   calin/math/covariance_calc.hpp -- Stephen Fegan -- 2016-04-24

   Utility functions for covariance calculation

   Copyright 2016, Stephen Fegan <sfegan@llr.in2p3.fr>
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

#include <cstdint>

namespace calin { namespace math { namespace covariance_calc {

double cov_i64_gen(int64_t sij, int64_t nij,
  int64_t si, int64_t ni, int64_t sj, int64_t nj);

double cov_double_gen(double sij, int64_t nij,
  double si, int64_t ni, double sj, int64_t nj);

template<typename sum_type, typename count_type>
inline double cov_gen(sum_type sij, count_type nij,
  sum_type si, count_type ni, sum_type sj, count_type nj) {
  return cov_i64_gen(sij, nij, si, ni, sj, nj); }

template<typename count_type>
inline double cov_gen(double sij, count_type nij,
  double si, count_type ni, double sj, count_type nj) {
  return cov_double_gen(sij, nij, si, ni, sj, nj); }

template<typename count_type>
inline double cov_gen(float sij, count_type nij,
  float si, count_type ni, float sj, count_type nj) {
  return cov_double_gen(sij, nij, si, ni, sj, nj); }

} } } // namespace calin::math::covariance_calc
