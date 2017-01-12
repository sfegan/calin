/*

   calin/math/geometry.hpp -- Stephen Fegan -- 2016-11-12

   Misc geometrical functions

   Copyright 2016, Stephen Fegan <sfegan@llr.in2p3.fr>
   LLR, Ecole polytechnique, CNRS/IN2P3, Universite Paris-Saclay

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

namespace calin { namespace math { namespace geometry {

inline bool box_has_future_intersection(double& tmin, double& tmax,
  const Eigen::Vector3d& min_corner, const Eigen::Vector3d& max_corner,
  const Eigen::Vector3d& pos, const Eigen::Vector3d& dir)
{
  // See: https://tavianator.com/fast-branchless-raybounding-box-intersections/
  // and: https://tavianator.com/fast-branchless-raybounding-box-intersections-part-2-nans/

  // Normalized direction vector
  const double vx = 1.0 / dir.x();
  const double vy = 1.0 / dir.y();
  const double vz = 1.0 / dir.z();

  Eigen::Vector3d min_rel = min_corner - pos;
  Eigen::Vector3d max_rel = max_corner - pos;

  const double tx1 = min_rel.x() * vx;
  const double tx2 = max_rel.x() * vx;
  tmin = std::min(tx1, tx2);
  tmax = std::max(tx1, tx2);

  const double ty1 = min_rel.y() * vy;
  const double ty2 = max_rel.y() * vy;
  tmin = std::max(tmin, std::min(std::min(ty1, ty2), tmax));
  tmax = std::min(tmax, std::max(std::max(ty1, ty2), tmin));

  const double tz1 = min_rel.z() * vz;
  const double tz2 = max_rel.z() * vz;
  tmin = std::max(tmin, std::min(std::min(tz1, tz2), tmax));
  tmax = std::min(tmax, std::max(std::max(tz1, tz2), tmin));

  return tmax > std::max(tmin, 0.0);
}

inline bool box_has_future_intersection(
  const Eigen::Vector3d& min_corner, const Eigen::Vector3d& max_corner,
  const Eigen::Vector3d& pos, const Eigen::Vector3d& dir)
{
  double tmin;
  double tmax;
  return box_has_future_intersection(tmin,tmax,min_corner,max_corner,pos,dir);
}

} } } // namespace calin::math::geometry