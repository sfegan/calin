/*

   calin/math/geometry.cpp -- Stephen Fegan -- 2021-04-30

   Functions for geometrical manuipulations

   Copyright 2021, Stephen Fegan <sfegan@llr.in2p3.fr>
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

#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <numeric>
#include <cmath>
#include <random>

#include <math/geometry.hpp>

using namespace calin::math::geometry;

Eigen::Quaterniond calin::math::geometry::
euler_to_quaternion(const calin::ix::common_types::EulerAngles3D& euler)
{
  using Eigen::Quaterniond;
  using Eigen::AngleAxisd;
  using Eigen::Vector3d;
  Eigen::Quaterniond q;
  switch(euler.rotation_order()) {
  case calin::ix::common_types::EulerAngles3D::ZXZ:
    return AngleAxisd(euler.alpha()*(M_PI/180.0), Vector3d::UnitZ())
          * AngleAxisd(euler.beta()*(M_PI/180.0), Vector3d::UnitX())
          * AngleAxisd(euler.gamma()*(M_PI/180.0), Vector3d::UnitZ());
  case calin::ix::common_types::EulerAngles3D::XYX:
    return AngleAxisd(euler.alpha()*(M_PI/180.0), Vector3d::UnitX())
          * AngleAxisd(euler.beta()*(M_PI/180.0), Vector3d::UnitY())
          * AngleAxisd(euler.gamma()*(M_PI/180.0), Vector3d::UnitX());
  case calin::ix::common_types::EulerAngles3D::YZY:
    return AngleAxisd(euler.alpha()*(M_PI/180.0), Vector3d::UnitY())
          * AngleAxisd(euler.beta()*(M_PI/180.0), Vector3d::UnitZ())
          * AngleAxisd(euler.gamma()*(M_PI/180.0), Vector3d::UnitY());
  case calin::ix::common_types::EulerAngles3D::ZYZ:
    return AngleAxisd(euler.alpha()*(M_PI/180.0), Vector3d::UnitZ())
          * AngleAxisd(euler.beta()*(M_PI/180.0), Vector3d::UnitY())
          * AngleAxisd(euler.gamma()*(M_PI/180.0), Vector3d::UnitZ());
  case calin::ix::common_types::EulerAngles3D::XZX:
    return AngleAxisd(euler.alpha()*(M_PI/180.0), Vector3d::UnitX())
          * AngleAxisd(euler.beta()*(M_PI/180.0), Vector3d::UnitZ())
          * AngleAxisd(euler.gamma()*(M_PI/180.0), Vector3d::UnitX());
  case calin::ix::common_types::EulerAngles3D::YXY:
    return AngleAxisd(euler.alpha()*(M_PI/180.0), Vector3d::UnitY())
          * AngleAxisd(euler.beta()*(M_PI/180.0), Vector3d::UnitX())
          * AngleAxisd(euler.gamma()*(M_PI/180.0), Vector3d::UnitY());
  default:
    throw std::runtime_error("Unsupported rotation order");
  }
}

Eigen::Matrix3d calin::math::geometry::
euler_to_matrix(const calin::ix::common_types::EulerAngles3D& euler)
{
  return euler_to_quaternion(euler).toRotationMatrix();
}

void calin::math::geometry::quaternion_to_euler(
  calin::ix::common_types::EulerAngles3D* euler, const Eigen::Quaterniond& q)
{
  return matrix_to_euler(euler, q.toRotationMatrix());
}

void calin::math::geometry::matrix_to_euler(
  calin::ix::common_types::EulerAngles3D* euler, const Eigen::Matrix3d& m)
{
  Eigen::Vector3d v;
  switch(euler->rotation_order()) {
  case calin::ix::common_types::EulerAngles3D::ZXZ:
    v = m.eulerAngles(2,0,2); break;
  case calin::ix::common_types::EulerAngles3D::XYX:
    v = m.eulerAngles(0,1,0); break;
  case calin::ix::common_types::EulerAngles3D::YZY:
    v = m.eulerAngles(1,2,1); break;
  case calin::ix::common_types::EulerAngles3D::ZYZ:
    v = m.eulerAngles(2,1,2); break;
  case calin::ix::common_types::EulerAngles3D::XZX:
    v = m.eulerAngles(0,2,0); break;
  case calin::ix::common_types::EulerAngles3D::YXY:
    v = m.eulerAngles(1,0,1); break;
  default:
    throw std::runtime_error("Unsupported rotation order");
  }
  if(v(1) < 0) {
    v(0) = std::fmod(v(0) + 2*M_PI, 2*M_PI) - M_PI;
    v(1) = std::fabs(v(1));
    v(2) = std::fmod(v(2) + 2*M_PI, 2*M_PI) - M_PI;
  }
  euler->set_alpha(v(0) * 180.0/M_PI);
  euler->set_beta(v(1) * 180.0/M_PI);
  euler->set_gamma(v(2) * 180.0/M_PI);
}

void calin::math::geometry::scattering_euler(
  calin::ix::common_types::EulerAngles3D* euler, double dispersion, math::rng::RNG& rng,
  double twist_dispersion)
{
  double alpha = 0;
  double beta = 0;
  double gamma = 0;

  if(dispersion>0)
  {
    alpha = rng.uniform() * 360.0 - 180.0;
    beta = dispersion*std::sqrt(-2.0*std::log(rng.uniform()));
    gamma = -alpha;
  }

  if(twist_dispersion > 0) {
    gamma = std::fmod(std::fmod(gamma + rng.uniform()*twist_dispersion + 180, 360) + 360, 360) - 180;
  }

  euler->set_alpha(alpha);
  euler->set_beta(beta);
  euler->set_gamma(gamma);
}

calin::ix::common_types::EulerAngles3D
calin::math::geometry::scattering_euler(double dispersion, calin::math::rng::RNG& rng,
  double twist_dispersion, calin::ix::common_types::EulerAngles3D::RotationOrder rotation_order)
{
  calin::ix::common_types::EulerAngles3D euler;
  euler.set_rotation_order(rotation_order);
  scattering_euler(&euler, dispersion, rng, twist_dispersion);
  return euler;
}

bool calin::math::geometry::euler_is_zero(
  const calin::ix::common_types::EulerAngles3D& euler)
{
  return euler.alpha()==0 and euler.beta()==0 and euler.gamma()==0;
}

namespace {
  struct Circle {
    double x, y, r;
    
    Circle(double x = 0, double y = 0, double r = 0) : x(x), y(y), r(r) {}
    
    Circle(const Eigen::Vector3d& v) : x(v[0]), y(v[1]), r(v[2]) {}
    
    bool contains(const Eigen::Vector3d& c) const {
      double dist = std::sqrt((x - c[0]) * (x - c[0]) + (y - c[1]) * (y - c[1]));
      return dist + c[2] <= r + 1e-9;  // Small epsilon for floating point errors
    }
    
    Eigen::Vector3d toVector() const {
        return Eigen::Vector3d(x, y, r);
    }
  };

  // Find circle passing through 2 circles (smallest circle containing both)
  inline Circle circleFrom2(const Eigen::Vector3d& c1, const Eigen::Vector3d& c2) {
    double dx = c2[0] - c1[0];
    double dy = c2[1] - c1[1];
    double dist = std::sqrt(dx * dx + dy * dy);
      
    // New circle center is on the line between the two circles
    // at a position such that both circles are on the boundary
    double cx = (c1[0] + c2[0]) / 2.0;
    double cy = (c1[1] + c2[1]) / 2.0;
    double cr = (dist + c1[2] + c2[2]) / 2.0;
      
    // Adjust center position
    if (dist > 1e-9) {
      double offset = (cr - c1[2]) - dist / 2.0;
      cx = c1[0] + dx / dist * (dist / 2.0 + offset);
      cy = c1[1] + dy / dist * (dist / 2.0 + offset);
    }
      
    return Circle(cx, cy, cr);
  }

  // Find circle passing through 3 circles
  inline Circle circleFrom3(const Eigen::Vector3d& c1, const Eigen::Vector3d& c2, const Eigen::Vector3d& c3) {
    // Try all pairs and return the smallest that contains all three
    Circle c12 = circleFrom2(c1, c2);
    Circle c23 = circleFrom2(c2, c3);
    Circle c13 = circleFrom2(c1, c3);
      
    if (c12.contains(c3) && (c12.r <= c23.r || !c23.contains(c1)) && 
        (c12.r <= c13.r || !c13.contains(c2))) {
        return c12;
    }
    if (c23.contains(c1) && (c23.r <= c13.r || !c13.contains(c2))) {
        return c23;
    }
    return c13;
  }

  // Welzl's algorithm - recursive helper
  Circle welzlHelper(const std::vector<Eigen::Vector3d>& circles, 
                    std::vector<int>& indices, int n, std::vector<int>& boundary) {
      
    // Base cases
    if (n == 0 || boundary.size() == 3) {
      if (boundary.size() == 1) {
        return Circle(circles[boundary[0]]);
      }
      if (boundary.size() == 2) {
        int i = boundary[0], j = boundary[1];
         return Circle(circles[i]).contains(circles[j]) 
                  ? Circle(circles[i])
                  : circleFrom2(circles[i], circles[j]);
      }
      // boundary.size() == 3
      int i = boundary[0], j = boundary[1], k = boundary[2];
      return circleFrom3(circles[i], circles[j], circles[k]);
    }
      
    // Pick a random circle
    int idx = indices[n - 1];
      
    // Recursively find circle without this circle
    Circle c = welzlHelper(circles, indices, n - 1, boundary);
      
    // If current circle is already contained, return
    if (c.contains(circles[idx])) {
      return c;
    }
      
    // Otherwise, it must be on the boundary
    boundary.push_back(idx);
    c = welzlHelper(circles, indices, n - 1, boundary);
    boundary.pop_back();
      
    return c;
  }

  Eigen::Vector3d smallestEnclosingCircle(const std::vector<Eigen::Vector3d>& circles) {      
    int n = circles.size();
      
    if (n == 1) return circles[0];
      
    // Fast O(n) check: start with the largest radius circle as candidate
    int candidateIdx = 0;
    for (int i = 1; i < n; i++) {
      if (circles[i][2] > circles[candidateIdx][2]) {
        candidateIdx = i;
      }
    }
      
    // Check if this candidate contains all other circles
    Circle candidate(circles[candidateIdx]);
    bool containsAll = true;
    for (int i = 0; i < n; i++) {
      if (i != candidateIdx && !candidate.contains(circles[i])) {
        containsAll = false;
        break;
      }
    }
      
    if (containsAll) {
      return circles[candidateIdx];
    }
      
    // Create shuffled indices for randomization (important for expected O(n) time)
    std::vector<int> indices(n);
    for (int i = 0; i < n; i++) indices[i] = i;
      
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);
      
    // Run Welzl's algorithm
    std::vector<int> boundary;
    Circle result = welzlHelper(circles, indices, n, boundary);
      
    return result.toVector();
  }
} // anonymous namespace

// Input circles are (x,y,radius) or circles and output is (x,y,z,radius) for containing circle
Eigen::Vector3d calin::math::geometry::containing_circle_2d(std::vector<Eigen::Vector3d>& circles)
{
  return smallestEnclosingCircle(circles);
}

// Input circles are (nx,ny,nz,radius) on sphere and output is (nx,ny,nz,radius) for containing circle
Eigen::Vector4d calin::math::geometry::containing_circle_sphere(std::vector<Eigen::Vector4d>& circles_on_sphere)
{
  std::vector<Eigen::Vector3d> circles_2d(circles_on_sphere.size());
  for(unsigned i=0; i<circles_on_sphere.size(); ++i) {
    circles_2d[i] = project_circle_on_sphere_to_2d(circles_on_sphere[i]);
  }
  Eigen::Vector3d containing_2d = containing_circle_2d(circles_2d);
  return deproject_circle_on_sphere_from_2d(containing_2d);
}

