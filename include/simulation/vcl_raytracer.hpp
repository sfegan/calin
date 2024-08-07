/*

   calin/simulation/vcl_raytracer.hpp -- Stephen Fegan -- 2018-09-10

   Class for raytracing on a single VSOTelescope using VCL

   Copyright 2018, Stephen Fegan <sfegan@llr.in2p3.fr>
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

#include <algorithm>
#include <limits>

#include <util/memory.hpp>
#include <util/vcl.hpp>
#include <math/special.hpp>
#include <math/ray_vcl.hpp>
#include <math/rng_vcl.hpp>
#include <math/hex_array_vcl.hpp>
#include <math/geometry_vcl.hpp>
#include <simulation/vso_telescope.hpp>
#include <simulation/vso_obscuration.hpp>
#include <util/log.hpp>

namespace calin { namespace simulation { namespace vcl_raytracer {

#ifndef SWIG

enum VCLScopeTraceStatus {
  STS_MASKED_ON_ENTRY,                        // 0
  STS_TRAVELLING_AWAY_REFLECTOR,              // 1
  STS_MISSED_REFLECTOR_SPHERE,                // 2
  STS_OUTSIDE_REFLECTOR_APERTURE,             // 3
  STS_NO_MIRROR,                              // 4
  STS_MISSED_MIRROR_SPHERE,                   // 5
  STS_MISSED_MIRROR_EDGE,                     // 6
  STS_OBSCURED_BEFORE_MIRROR,                 // 7
      TS_MISSED_WINDOW,                       // 8
  STS_OBSCURED_BEFORE_FOCAL_PLANE,            // 9
  STS_TRAVELLING_AWAY_FROM_FOCAL_PLANE,       // 10
  STS_OUTSIDE_FOCAL_PLANE_APERTURE,           // 11
  STS_TS_NO_PIXEL,                            // 12
  STS_TS_FOUND_PIXEL                          // 13
};

template<typename VCLReal> class alignas(VCLReal::vec_bytes) VCLScopeTraceInfo: public VCLReal
{
public:
  using typename VCLReal::real_t;
  using typename VCLReal::real_vt;
  using typename VCLReal::bool_vt;
  using typename VCLReal::int_vt;
  using typename VCLReal::uint_vt;
  using typename VCLReal::vec3_vt;
  using typename VCLReal::mat3_vt;

  int_vt              status;    // Status of ray at end of tracing

  real_vt             reflec_x;  // Ray intersection point on reflector sphere
  real_vt             reflec_y;  // Ray intersection point on reflector sphere
  real_vt             reflec_z;  // Ray intersection point on reflector sphere

  uint_vt             pre_reflection_obs_hitmask;   // Bitmask for pre-reflection obscurations hit by ray
  uint_vt             post_reflection_obs_hitmask;  // Bitmask for post-reflection obscurations hit by ray
  uint_vt             camera_obs_hitmask;  // Bitmask for camera frame obscurations hit by ray

  int_vt              mirror_hexid;  // Grid hex ID of mirror facet hit by ray
  int_vt              mirror_id;     // Sequential ID of mirror facet hit by ray
  real_vt             mirror_x;      // Ray intersection point with mirror facet
  real_vt             mirror_y;      // Ray intersection point with mirror facet
  real_vt             mirror_z;      // Ray intersection point with mirror facet
  real_vt             mirror_n_dot_u; // Cosine if angle between ray and mirror normal

  real_vt             fplane_x;  // Ray intersection point on focal plane
  real_vt             fplane_z;  // Ray intersection point on focal plane
  real_vt             fplane_t;  // Ray intersection time on focal plane
  real_vt             fplane_ux; // X directional cosine of ray at focal plane
  real_vt             fplane_uy; // Cosine of angle between ray and focal plane (normal)
  real_vt             fplane_uz; // X directional cosine of ray at focal plane

  int_vt              pixel_hexid; // Grid hex ID of pixel on focal plane
  int_vt              pixel_id;    // Sequential ID of pixel on focal plane (or -1)
};

template<typename VCLReal> class alignas(VCLReal::vec_bytes) VCLObscuration: public VCLReal
{
public:
  using typename VCLReal::real_vt;
  using typename VCLReal::bool_vt;
  using typename VCLReal::vec3_vt;
  using typename VCLReal::mat3_vt;
  using Ray = calin::math::ray::VCLRay<VCLReal>;

  using typename VCLReal::vec3_t;
  using typename VCLReal::real_t;

  virtual ~VCLObscuration() {
    // nothing to see here
  }
  virtual bool_vt doesObscure(const Ray& ray_in, Ray& ray_out, real_vt n) const = 0;
  virtual VCLObscuration<VCLReal>* clone() const = 0;
};

template<typename VCLReal> class VCLAlignedBoxObscuration;
template<typename VCLReal> class VCLAlignedRectangularAperture;
template<typename VCLReal> class VCLAlignedCircularAperture;
template<typename VCLReal> class VCLTubeObscuration;

template<typename VCLReal> class alignas(VCLReal::vec_bytes) VCLScopeRayTracer: public VCLReal
{
public:
  using typename VCLReal::real_t;
  using typename VCLReal::int_t;
  using typename VCLReal::uint_t;
  using typename VCLReal::bool_int_vt;
  using typename VCLReal::bool_uint_vt;
  using typename VCLReal::mat3_t;
  using typename VCLReal::vec3_t;
  using typename VCLReal::real_vt;
  using typename VCLReal::bool_vt;
  using typename VCLReal::int_vt;
  using typename VCLReal::uint_vt;
  using typename VCLReal::vec3_vt;
  using typename VCLReal::mat3_vt;
  using Ray = calin::math::ray::VCLRay<VCLReal>;
  using TraceInfo = VCLScopeTraceInfo<VCLReal>;
  using RNG = calin::math::rng::VCLRealRNG<VCLReal>;

  VCLScopeRayTracer(const calin::simulation::vs_optics::VSOTelescope* scope,
      real_t refractive_index = 1.0,
      RNG* rng = nullptr, bool adopt_rng = false):
    VCLReal(), rng_(rng==nullptr ? new RNG(__PRETTY_FUNCTION__) : rng),
    adopt_rng_(rng==nullptr ? true : adopt_rng)
  {
    using calin::math::special::SQR;
    using calin::simulation::vs_optics::VSOAlignedBoxObscuration;
    using calin::simulation::vs_optics::VSOAlignedRectangularAperture;
    using calin::simulation::vs_optics::VSOAlignedCircularAperture;
    using calin::simulation::vs_optics::VSOTubeObscuration;

    global_to_reflector_off_ = scope->translationGlobalToReflector().cast<real_t>();
    global_to_reflector_rot_ = scope->rotationGlobalToReflector().cast<real_t>();
    ref_index_               = refractive_index;

    reflec_curvature_radius_ = scope->curvatureRadius();
    reflec_aperture2_        = 0;
    reflec_crot_             = scope->cosReflectorRotation();
    reflec_srot_             = scope->sinReflectorRotation();
    reflec_scaleinv_         = 1.0/scope->facetSpacing();
    reflec_shift_x_          = scope->facetGridShiftX();
    reflec_shift_z_          = scope->facetGridShiftZ();
    reflec_cw_               = scope->mirrorParity();

    mirror_hexid_end_        = scope->numMirrorHexSites();
    mirror_id_end_           = scope->numMirrors();

    calin::util::memory::aligned_calloc(mirror_id_lookup_, mirror_hexid_end_+1);
    calin::util::memory::aligned_calloc(mirror_nx_lookup_, mirror_id_end_+1);
    calin::util::memory::aligned_calloc(mirror_nz_lookup_, mirror_id_end_+1);
    calin::util::memory::aligned_calloc(mirror_ny_lookup_, mirror_id_end_+1);
    calin::util::memory::aligned_calloc(mirror_r_lookup_, mirror_id_end_+1);
    calin::util::memory::aligned_calloc(mirror_x_lookup_, mirror_id_end_+1);
    calin::util::memory::aligned_calloc(mirror_z_lookup_, mirror_id_end_+1);
    calin::util::memory::aligned_calloc(mirror_y_lookup_, mirror_id_end_+1);
    calin::util::memory::aligned_calloc(mirror_normdisp_lookup_, mirror_id_end_+1);

    for(int ihexid=0; ihexid<mirror_hexid_end_; ihexid++) {
      auto* mirror = scope->mirrorByHexID(ihexid);
      mirror_id_lookup_[ihexid] =
        (mirror==nullptr or mirror->removed()) ? mirror_id_end_ : mirror->id();
    }
    mirror_id_lookup_[mirror_hexid_end_] = mirror_id_end_;

    for(int iid=0; iid<mirror_id_end_; iid++) {
      auto* mirror = scope->mirror(iid);
      mirror_nx_lookup_[iid] = mirror->align().x();
      mirror_nz_lookup_[iid] = mirror->align().z();
      mirror_ny_lookup_[iid] = mirror->align().y();
      mirror_r_lookup_[iid] = 2.0*mirror->focalLength();
      mirror_x_lookup_[iid] = mirror->pos().x();
      mirror_z_lookup_[iid] = mirror->pos().z();
      mirror_y_lookup_[iid] = mirror->pos().y();
      mirror_normdisp_lookup_[iid] = 0.25*mirror->spotSize()/mirror->focalLength();
      reflec_aperture2_ = std::max(reflec_aperture2_, real_t(SQR(mirror->pos().x())+SQR(mirror->pos().z())));
    }
    mirror_nx_lookup_[mirror_id_end_] = 0.0;
    mirror_nz_lookup_[mirror_id_end_] = 0.0;
    mirror_ny_lookup_[mirror_id_end_] = 1.0;
    mirror_r_lookup_[mirror_id_end_] = 100000.0; // 1km - should be enough!
    mirror_x_lookup_[mirror_id_end_] = 0.0;
    mirror_z_lookup_[mirror_id_end_] = 0.0;
    mirror_y_lookup_[mirror_id_end_] = 0.0;
    mirror_normdisp_lookup_[mirror_id_end_] = 0.0;
    mirror_dhex_max_ = 0.5*scope->facetSize();
    reflec_aperture2_ = SQR(std::sqrt(reflec_aperture2_) + scope->facetSpacing());

    fp_pos_                  = scope->focalPlanePosition().cast<real_t>();
    fp_has_rot_              = scope->hasFPRotation();
    fp_rot_                  = scope->rotationReflectorToFP().cast<real_t>();
    fp_aperture2_            = 0;

    pixel_crot_              = scope->cosPixelRotation();
    pixel_srot_              = scope->sinPixelRotation();
    pixel_scaleinv_          = 1.0/scope->pixelSpacing();
    pixel_shift_x_           = scope->pixelGridShiftX();
    pixel_shift_z_           = scope->pixelGridShiftZ();
    pixel_cw_                = scope->pixelParity();

    pixel_hexid_end_         = scope->numPixelHexSites();
    pixel_id_end_            = scope->numPixels();

    calin::util::memory::aligned_calloc(pixel_id_lookup_, pixel_hexid_end_+1);
    for(int ihexid = 0; ihexid<pixel_hexid_end_; ihexid++) {
      const auto* pixel = scope->pixelByHexID(ihexid);
      if(pixel==nullptr or pixel->removed()) {
        pixel_id_lookup_[ihexid] = pixel_hexid_end_;
      } else {
        pixel_id_lookup_[ihexid] = pixel->id();
        fp_aperture2_ =
          std::max(fp_aperture2_, real_t(SQR(pixel->pos().x())+SQR(pixel->pos().z())));
      }
    }
    pixel_id_lookup_[pixel_hexid_end_] = pixel_hexid_end_;
    fp_aperture2_ = SQR(std::sqrt(fp_aperture2_) + scope->pixelSpacing());

    populate_obscuration(pre_reflection_obscuration,
      scope->all_pre_reflection_obscurations(), "pre-reflection");

    populate_obscuration(post_reflection_obscuration,
      scope->all_post_reflection_obscurations(), "post-reflection");

    populate_obscuration(camera_obscuration,
      scope->all_camera_obscurations(), "in-camera");

#if 0
    std::cout << pixel_crot_ << ' ' << pixel_srot_ << ' ' << pixel_scaleinv_ << ' '
      << pixel_shift_x_ << ' ' << pixel_shift_z_ << ' ' << pixel_cw_ << ' '
      << pixel_hexid_end_ << ' ' << pixel_id_end_ << ' '
      << fp_aperture2_ << ' ' << std::sqrt(fp_aperture2_) << '\n';
#endif
  }

  ~VCLScopeRayTracer()
  {
    free(mirror_id_lookup_);
    free(mirror_nx_lookup_);
    free(mirror_nz_lookup_);
    free(mirror_ny_lookup_);
    free(mirror_r_lookup_);
    free(mirror_x_lookup_);
    free(mirror_z_lookup_);
    free(mirror_y_lookup_);
    free(mirror_normdisp_lookup_);
    free(pixel_id_lookup_);
    for(auto* obs: pre_reflection_obscuration)free(obs);
    for(auto* obs: post_reflection_obscuration)free(obs);
    for(auto* obs: camera_obscuration)free(obs);
    if(adopt_rng_)delete rng_;
  }

  void point_telescope(const calin::simulation::vs_optics::VSOTelescope* scope) {
    global_to_reflector_off_ = scope->translationGlobalToReflector().cast<real_t>();
    global_to_reflector_rot_ = scope->rotationGlobalToReflector().cast<real_t>();
  }

  static void transform_to_scope_reflector_frame(Ray& ray,
      const calin::simulation::vs_optics::VSOTelescope* scope)
  {
    ray.translate_origin(scope->translationGlobalToReflector().cast<real_vt>());
    ray.rotate(scope->rotationGlobalToReflector().cast<real_vt>());
  }

  bool_vt trace_global_frame(bool_vt mask, Ray& ray, TraceInfo& info,
    bool do_derotation = true)
  {
    // *************************************************************************
    // ********************** RAY STARTS IN GLOBAL FRAME ***********************
    // *************************************************************************

    ray.translate_origin(global_to_reflector_off_.template cast<real_vt>());
    ray.rotate(global_to_reflector_rot_.template cast<real_vt>());
    mask = trace_reflector_frame(mask, ray, info);
    if(do_derotation) {
      ray.derotate(global_to_reflector_rot_.template cast<real_vt>());
      ray.untranslate_origin(global_to_reflector_off_.template cast<real_vt>());
    }
    return mask;
  }

  bool_vt trace_scope_centered_global_frame(bool_vt mask, Ray& ray, TraceInfo& info,
    bool do_derotation = true)
  {
    // *************************************************************************
    // *************** RAY STARTS IN SCOPE CENTERED GLOBAL FRAME ***************
    // *************************************************************************

    ray.rotate(global_to_reflector_rot_.template cast<real_vt>());
    mask = trace_reflector_frame(mask, ray, info);
    if(do_derotation) {
      ray.derotate(global_to_reflector_rot_.template cast<real_vt>());
    }
    return mask;
  }

//#define DEBUG_STATUS

  bool_vt trace_reflector_frame(bool_vt mask, Ray& ray, TraceInfo& info)
  {
    info.status = STS_MASKED_ON_ENTRY;
#ifdef DEBUG_STATUS
    std::cout << mask[0] << '/' << info.status[0];
#endif

    info.status = select(bool_int_vt(mask), STS_TRAVELLING_AWAY_REFLECTOR, info.status);
    mask &= ray.uy() < 0;
#ifdef DEBUG_STATUS
    std::cout << ' ' << mask[0] << '/' << info.status[0];
#endif

    // *************************************************************************
    // ****************** RAY STARTS IN RELECTOR COORDINATES *******************
    // *************************************************************************

    // Test for obscuration of incoming ray
    bool_vt was_obscured = false;
    real_vt ct_obscured = std::numeric_limits<real_t>::infinity();
    uint_vt hitmask = 1;
    info.pre_reflection_obs_hitmask = uint_vt(0);
    for(const auto* obs : pre_reflection_obscuration) {
      Ray ray_out;
      bool_vt was_obscured_here = obs->doesObscure(ray, ray_out, ref_index_);
      ct_obscured = vcl::select(was_obscured_here,
        vcl::min(ct_obscured, ray_out.ct()), ct_obscured);
      was_obscured |= was_obscured_here;
      info.pre_reflection_obs_hitmask |= vcl::select(bool_uint_vt(was_obscured_here), hitmask, 0);
      hitmask <<= 1;
    }

    // Remember initial ct to test reflection happens after emission
    real_vt ct0 = ray.ct();

    // Propagate to intersection with the reflector sphere (allow to go backwards a bit)
    info.status = select(bool_int_vt(mask), STS_MISSED_REFLECTOR_SPHERE, info.status);
    mask = ray.propagate_to_y_sphere_2nd_interaction_mostly_fwd_with_mask(mask,
      reflec_curvature_radius_, 0, (-2.0/CALIN_HEX_ARRAY_SQRT3)*mirror_dhex_max_, ref_index_);
#ifdef DEBUG_STATUS
    std::cout << ' ' << mask[0] << '/' << info.status[0];
#endif

    info.reflec_x     = select(mask, ray.position().x(), 0);
    info.reflec_y     = select(mask, ray.position().y(), 0);
    info.reflec_z     = select(mask, ray.position().z(), 0);

    // Test aperture
    info.status = select(bool_int_vt(mask), STS_OUTSIDE_REFLECTOR_APERTURE, info.status);
    mask &= (info.reflec_x*info.reflec_x + info.reflec_z*info.reflec_z) <= reflec_aperture2_;
#ifdef DEBUG_STATUS
    std::cout << ' ' << mask[0] << '/' << info.status[0];
#endif

    if(not horizontal_or(mask)) {
      // We outie ...
      return mask;
    }

    // Assume mirrors on hexagonal grid - use hex_array routines to find which hit
    info.status = select(bool_int_vt(mask), STS_NO_MIRROR, info.status);
    info.mirror_hexid = calin::math::hex_array::VCLReal<VCLReal>::
      xy_trans_to_hexid_scaleinv(info.reflec_x, info.reflec_z,
        reflec_crot_, reflec_srot_, reflec_scaleinv_, reflec_shift_x_, reflec_shift_z_,
        reflec_cw_);

    // Test we have a valid mirror hexid
    mask &= typename VCLReal::bool_vt(info.mirror_hexid < mirror_hexid_end_);
#ifdef DEBUG_STATUS
    std::cout << ' ' << mask[0] << '/' << info.status[0];
#endif

    if(not horizontal_or(mask)) {
      // We outie ...
      return mask;
    }

    info.mirror_hexid = select(bool_int_vt(mask), info.mirror_hexid, mirror_hexid_end_);

    // Find the mirror ID
    info.mirror_id = vcl::lookup<0x40000000>(info.mirror_hexid, mirror_id_lookup_);

    // Test we have a valid mirror id
    mask &= typename VCLReal::bool_vt(info.mirror_id < mirror_id_end_);
#ifdef DEBUG_STATUS
    std::cout << ' ' << mask[0] << '/' << info.status[0];
#endif

    vec3_vt mirror_dir;
    mirror_dir.x() = vcl::lookup<0x40000000>(info.mirror_id, mirror_nx_lookup_);
    mirror_dir.z() = vcl::lookup<0x40000000>(info.mirror_id, mirror_nz_lookup_);
#if 1
    // Is it faster to use lookup table than to compute ?
    mirror_dir.y() = vcl::lookup<0x40000000>(info.mirror_id, mirror_ny_lookup_);
#else
    mirror_dir.y() = sqrt(nmul_add(mirror_dir.z(), mirror_dir.z(),
      nmul_add( mirror_dir.x(), mirror_dir.x(),1.0)));
#endif

    real_vt mirror_r = vcl::lookup<0x40000000>(info.mirror_id, mirror_r_lookup_);

    vec3_vt mirror_pos;
    mirror_pos.x() = vcl::lookup<0x40000000>(info.mirror_id, mirror_x_lookup_);
    mirror_pos.y() = vcl::lookup<0x40000000>(info.mirror_id, mirror_y_lookup_);
    mirror_pos.z() = vcl::lookup<0x40000000>(info.mirror_id, mirror_z_lookup_);

    vec3_vt mirror_center = mirror_pos + mirror_dir * mirror_r;

    ray.translate_origin(mirror_center);

    // *************************************************************************
    // ******************* RAY IS NOW IN MIRROR COORDINATES ********************
    // *************************************************************************

    // Propagate to intersection with the mirror sphere
    info.status = select(bool_int_vt(mask), STS_MISSED_MIRROR_SPHERE, info.status);
    mask = ray.propagate_to_y_sphere_2nd_interaction_fwd_bwd_with_mask(mask,
      mirror_r, -mirror_r, ref_index_);
    mask &= ray.ct() >= ct0;
#ifdef DEBUG_STATUS
    std::cout << ' ' << mask[0] << '/' << info.status[0];
#endif

    // Impact point relative to facet attchment point
    vec3_vt ray_pos = ray.position() + mirror_center - mirror_pos;
    calin::math::geometry::VCL<VCLReal>::
      derotate_in_place_Ry(ray_pos, reflec_crot_, reflec_srot_);

    calin::math::geometry::VCL<VCLReal>::
      derotate_in_place_Ry(mirror_dir, reflec_crot_, reflec_srot_);

    calin::math::geometry::VCL<VCLReal>::
      derotate_in_place_y_to_u_Ryxy(ray_pos, mirror_dir);

    // Verify that ray impacts inside of hexagonal mirror surface
    const real_vt cos60 = 0.5;
    const real_vt sin60 = 0.5*CALIN_HEX_ARRAY_SQRT3;

    const real_vt x_cos60 = ray_pos.x() * cos60;
    const real_vt z_sin60 = ray_pos.z() * sin60;

    const real_vt dhex_pos60 = abs(x_cos60 - z_sin60);
    const real_vt dhex_neg60 = abs(x_cos60 + z_sin60);

    info.status = select(bool_int_vt(mask), STS_MISSED_MIRROR_EDGE, info.status);
    mask &= max(max(dhex_pos60, dhex_neg60), abs(ray_pos.x())) < mirror_dhex_max_;
#ifdef DEBUG_STATUS
    std::cout << ' ' << mask[0] << '/' << info.status[0];
#endif

    // Calculate mirror normal at impact point
    vec3_vt mirror_normal = -ray.position() * (1.0/mirror_r);

    // Scatter the normal to account for the spot size ot the focal length of the
    // radius. The spot size is given as the DIAMETER at the focal distance.
    // Must divide by 2 (for reflection) and another 2 for diameter -> radius
    real_vt mirror_normal_dispersion =
      vcl::lookup<0x40000000>(info.mirror_id, mirror_normdisp_lookup_);

    // Scatter the normal direction randomly
    calin::math::geometry::VCL<VCLReal>::scatter_direction_in_place(
      mirror_normal, mirror_normal_dispersion, *rng_);

    // Reflect ray
#if 1
    info.mirror_n_dot_u = ray.direction().dot(mirror_normal);
    ray.mutable_direction() -= mirror_normal * select(mask, 2.0*info.mirror_n_dot_u, 0);
#else
    // Do not use this function any longer as we wish to keep u dot n
    ray.reflect_from_surface_with_mask(mask, info.mirror_normal_scattered);
#endif

    // Translate back to reflector frame
    ray.untranslate_origin(mirror_center);

    info.mirror_x     = select(mask, ray.position().x(), 0);
    info.mirror_y     = select(mask, ray.position().y(), 0);
    info.mirror_z     = select(mask, ray.position().z(), 0);

    // *************************************************************************
    // *************** RAY IS NOW BACK IN REFLECTOR COORDINATES ****************
    // *************************************************************************

    // Finish checking obscuration before mirror hit
    info.status = select(bool_int_vt(mask), STS_OBSCURED_BEFORE_MIRROR, info.status);
    mask &= ~(was_obscured & (ct_obscured < ray.ct()));

    if(not horizontal_or(mask)) {
      // We outie ...
      return mask;
    }

    // Test for obscuration on way to focal plane - first with obscurations
    // that are given in reflector coordinates (telescope arms etc)
    was_obscured = false;
    ct_obscured = std::numeric_limits<real_t>::infinity();
    hitmask = uint_vt(1);
    info.post_reflection_obs_hitmask = uint_vt(0);
    for(const auto* obs : post_reflection_obscuration) {
      Ray ray_out;
      bool_vt was_obscured_here = obs->doesObscure(ray, ray_out, ref_index_);
      ct_obscured = vcl::select(was_obscured_here,
        vcl::min(ct_obscured, ray_out.ct()), ct_obscured);
      was_obscured |= was_obscured_here;
      info.post_reflection_obs_hitmask |= vcl::select(bool_uint_vt(was_obscured_here), hitmask, 0);
      hitmask <<= 1;
    }

    // Refract in window

    ray.translate_origin(fp_pos_.template cast<real_vt>());
    if(fp_has_rot_)ray.rotate(fp_rot_.template cast<real_vt>());

    // *************************************************************************
    // ***************** RAY IS NOW IN FOCAL PLANE COORDINATES *****************
    // *************************************************************************

    // Test for obscuration on way to focal plane - second with obscurations
    // that are given in focal plane coordinates
    hitmask = uint_vt(1);
    info.camera_obs_hitmask = uint_vt(0);
    for(const auto* obs : camera_obscuration) {
      Ray ray_out;
      bool_vt was_obscured_here = obs->doesObscure(ray, ray_out, ref_index_);
      ct_obscured = vcl::select(was_obscured_here,
        vcl::min(ct_obscured, ray_out.ct()), ct_obscured);
      was_obscured |= was_obscured_here;
      info.camera_obs_hitmask |= vcl::select(bool_uint_vt(was_obscured_here), hitmask, 0);
      hitmask <<= 1;
    }

    // Propagate to focal plane
    info.status = select(bool_int_vt(mask), STS_TRAVELLING_AWAY_FROM_FOCAL_PLANE, info.status);
    mask = ray.propagate_to_y_plane_with_mask(mask, 0, false, ref_index_);
#ifdef DEBUG_STATUS
    std::cout << ' ' << mask[0] << '/' << info.status[0];
#endif

    // Finish checking obscuration after mirror reflection
    info.status = select(bool_int_vt(mask), STS_OBSCURED_BEFORE_FOCAL_PLANE, info.status);
    mask &= ~(was_obscured & (ct_obscured < ray.ct()));

    // We good, record position on focal plane etc
    info.fplane_x = select(mask, ray.x(), 0);
    info.fplane_z = select(mask, ray.z(), 0);
    info.fplane_t = select(mask, ray.time(), 0);
    info.fplane_ux = select(mask, ray.ux(), 0);
    info.fplane_uy = select(mask, ray.uy(), 0);
    info.fplane_uz = select(mask, ray.uz(), 0);

    info.status = select(bool_int_vt(mask), STS_OUTSIDE_FOCAL_PLANE_APERTURE, info.status);
    mask &= (info.fplane_x*info.fplane_x + info.fplane_z*info.fplane_z) <= fp_aperture2_;
#ifdef DEBUG_STATUS
    std::cout << ' ' << mask[0] << '/' << info.status[0];
#endif

    info.pixel_hexid =
    calin::math::hex_array::VCLReal<VCLReal>::
      xy_trans_to_hexid_scaleinv(info.fplane_x, info.fplane_z,
        pixel_crot_, pixel_srot_, pixel_scaleinv_, pixel_shift_x_, pixel_shift_z_,
        pixel_cw_);

    // Test we have a valid pixel hexid
    info.status = select(bool_int_vt(mask), STS_TS_NO_PIXEL, info.status);
    mask &= typename VCLReal::bool_vt(info.pixel_hexid < pixel_hexid_end_);
#ifdef DEBUG_STATUS
    std::cout << ' ' << mask[0] << '/' << info.status[0];
#endif

    // Find the pixel ID
    info.pixel_id =
      vcl::lookup<0x40000000>(select(bool_int_vt(mask), info.pixel_hexid, pixel_hexid_end_),
        pixel_id_lookup_);

    mask &= typename VCLReal::bool_vt(info.pixel_id < pixel_id_end_);
#ifdef DEBUG_STATUS
    std::cout << ' ' << mask[0] << '/' << info.status[0];
#endif

    info.status = select(bool_int_vt(mask), STS_TS_FOUND_PIXEL, info.status);
#ifdef DEBUG_STATUS
    std::cout << ' ' << mask[0] << '/' << info.status[0];
#endif

    if(fp_has_rot_)ray.derotate(fp_rot_.template cast<real_vt>());
    ray.untranslate_origin(fp_pos_.template cast<real_vt>());

    // *************************************************************************
    // ************ RAY IS NOW BACK IN REFLECTOR COORDINATES AGAIN *************
    // *************************************************************************

#ifdef DEBUG_STATUS
    std::cout << '\n';
#endif
    return mask;
  }

private:

  void populate_obscuration(std::vector<VCLObscuration<VCLReal>*>& to,
    const std::vector<const calin::simulation::vs_optics::VSOObscuration*>& from,
    const std::string& type)
  {
    using namespace calin::simulation::vs_optics;
    for(const auto* obs : from) {
      if(const auto* dc_obs = dynamic_cast<const VSOAlignedBoxObscuration*>(obs)) {
        to.push_back(new VCLAlignedBoxObscuration<VCLReal>(*dc_obs));
      } else if(const auto* dc_obs = dynamic_cast<const VSOAlignedRectangularAperture*>(obs)) {
        to.push_back(new VCLAlignedRectangularAperture<VCLReal>(*dc_obs));
      } else if(const auto* dc_obs = dynamic_cast<const VSOAlignedCircularAperture*>(obs)) {
        to.push_back(new VCLAlignedCircularAperture<VCLReal>(*dc_obs));
      } else if(const auto* dc_obs = dynamic_cast<const VSOTubeObscuration*>(obs)) {
        to.push_back(new VCLTubeObscuration<VCLReal>(*dc_obs));
      } else {
        throw std::runtime_error("Unsupported " + type + " obscuration type");
      }
    }
  }

  vec3_t          global_to_reflector_off_;
  mat3_t          global_to_reflector_rot_;
  real_t          ref_index_;

  real_t          reflec_curvature_radius_;
  real_t          reflec_aperture2_;
  real_t          reflec_crot_;
  real_t          reflec_srot_;
  real_t          reflec_scaleinv_;
  real_t          reflec_shift_x_;
  real_t          reflec_shift_z_;
  bool            reflec_cw_;

  int_t           mirror_hexid_end_;
  int_t           mirror_id_end_;
  int_t*          mirror_id_lookup_ = nullptr;
  real_t*         mirror_nx_lookup_ = nullptr;
  real_t*         mirror_nz_lookup_ = nullptr;
  real_t*         mirror_ny_lookup_ = nullptr;
  real_t*         mirror_r_lookup_ = nullptr;
  real_t*         mirror_x_lookup_ = nullptr;
  real_t*         mirror_z_lookup_ = nullptr;
  real_t*         mirror_y_lookup_ = nullptr;
  real_t          mirror_dhex_max_;
  real_t*         mirror_normdisp_lookup_ = nullptr;

  vec3_t          fp_pos_;
  bool            fp_has_rot_;
  mat3_t          fp_rot_;
  real_t          fp_aperture2_;

  real_t          pixel_crot_;
  real_t          pixel_srot_;
  real_t          pixel_scaleinv_;
  real_t          pixel_shift_x_;
  real_t          pixel_shift_z_;
  bool            pixel_cw_;

  int_t           pixel_hexid_end_;
  int_t           pixel_id_end_;
  int_t*          pixel_id_lookup_ = nullptr;

  std::vector<VCLObscuration<VCLReal>*> pre_reflection_obscuration;
  std::vector<VCLObscuration<VCLReal>*> post_reflection_obscuration;
  std::vector<VCLObscuration<VCLReal>*> camera_obscuration;

  RNG* rng_ = nullptr;
  bool adopt_rng_ = false;
};

template<typename VCLReal> class alignas(VCLReal::vec_bytes) VCLAlignedBoxObscuration:
  public VCLObscuration<VCLReal>
{
public:
  using typename VCLObscuration<VCLReal>::real_vt;
  using typename VCLObscuration<VCLReal>::bool_vt;
  using typename VCLObscuration<VCLReal>::vec3_vt;
  using typename VCLObscuration<VCLReal>::Ray;
  using typename VCLObscuration<VCLReal>::vec3_t;
  using typename VCLObscuration<VCLReal>::real_t;

  VCLAlignedBoxObscuration(const vec3_t& max_corner, const vec3_t& min_corner):
    VCLObscuration<VCLReal>(), min_corner_(min_corner), max_corner_(max_corner)
  {
    // nothing to see here
  }
  VCLAlignedBoxObscuration(const calin::simulation::vs_optics::VSOAlignedBoxObscuration& o):
    VCLObscuration<VCLReal>(),
    min_corner_(o.min_corner().template cast<real_t>()),
    max_corner_(o.max_corner().template cast<real_t>()) /* still need cast for double -> float */
  {
    // nothing to see here
  }
  virtual ~VCLAlignedBoxObscuration()
  {
    // nothing to see here
  }
  bool_vt doesObscure(const Ray& ray_in, Ray& ray_out, real_vt n) const override
  {
    real_vt tmin;
    real_vt tmax;
    bool_vt mask = ray_in.box_has_future_intersection(tmin, tmax,
      min_corner_.template cast<real_vt>(), max_corner_.template cast<real_vt>());
    ray_out = ray_in;
    ray_out.propagate_dist_with_mask(mask & (tmin>0), tmin, n);
    return mask;
  }
  virtual VCLAlignedBoxObscuration<VCLReal>* clone() const override
  {
    return new VCLAlignedBoxObscuration<VCLReal>(*this);
  }
private:
  vec3_t min_corner_;
  vec3_t max_corner_;
};

template<typename VCLReal> class alignas(VCLReal::vec_bytes) VCLAlignedCircularAperture:
  public VCLObscuration<VCLReal>
{
public:
  using typename VCLObscuration<VCLReal>::real_vt;
  using typename VCLObscuration<VCLReal>::bool_vt;
  using typename VCLObscuration<VCLReal>::vec3_vt;
  using typename VCLObscuration<VCLReal>::Ray;
  using typename VCLObscuration<VCLReal>::vec3_t;
  using typename VCLObscuration<VCLReal>::real_t;

  VCLAlignedCircularAperture(const vec3_t& center, const real_t& diameter, bool invert = false):
    VCLObscuration<VCLReal>(), center_(center),
    radius_sq_(0.25*diameter*diameter), inverted_(invert)
  {
    // nothing to see here
  }
  VCLAlignedCircularAperture(const calin::simulation::vs_optics::VSOAlignedCircularAperture& o):
    VCLObscuration<VCLReal>(),
    center_(o.center().template cast<real_t>()), radius_sq_(o.radius_sq()),
    inverted_(o.inverted())
  {
    // nothing to see here
  }
  virtual ~VCLAlignedCircularAperture()
  {
    // nothing to see here
  }
  bool_vt doesObscure(const Ray& ray_in, Ray& ray_out, real_vt n) const override
  {
    using calin::math::special::SQR;
    ray_out = ray_in;
    bool_vt ray_reaches_plane = ray_out.propagate_to_y_plane(-center_.y(),
      /*time_reversal_ok=*/ false, n);
    const real_vt r2 =
      SQR(ray_out.x()-center_.x())+SQR(ray_out.z()-center_.z())-radius_sq_;
    if(inverted_) {
      return ray_reaches_plane & (r2<=0);
    } else {
      return ray_reaches_plane & (r2>0);
    }
  }
  virtual VCLAlignedCircularAperture<VCLReal>* clone() const override
  {
    return new VCLAlignedCircularAperture<VCLReal>(*this);
  }
private:
  vec3_t center_;
  real_t radius_sq_;
  bool inverted_;
};

template<typename VCLReal> class alignas(VCLReal::vec_bytes) VCLAlignedRectangularAperture:
  public VCLObscuration<VCLReal>
{
public:
  using typename VCLObscuration<VCLReal>::real_vt;
  using typename VCLObscuration<VCLReal>::bool_vt;
  using typename VCLObscuration<VCLReal>::vec3_vt;
  using typename VCLObscuration<VCLReal>::Ray;
  using typename VCLObscuration<VCLReal>::vec3_t;
  using typename VCLObscuration<VCLReal>::real_t;

  VCLAlignedRectangularAperture(const vec3_t& center,
      const real_t& flat_to_flat_x, const real_t& flat_to_flat_z, bool invert = false):
    VCLObscuration<VCLReal>(), center_(center),
    flat_to_flat_x_2_(0.5*flat_to_flat_x), flat_to_flat_z_2_(0.5*flat_to_flat_z),
    inverted_(invert)
  {
    // nothing to see here
  }
  VCLAlignedRectangularAperture(const calin::simulation::vs_optics::VSOAlignedRectangularAperture& o):
    VCLObscuration<VCLReal>(),
    center_(o.center().template cast<real_t>()),
    flat_to_flat_x_2_(o.flat_to_flat_x_2()), flat_to_flat_z_2_(o.flat_to_flat_z_2()),
    inverted_(o.inverted())
  {
    // nothing to see here
  }
  virtual ~VCLAlignedRectangularAperture()
  {
    // nothing to see here
  }
  bool_vt doesObscure(const Ray& ray_in, Ray& ray_out, real_vt n) const override
  {
    using calin::math::special::SQR;
    ray_out = ray_in;
    bool_vt ray_reaches_plane = ray_out.propagate_to_y_plane(-center_.y(),
      /*time_reversal_ok=*/ false, n);
    const real_vt dx = vcl::abs(ray_out.x()-center_.x()) - flat_to_flat_x_2_;
    const real_vt dz = vcl::abs(ray_out.z()-center_.z()) - flat_to_flat_z_2_;
    if(inverted_) {
      return ray_reaches_plane & (vcl::max(dx,dz)<=0);
    } else {
      return ray_reaches_plane & (vcl::max(dx,dz)>0);
    }
  }
  virtual VCLAlignedRectangularAperture<VCLReal>* clone() const override
  {
    return new VCLAlignedRectangularAperture<VCLReal>(*this);
  }
private:
  vec3_t center_;
  real_t flat_to_flat_x_2_;
  real_t flat_to_flat_z_2_;
  bool inverted_;
};

template<typename VCLReal> class alignas(VCLReal::vec_bytes) VCLTubeObscuration:
  public VCLObscuration<VCLReal>
{
public:
  using typename VCLObscuration<VCLReal>::real_vt;
  using typename VCLObscuration<VCLReal>::bool_vt;
  using typename VCLObscuration<VCLReal>::vec3_vt;
  using typename VCLObscuration<VCLReal>::Ray;
  using typename VCLObscuration<VCLReal>::vec3_t;
  using typename VCLObscuration<VCLReal>::mat3_t;
  using typename VCLObscuration<VCLReal>::real_t;

  VCLTubeObscuration(const vec3_t& x1, const vec3_t& x2, const real_t& radius):
    VCLObscuration<VCLReal>(), x1_(x1), x2_(x2), r_(radius),
    xc_(0.5*(x1_+x2_)), r2_(r_*r_)
  {
    nhat_ = x2_-x1_;
    half_length_ = 0.5*nhat_.norm();
    nhat_.normalize();
  }

  VCLTubeObscuration(const calin::simulation::vs_optics::VSOTubeObscuration& o):
    VCLTubeObscuration(o.end1_pos().template cast<real_t>(),
      o.end2_pos().template cast<real_t>(), o.radius())
  {
    // nothing to see here
  }

  virtual ~VCLTubeObscuration()
  {
    // nothing to see here
  }

  bool_vt doesObscure(const Ray& ray_in, Ray& ray_out, real_vt n) const override
  {
    constexpr real_t inf = std::numeric_limits<real_t>::infinity();

    const vec3_vt u = nhat_.template cast<real_vt>();
    const vec3_vt v = ray_in.direction();

    const vec3_vt D0 = ray_in.position() - xc_.template cast<real_vt>();
    const real_vt u_dot_v = u.dot(v);
    const real_vt u_dot_D0 = u.dot(D0);
    const real_vt v_dot_D0 = v.dot(D0);

    // Calculate quadratic equation coefficients and discriminant
    const real_vt a = vcl::nmul_add(u_dot_v, u_dot_v, 1.0); // 1-(u.v)^2
    const real_vt b = 2.0*nmul_add(u_dot_D0, u_dot_v, v_dot_D0); // D0.v - (D0.u)*(u.v)
    const real_vt c = nmul_add(u_dot_D0, u_dot_D0, D0.squaredNorm()) - r2_; // D0^2 - (D0.u)^2 - r^2

    const real_vt disc = nmul_add(4.0*a, c, b*b); // b^2 - 4*a*c

    // Find rays that come close enough to possibly obscure - rest can be ignored
    // Must handle case of rays parallel to cylinder (a=b=0)
    const bool_vt intersects_cylinder = (vcl::select(a>0, disc, -c) >= 0);

    if(not horizontal_or(intersects_cylinder)) {
      // Fast exit for case where closest approach for all rays with cylinder is greater than r^2
      return false;
    }

    const real_vt q = -0.5*(b + sign_combine(vcl::sqrt(vcl::max(disc, 0)), b));

    // Calculate time ray passes through cylinder - if parallel to them then
    // set times to tp1/2=-/+inf if ray between planes, tp1=tp2=+inf if outside planes
    const real_vt tc1 = vcl::select(a>0, c/q, -inf);
    const real_vt tc2 = vcl::select(a>0, q/a, inf);

    // Calculate time ray passes through end planes - if parallel to them then
    // set times to tp1/2=-/+inf if ray between planes, tp1=tp2=+inf if outside planes
    const real_vt u_dot_v_inv = real_vt(1.0)/u_dot_v;
    const real_vt tp1 = vcl::select(u_dot_v==0, vcl::sign_combine(inf, vcl::abs(u_dot_D0)-half_length_), (-u_dot_D0 - half_length_)*u_dot_v_inv);
    const real_vt tp2 = vcl::select(u_dot_v==0, inf, (-u_dot_D0 + half_length_)*u_dot_v_inv);

    const real_vt t_in = vcl::max(vcl::min(tc1, tc2), vcl::min(tp1, tp2));
    const real_vt t_out = vcl::min(vcl::max(tc1, tc2), vcl::max(tp1, tp2));

    const real_vt t_prop = vcl::max(t_in, 0);
    const bool_vt does_obscure = t_out > t_prop;

    // using calin::util::log::LOG;
    // using calin::util::log::INFO;
    // if(does_obscure[0]) {
    //   LOG(INFO) << ray_in.x()[0] << ' ' << ray_in.y()[0] << ' '
    //     << tc1[0] << ' ' << tc2[0] << ' ' << tp1[0] << ' ' << tp2[0];
    // }

    ray_out = ray_in;
    ray_out.propagate_dist_with_mask(does_obscure, t_prop, n);

    return does_obscure;
  }

  virtual VCLTubeObscuration<VCLReal>* clone() const override
  {
    return new VCLTubeObscuration<VCLReal>(*this);
  }

private:
  vec3_t x1_;
  vec3_t x2_;
  real_t r_;

  vec3_t xc_;
  vec3_t nhat_;

  real_t r2_;
  real_t half_length_;
};

// std::ostream& operator <<(std::ostream& stream, const VSORayTracer::TraceInfo& o);

#endif

bool test_tube_obscuration(const Eigen::Vector3d& x1, const Eigen::Vector3d& x2, double radius,
  const Eigen::Vector3d& r, const Eigen::Vector3d& u);

} } } // namespace calin::simulations::vcl_raytracer
