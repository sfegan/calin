/*

   calin/simulation/vso_array.cpp -- Stephen Fegan -- 2015-11-25

   Class for array of telescopes

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

//-*-mode:c++; mode:font-lock;-*-

/*! \file ArrayParameters.cpp
  Array code file

  \author  Stephen Fegan               \n
           UCLA                        \n
	   sfegan@astro.ucla.edu       \n

  \date    12/05/2004
  \version 0.2
  \note
*/

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <set>
#include <map>

#include <math/hex_array.hpp>
#include <math/vector3d_util.hpp>
#include <util/string.hpp>
#include <simulation/vso_array.hpp>

using namespace calin::simulation::vs_optics;
using namespace calin::ix::simulation::vs_optics;

VSOArray::VSOArray():
    fLatitude(), fLongitude(), fAltitude(), /* fSpacing(), fArrayParity(), */
    fTelescopes() /* , fTelescopesByHexID() */
{
  // nothing to see here
}

VSOArray::VSOArray(const VSOArray& array):
  fLatitude(array.fLatitude), fLongitude(array.fLongitude), fAltitude(array.fAltitude),
  fTelescopes(array.fTelescopes.size())
{
  for(unsigned iscope=0;iscope<array.fTelescopes.size();++iscope) {
    fTelescopes[iscope] = new VSOTelescope(*array.fTelescopes[iscope]);
  }
}

VSOArray::~VSOArray()
{
  for(std::vector<VSOTelescope*>::iterator i = fTelescopes.begin();
      i!=fTelescopes.end(); i++)delete *i;
}

// ****************************************************************************
// General functions
// ****************************************************************************

bool VSOArray::pointTelescopes(const Eigen::Vector3d& v)
{
  if(v.squaredNorm() == 0)
    return false;

  bool good = true;
  for(std::vector<VSOTelescope*>::iterator i = fTelescopes.begin();
      i!=fTelescopes.end(); i++)good &= (*i)->pointTelescope(v);

  return good;
}

bool VSOArray::
pointTelescopesAzEl(const double az_rad, const double el_rad)
{
  bool good = true;
  for(std::vector<VSOTelescope*>::iterator i = fTelescopes.begin();
      i!=fTelescopes.end(); i++)
    good &= (*i)->pointTelescopeAzEl(az_rad,el_rad);
  return good;
}

bool VSOArray::
pointTelescopesAzElPhi(const double az_rad, const double el_rad, const double phi_rad)
{
  bool good = true;
  for(std::vector<VSOTelescope*>::iterator i = fTelescopes.begin();
      i!=fTelescopes.end(); i++)
    good &= (*i)->pointTelescopeAzElPhi(az_rad,el_rad,phi_rad);
  return good;
}


// ****************************************************************************
// Array creation
// ****************************************************************************

void VSOArray::
generateFromArrayParameters(const IsotropicDCArrayParameters& param,
                            math::rng::RNG& rng)
{
  // Array
  fLatitude    = param.array_origin().latitude()/180.0*M_PI;
  fLongitude   = param.array_origin().longitude()/180.0*M_PI;
  fAltitude    = param.array_origin().elevation();

  std::vector<Eigen::Vector3d> scope_pos;

  if(param.array_layout_case() ==
     IsotropicDCArrayParameters::kHexArrayLayout)
  {
    const auto& layout = param.hex_array_layout();
    double spacing     = layout.scope_spacing();
    bool array_parity  = layout.scope_labeling_parity();

    unsigned num_telescopes =
        math::hex_array::ringid_to_nsites_contained(layout.num_scope_rings());
    std::set<unsigned> scopes_missing;
    for(auto hexid : layout.scope_missing_list())
      scopes_missing.insert(hexid);

    for(unsigned hexid=0; hexid<num_telescopes; hexid++)
      if(scopes_missing.find(hexid) == scopes_missing.end())
      {
        Eigen::Vector3d pos;
        math::hex_array::hexid_to_xy(hexid, pos.x(), pos.y());
        if(array_parity)pos.x() = -pos.x();
        pos.x()  = pos.x() * spacing +
                   rng.normal() * layout.scope_position_dispersion_xy();
        pos.y()  = pos.y() * spacing +
                   rng.normal() * layout.scope_position_dispersion_xy();
        pos.z() += rng.normal() * layout.scope_position_dispersion_z();
        scope_pos.push_back(pos);
      }
  }
  else if(param.array_layout_case() ==
          IsotropicDCArrayParameters::kPrescribedArrayLayout)
  {
    for(auto pos : param.prescribed_array_layout().scope_positions())
      scope_pos.emplace_back(calin::math::vector3d_util::from_proto(pos));
  }
  else
  {
    assert(0);
  }

  // Mirrors
  unsigned num_hex_mirror_rings = param.reflector().facet_num_hex_rings();
  if(param.reflector().aperture() > 0)
  {
    unsigned aperture_num_hex_mirror_rings =
        std::floor(param.reflector().aperture()/
           (2.0*param.reflector().facet_spacing()*CALIN_HEX_ARRAY_SQRT3*0.5))+2;
    if(aperture_num_hex_mirror_rings<num_hex_mirror_rings)
      num_hex_mirror_rings = aperture_num_hex_mirror_rings;
  }

  // Camera
  Eigen::Vector3d camera_fp_trans(
    calin::math::vector3d_util::from_proto(param.focal_plane().translation()));

  double FoV =
      2.0*atan(param.focal_plane().camera_diameter()/
               (2.0*camera_fp_trans.norm()))*180.0/M_PI;

  if(param.focal_plane().field_of_view() > 0) {
    FoV = param.focal_plane().field_of_view();
  }

  for(unsigned i=0; i<scope_pos.size(); i++)
    {
      // Position
      Eigen::Vector3d pos(scope_pos[i]);

      std::vector<VSOObscuration*> obsvec_pre;
      for(const auto& obs : param.pre_reflection_obscuration())
        obsvec_pre.push_back(VSOObscuration::create_from_proto(obs));
      std::vector<VSOObscuration*> obsvec_post;
      for(const auto& obs : param.post_reflection_obscuration())
        obsvec_post.push_back(VSOObscuration::create_from_proto(obs));
      std::vector<VSOObscuration*> obsvec_cam;
      for(const auto& obs : param.camera_obscuration())
        obsvec_cam.push_back(VSOObscuration::create_from_proto(obs));

      double win_front = 0;
      double win_radius = 0;
      double win_thickness = 0;
      double win_n = 0;

      if(param.has_spherical_window()
        and param.spherical_window().outer_radius()>=0
        and param.spherical_window().thickness()>0
        and param.spherical_window().refractive_index()>0)
      {
        win_front = param.spherical_window().front_y_coord();
        if(param.spherical_window().outer_radius()==0 or
            std::isinf(param.spherical_window().outer_radius())) {
          win_radius = 0;
        } else {
          win_radius = param.spherical_window().outer_radius();
        }
        win_thickness = param.spherical_window().thickness();
        win_n = param.spherical_window().refractive_index();
      }

      VSOTelescope* telescope =
      	new VSOTelescope(fTelescopes.size(), pos,
      			 param.reflector_frame().fp_offset()*M_PI/180.0,
             param.reflector_frame().alpha_x()*M_PI/180.0,
             param.reflector_frame().alpha_y()*M_PI/180.0,
             param.reflector_frame().altaz().altitude()*M_PI/180.0,
             param.reflector_frame().altaz().azimuth()*M_PI/180.0,
      			 calin::math::vector3d_util::from_proto(param.reflector_frame().translation()),
             param.reflector_frame().azimuth_elevation_axes_separation(),
      			 param.reflector().curvature_radius(),
             param.reflector().aperture(),
             param.reflector().facet_spacing(),
             param.reflector().facet_size(),
             param.reflector_frame().optic_axis_rotation()*M_PI/180.0,
             param.reflector().facet_grid_shift_x(),
             param.reflector().facet_grid_shift_z(),
             num_hex_mirror_rings,
             0.0, Eigen::Vector3d::Zero(), /*param.reflector().reflector_ip(),*/
             param.reflector().facet_labeling_parity(),
	           camera_fp_trans,
             param.reflector().alignment_image_plane() <= 0 ? camera_fp_trans[1] : param.reflector().alignment_image_plane(),
             param.focal_plane().camera_diameter(),
             FoV,
             param.pixel().cone_inner_diameter(),
             param.pixel().spacing(),
             param.pixel().grid_rotation()*M_PI/180.0,
             param.pixel().pixel_grid_shift_x(),
             param.pixel().pixel_grid_shift_z(),
             param.pixel().cone_survival_prob(),
             calin::math::vector3d_util::from_scaled_proto(param.focal_plane().rotation(), M_PI/180.0),
             0.0,
             param.pixel().pixel_labeling_parity(),
             win_front, win_radius, win_thickness, win_n,
	           obsvec_pre,
             obsvec_post,
             obsvec_cam
		         );

      telescope->populateMirrorsAndPixelsRandom(param,rng);

      fTelescopes.push_back(telescope);
    }

  return;
}

calin::ix::simulation::vs_optics::VSOArrayData* VSOArray::
dump_as_proto(calin::ix::simulation::vs_optics::VSOArrayData* d) const
{
  if(d == nullptr)d = new calin::ix::simulation::vs_optics::VSOArrayData;
  d->mutable_array_origin()->set_latitude(fLatitude*180.0/M_PI);
  d->mutable_array_origin()->set_longitude(fLongitude*180.0/M_PI);
  d->mutable_array_origin()->set_elevation(fAltitude);
  for(const auto* scope : fTelescopes)
    scope->dump_as_proto(d->add_telescope());
  return d;
}

VSOArray* VSOArray::
create_from_proto(const ix::simulation::vs_optics::VSOArrayData& d)
{
  VSOArray* array = new VSOArray;
  array->fLatitude  = d.array_origin().latitude()*M_PI/180.0;
  array->fLongitude = d.array_origin().longitude()*M_PI/180.0;
  array->fAltitude  = d.array_origin().elevation();
  for(const auto& scope : d.telescope())
    array->fTelescopes.push_back(VSOTelescope::create_from_proto(scope));
  return array;
}

calin::ix::iact_data::instrument_layout::ArrayLayout*
calin::simulation::vs_optics::dc_parameters_to_array_layout(
  const ix::simulation::vs_optics::IsotropicDCArrayParameters& param,
  calin::ix::iact_data::instrument_layout::ArrayLayout* d)
{
  if(d == nullptr)d = new calin::ix::iact_data::instrument_layout::ArrayLayout;
  d->set_array_type(calin::ix::iact_data::instrument_layout::ArrayLayout::NO_ARRAY);
  *d->mutable_array_origin() = param.array_origin();

  std::vector<Eigen::Vector3d> scope_pos;
  if(param.array_layout_case() ==
     IsotropicDCArrayParameters::kHexArrayLayout)
  {
    const auto& layout = param.hex_array_layout();
    double spacing     = layout.scope_spacing();
    bool array_parity  = layout.scope_labeling_parity();

    unsigned num_telescopes =
        math::hex_array::ringid_to_nsites_contained(layout.num_scope_rings());
    std::set<unsigned> scopes_missing;
    for(auto hexid : layout.scope_missing_list())
      scopes_missing.insert(hexid);

    for(unsigned hexid=0; hexid<num_telescopes; hexid++)
      if(scopes_missing.find(hexid) == scopes_missing.end())
      {
        Eigen::Vector3d pos;
        math::hex_array::hexid_to_xy(hexid, pos.x(), pos.y());
        if(array_parity)pos.x() = -pos.x();
        pos.x()  = pos.x() * spacing;
        pos.y()  = pos.y() * spacing;
        scope_pos.push_back(pos);
      }
  }
  else if(param.array_layout_case() ==
          IsotropicDCArrayParameters::kPrescribedArrayLayout)
  {
    for(auto pos : param.prescribed_array_layout().scope_positions())
      scope_pos.emplace_back(calin::math::vector3d_util::from_proto(pos));
  }
  else
  {
    assert(0);
  }

  for(unsigned i=0; i<scope_pos.size(); i++)
    dc_parameters_to_telescope_layout(param, i, scope_pos[i], d->add_telescopes());

  return d;
}

std::string VSOArray::banner(const std::string& indent0, const std::string& indentN) const
{
  using calin::util::string::double_to_string_with_commas;
  std::map<std::string, unsigned> scope_banners;
  double xmin = std::numeric_limits<double>::infinity();
  double xmax = -std::numeric_limits<double>::infinity();
  double ymin = std::numeric_limits<double>::infinity();
  double ymax = -std::numeric_limits<double>::infinity();
  for(const auto* scope : fTelescopes) {
    xmin = std::min(xmin, scope->pos()[0]);
    xmax = std::max(xmax, scope->pos()[0]);
    ymin = std::min(ymin, scope->pos()[1]);
    ymax = std::max(ymax, scope->pos()[1]);
    scope_banners[scope->banner()]++;
  }
  double A = (xmax-xmin)*(ymax-ymin)*1e-10;
  std::ostringstream stream;
  stream << indent0 << "Array of " << this->numTelescopes()
    << " DC-like telescopes covering "
    << double_to_string_with_commas(A,3) << " km^2.";
  for(const auto& scope_banner : scope_banners) {
    stream << '\n' << indentN << scope_banner.second << " x " << scope_banner.first;
  }
  return stream.str();
}
