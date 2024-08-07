//-*-mode:swig;-*-

/*

   calin/simulation/vs_optics.i -- Stephen Fegan -- 2015-12-17

   SWIG interface file for calin.simulation.vs_optics

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

%module (package="calin.simulation") vs_optics
%feature(autodoc,2);

%{
#include "simulation/vso_pixel.hpp"
#include "simulation/vso_mirror.hpp"
#include "simulation/vso_obscuration.hpp"
#include "simulation/vso_telescope.hpp"
#include "simulation/vso_array.hpp"
#include "simulation/vso_raytracer.hpp"
#include "simulation/vcl_raytracer.hpp"
#define SWIG_FILE_WITH_INIT
  %}

%init %{
  import_array();
%}

%include "calin_typemaps.i"
%import "calin_global_definitions.i"

%import "math/ray.i"
%import "math/rng.i"

%import "iact_data/instrument_layout.pb.i"
%import "simulation/vs_optics.pb.i"

%newobject *::dump_as_proto() const;
%newobject *::create_from_proto;
%newobject *::convert_to_telescope_layout() const;
%newobject dc_parameters_to_telescope_layout;
%newobject dc_parameters_to_array_layout;

%include "simulation/vso_pixel.hpp"
%template(VectorVSOPixel) std::vector<calin::simulation::vs_optics::VSOPixel*>;
%include "simulation/vso_mirror.hpp"
%template(VectorVSOMirror) std::vector<calin::simulation::vs_optics::VSOMirror*>;
%include "simulation/vso_obscuration.hpp"
%template(VectorVSOObscuration) std::vector<calin::simulation::vs_optics::VSOObscuration*>;

%apply Eigen::Vector3d& INOUT { Eigen::Vector3d& v };

%include "simulation/vso_telescope.hpp"
%template(VectorVSOTelescope) std::vector<calin::simulation::vs_optics::VSOTelescope*>;
%include "simulation/vso_array.hpp"
%include "simulation/vso_raytracer.hpp"
%include "simulation/vcl_raytracer.hpp"
