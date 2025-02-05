//-*-mode:swig;-*-

/*

   calin/simulation/atmosphere.i -- Stephen Fegan -- 2015-06-11

   SWIG interface file for calin.simulation.atmosphere

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

%module (package="calin.simulation") atmosphere
%feature(autodoc,2);

%{
#include "simulation/atmosphere.hpp"
#define SWIG_FILE_WITH_INIT
  %}

%init %{
  import_array();
%}

%include "calin_typemaps.i"
%import "calin_global_definitions.i"

%import "math/spline_interpolation.i"
%import "math/ray.i"
%import "simulation/atmosphere.pb.i"

%template (VectorAtmSlice) std::vector<calin::simulation::atmosphere::AtmSlice>;
%template (VectorLayeredAtmosphereLevel) std::vector<calin::simulation::atmosphere::LayeredAtmosphereLevel>;

%apply double &OUTPUT { double& n_minus_one };
%apply double &OUTPUT { double& propagation_ct_correction };

%newobject clone;

%include "simulation/atmosphere.hpp"
