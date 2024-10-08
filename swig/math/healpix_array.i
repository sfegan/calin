//-*-mode:swig;-*-

/*

   calin/math/healpix_array.i -- Stephen Fegan -- 2017-01-10

   SWIG interface file for calin.math.healpix_array

   Copyright 2017, Stephen Fegan <sfegan@llr.in2p3.fr>
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

%module (package="calin.math") healpix_array
%feature(autodoc,2);

%{
#include "math/healpix_array.hpp"
#define SWIG_FILE_WITH_INIT
  %}

%init %{
  import_array();
%}

%include "calin_typemaps.i"
%import "calin_global_definitions.i"

%apply double &OUTPUT { double& x, double& y, double& z,
  double& theta, double& phi };
%apply Eigen::Vector3d &OUTPUT { Eigen::Vector3d& vec };

%include "math/healpix_array.hpp"
