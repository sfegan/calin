/*

   calin/math/fftw_codelet_256f.hpp -- Stephen Fegan -- 2025-10-19

   Container for FFTW codelets for AVX/AVX2 float

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

#include <stdexcept>

#include <util/vcl.hpp>
#include <math/fftw_util.hpp>

#include "fftw_codelet_container.hpp"

template class FFTWCodelet_Container<calin::util::vcl::VCL256FloatReal>;
template class calin::math::fftw_util::FFTWCodelet<calin::util::vcl::VCL256FloatReal>;
