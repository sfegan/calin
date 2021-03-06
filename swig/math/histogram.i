/*

   calin/math/histogram.i -- Stephen Fegan -- 2015-04-23

   SWIG interface file for calin.math.histogram

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

%module (package="calin.math") histogram
%feature(autodoc,2);

%{
#include "math/histogram.hpp"
#define SWIG_FILE_WITH_INIT
  %}

%init %{
  import_array();
%}

%include "calin_typemaps.i"
%import "calin_global_definitions.i"

%import "math/histogram.pb.i"

%newobject calin::math::histogram::BasicHistogram1D::dump_as_proto() const;
%newobject calin::math::histogram::BasicHistogram1D::create_from_proto(
  calin::ix::math::histogram::Histogram1DData& data);
%newobject calin::math::histogram::BasicHistogram1D::serialize() const;
%newobject calin::math::histogram::BasicHistogram1D::dump_as_compactified_proto(
  int max_dense_bins_in_output, int max_output_rebinning) const;

%newobject calin::math::histogram::new_histogram(const calin::ix::math::histogram::AccumulatedAndSerializedHistogram1DConfig& config);
%newobject calin::math::histogram::compactify(const calin::ix::math::histogram::Histogram1DData& original_hist,
  int max_dense_bins_in_output, int max_output_rebinning);
%newobject calin::math::histogram::rebin(const calin::ix::math::histogram::Histogram1DData& original_hist, unsigned rebinning_factor);
%newobject calin::math::histogram::sparsify(const calin::ix::math::histogram::Histogram1DData& original_hist);
%newobject calin::math::histogram::densify(const calin::ix::math::histogram::Histogram1DData& original_hist);

%include "math/histogram.hpp"
