/*

   calin/simulation/pe_processor.cpp -- Stephen Fegan -- 2017-01-16

   Multi-purpose PE (weight, scope, pixel & time) processor.

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

#include <stdexcept>
#include <algorithm>
#include <simulation/pe_processor.hpp>

using namespace calin::simulation::pe_processor;

#include <util/log.hpp>
using namespace calin::util::log;

PEProcessor::~PEProcessor()
{
  // nothing to see here
}

void PEProcessor::start_processing()
{
  // nothing to see here
}

void PEProcessor::process_focal_plane_hit(unsigned scope_id, int pixel_id,
    double x, double y, double t0, double pe_weight)
{
  // nothing to see here
}

void PEProcessor::finish_processing()
{
  // nothing to see here
}

SimpleImagePEProcessor::
SimpleImagePEProcessor(unsigned nscope, unsigned npix, bool auto_clear):
  PEProcessor(), auto_clear_(auto_clear),
  images_(nscope, std::vector<Accumulator>(npix))
{
  // nothing to see here
}

SimpleImagePEProcessor::
SimpleImagePEProcessor(const std::vector<unsigned> npix, bool auto_clear):
  PEProcessor(), auto_clear_(auto_clear), images_()
{
  for(auto inpix : npix)images_.emplace_back(inpix);
}

SimpleImagePEProcessor::~SimpleImagePEProcessor()
{
  // nothing to see here
}

void SimpleImagePEProcessor::start_processing()
{
  if(auto_clear_)clear_all_images();
}

void SimpleImagePEProcessor::
process_focal_plane_hit(unsigned scope_id, int pixel_id, double x, double y,
  double t0, double pe_weight)
{
#if 0
  static unsigned counter = 0;
  if(counter++ < 10)
    LOG(INFO) << scope_id << ' ' << pixel_id << ' ' << x << ' ' << y << ' '
              << t0 << ' ' << pe_weight;
#endif

  if(pixel_id<0)return;
  if(scope_id >= images_.size())
    throw std::out_of_range("SimpleImagePEProcessor::process_focal_plane_hit: scope_id out "
      "of range");
  if(unsigned(pixel_id) >= images_[scope_id].size())
    throw std::out_of_range("SimpleImagePEProcessor::process_focal_plane_hit: pixel_id out "
      "of range");
  images_[scope_id][pixel_id].accumulate(pe_weight);
}

const std::vector<double>
SimpleImagePEProcessor::scope_image(unsigned iscope) const
{
  if(iscope >= images_.size())
    throw std::out_of_range("SimpleImagePEProcessor::scope_image: iscope out "
      "of range");
  std::vector<double> image(images_[iscope].size());
  std::transform(images_[iscope].begin(), images_[iscope].end(), image.begin(),
    [](const Accumulator& acc){ return acc.total(); });
  return image;
}

void SimpleImagePEProcessor::clear_all_images()
{
  for(auto& image : images_)
    std::for_each(image.begin(), image.end(),
      [](Accumulator& acc){ acc.reset(); });
}

TelescopePSFCalcPEProcessor::
TelescopePSFCalcPEProcessor(unsigned iscope, bool auto_clear): PEProcessor(),
  auto_clear_(auto_clear), iscope_(iscope), mom_()
{
  // nothing to see here
}

TelescopePSFCalcPEProcessor::~TelescopePSFCalcPEProcessor()
{
  // nothing to see here
}

void TelescopePSFCalcPEProcessor::start_processing()
{
  if(auto_clear_)clear();
}

void TelescopePSFCalcPEProcessor::process_focal_plane_hit(unsigned scope_id, int pixel_id,
  double x, double y, double t0, double pe_weight)
{
  if(scope_id == iscope_)mom_.accumulate(x, y, pe_weight);
}

TelescopePSFCalcThirdMomentPEProcessor::
TelescopePSFCalcThirdMomentPEProcessor(unsigned iscope, bool auto_clear):
  PEProcessor(), auto_clear_(auto_clear), iscope_(iscope), mom_()
{
  // nothing to see here
}

TelescopePSFCalcThirdMomentPEProcessor::
~TelescopePSFCalcThirdMomentPEProcessor()
{
  // nothing to see here
}

void TelescopePSFCalcThirdMomentPEProcessor::start_processing()
{
  if(auto_clear_)clear();
}

void TelescopePSFCalcThirdMomentPEProcessor::
process_focal_plane_hit(unsigned scope_id, int pixel_id,
  double x, double y, double t0, double pe_weight)
{
  if(scope_id == iscope_)mom_.accumulate(x, y, pe_weight);
}
