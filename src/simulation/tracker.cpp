/*

   calin/simulation/tracker.cpp -- Stephen Fegan -- 2015-07-17

   Base class for all air shower track visitors

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

#include <sys/time.h>
#include <iostream>
#include <cassert>
#include <simulation/tracker.hpp>
#include <util/log.hpp>

using namespace calin::simulation::tracker;
using namespace calin::util::log;

ShowerGenerator::~ShowerGenerator()
{
  // nothing to see here
}

std::ostream& calin::simulation::tracker::operator<<(std::ostream& stream,
  const calin::simulation::tracker::Track& t)
{
  stream
    << "================================== TRACK ==================================\n"
    << "type: " << particle_type_to_string(t.type) << " (PDG: " << t.pdg_type << ")  q: " << t.q << "  mass: " << t.mass << "  weight: " << t.weight << '\n'
    << "x0: " << t.x0.transpose() << '\n'
    << "x1: " << t.x1.transpose() << '\n'
    << "dx: " << t.dx << "  dx_hat: " << t.dx_hat.transpose() << '\n'
    << "u0: " << t.u0.transpose() << '\n'
    << "u1: " << t.u1.transpose() << '\n'
    << "e0: " << t.e0 << "  e1: " << t.e1 << "  de: " << t.de << '\n'
    << "t0: " << t.t0 << "  t1: " << t.t1 << "  dt: " << t.dt << '\n'
    << "===========================================================================\n";
  return stream;
}

calin::simulation::tracker::ParticleType
calin::simulation::tracker::pdg_type_to_particle_type(int pdg_type)
{
  switch(pdg_type)
  {
    case 22:         return ParticleType::GAMMA;
    case 11:         return ParticleType::ELECTRON;
    case -11:        return ParticleType::POSITRON;
    case 13:         return ParticleType::MUON;
    case -13:        return ParticleType::ANTI_MUON;
    case 2212:       return ParticleType::PROTON;
    case -2212:      return ParticleType::ANTI_PROTON;
    case 2112:       return ParticleType::NEUTRON;
    case 1000020040: return ParticleType::HELIUM;
    case 1000060120: return ParticleType::CARBON;
    case 1000080160: return ParticleType::OXYGEN;
    case 1000120240: return ParticleType::MAGNESIUM;
    case 1000140280: return ParticleType::SILICON;
    case 1000260560: return ParticleType::IRON;
    default:         return ParticleType::OTHER;
  };
  assert(0);
  return ParticleType::OTHER;
}

int calin::simulation::tracker::
particle_type_to_pdg_type(calin::simulation::tracker::ParticleType track_type)
{
  switch(track_type)
  {
    case ParticleType::GAMMA:       return 22;
    case ParticleType::ELECTRON:    return 11;
    case ParticleType::POSITRON:    return -11;
    case ParticleType::MUON:        return 13;
    case ParticleType::ANTI_MUON:   return -13;
    case ParticleType::PROTON:      return 2212;
    case ParticleType::ANTI_PROTON: return -2212;
    case ParticleType::NEUTRON:     return 2112;
    case ParticleType::HELIUM:      return 1000020040;
    case ParticleType::CARBON:      return 1000060120;
    case ParticleType::OXYGEN:      return 1000080160;
    case ParticleType::MAGNESIUM:   return 1000120240;
    case ParticleType::SILICON:     return 1000140280;
    case ParticleType::IRON:        return 1000260560;
    case ParticleType::OTHER:
      throw(std::runtime_error("ParticleType::OTHER has no PDG type code"));
  };
  assert(0);
  return 0;
}

// Masses extracted from G4 in Python with:
// for pdg in [ 22, 11, -11, 13, -13, 2212, -2212, 2112, 1000020040, 1000060120, 1000080160, 1000120240, 1000140280, 1000260560]:
//     print(pdg, generator.pdg_type_to_string(pdg), generator.pdg_type_to_mass(pdg))

double calin::simulation::tracker::
particle_type_to_mass(ParticleType track_type)
{
  switch(track_type)
  {
    case ParticleType::GAMMA:
      return 0.0;
    case ParticleType::ELECTRON:
    case ParticleType::POSITRON:
      return 0.51099891;
    case ParticleType::MUON:
    case ParticleType::ANTI_MUON:
      return 105.6583715;
    case ParticleType::PROTON:
    case ParticleType::ANTI_PROTON:
      return 938.272013;
    case ParticleType::NEUTRON:
      return 939.56536;
    case ParticleType::HELIUM:
      return 3727.379;
    case ParticleType::CARBON:
      return 11174.86338798439;
    case ParticleType::OXYGEN:
      return 14895.081534649964;
    case ParticleType::MAGNESIUM:
      return 22335.796596658216;
    case ParticleType::SILICON:
      return 26053.193927338805;
    case ParticleType::IRON:
      return 52089.808009454995;
    case ParticleType::OTHER:
      throw(std::runtime_error("ParticleType::OTHER has no mass"));
  };
  assert(0);
  return 0;
}

double calin::simulation::tracker::
particle_type_to_charge(ParticleType track_type)
{
  switch(track_type)
  {
  case ParticleType::GAMMA:
  case ParticleType::NEUTRON:
    return 0.0;
  case ParticleType::ELECTRON:
  case ParticleType::MUON:
  case ParticleType::ANTI_PROTON:
    return -1.0;
  case ParticleType::POSITRON:
  case ParticleType::ANTI_MUON:
  case ParticleType::PROTON:
    return 1.0;
  case ParticleType::HELIUM:
    return 2.0;
  case ParticleType::CARBON:
    return 6.0;
  case ParticleType::OXYGEN:
    return 8.0;
  case ParticleType::MAGNESIUM:
    return 12.0;
  case ParticleType::SILICON:
    return 14.0;
  case ParticleType::IRON:
    return 26.0;
  case ParticleType::OTHER:
    throw(std::runtime_error("ParticleType::OTHER has no charge"));
  };
  assert(0);
  return 0;
}

std::string calin::simulation::tracker::
particle_type_to_string(ParticleType track_type)
{
  switch(track_type)
  {
  case ParticleType::GAMMA:
    return "GAMMA";
  case ParticleType::ELECTRON:
    return "ELECTRON";
  case ParticleType::POSITRON:
    return "POSITRON";
  case ParticleType::MUON:
    return "MUON-";
  case ParticleType::ANTI_MUON:
    return "MUON+";
  case ParticleType::ANTI_PROTON:
    return "PROTON-";
  case ParticleType::PROTON:
    return "PROTON+";
  case ParticleType::NEUTRON:
    return "NEUTRON";
  case ParticleType::HELIUM:
    return "HELIUM";
  case ParticleType::CARBON:
    return "CARBON";
  case ParticleType::OXYGEN:
    return "OXYGEN";
  case ParticleType::MAGNESIUM:
    return "MAGNESIUM";
  case ParticleType::SILICON:
    return "SILICON";
  case ParticleType::IRON:
    return "IRON";
  case ParticleType::OTHER:
    return "OTHER";
  };
  assert(0);
  return "UNKNOWN";
}

TrackVisitor::~TrackVisitor()
{
  // nothing to see here
}

void TrackVisitor::visit_event(const Event& event, bool& kill_event)
{
  // default is to do nothing
}

void TrackVisitor::visit_track(const Track& track, bool& kill_track)
{
  // default is to do nothing
}

void TrackVisitor::leave_event()
{
  // default is to do nothing
}

MultiDelegatingTrackVisitor::~MultiDelegatingTrackVisitor()
{
  for(auto* ivisitor : adopted_visitors_)delete ivisitor;
}

void MultiDelegatingTrackVisitor::visit_event(const Event& event, bool& kill_event)
{
  for(auto* ivisitor : visitors_)
  {
    bool ikill_event = false;
    ivisitor->visit_event(event, ikill_event);
    kill_event |= ikill_event;
  }
}

void MultiDelegatingTrackVisitor::visit_track(const Track& track, bool& kill_track)
{
  for(auto* ivisitor : visitors_)
  {
    bool ikill_track = false;
    ivisitor->visit_track(track, ikill_track);
    kill_track |= ikill_track;
  }
}

void MultiDelegatingTrackVisitor::leave_event()
{
  for(auto* ivisitor : visitors_)ivisitor->leave_event();
}

void MultiDelegatingTrackVisitor::add_visitor(TrackVisitor* visitor, bool adopt_visitor)
{
  visitors_.emplace_back(visitor);
  if(adopt_visitor)adopted_visitors_.emplace_back(visitor);
}

void MultiDelegatingTrackVisitor::
add_visitor_at_front(TrackVisitor* visitor, bool adopt_visitor)
{
  visitors_.insert(visitors_.begin(), visitor);
  if(adopt_visitor)adopted_visitors_.insert(adopted_visitors_.begin(), visitor);
}
