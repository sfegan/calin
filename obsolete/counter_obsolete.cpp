/*

   calin/diagnostics/counter.cpp -- Stephen Fegan -- 2016-03-04

   Counter diagnostics visitor

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

#include <util/log.hpp>
#include <diagnostics/counter.hpp>

using namespace calin::util::log;
using namespace calin::diagnostics::counter;

CounterDeltaTDiagnostics::CounterDeltaTDiagnostics(): TelescopeEventVisitor()
{
  // nothing to see here
}

CounterDeltaTDiagnostics::~CounterDeltaTDiagnostics()
{
  // nothing to see here
}

bool CounterDeltaTDiagnostics::demand_waveforms()
{
  return false;
}

bool CounterDeltaTDiagnostics::visit_telescope_run(
  const calin::ix::iact_data::telescope_run_configuration::
    TelescopeRunConfiguration* run_config)
{
  return true;
}

bool CounterDeltaTDiagnostics::visit_telescope_event(uint64_t seq_index,
  calin::ix::iact_data::telescope_event::TelescopeEvent* event)
{
  return true;
}
