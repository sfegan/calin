/*

   calin/simulation/corsika8_shower_generator.cpp -- Stephen Fegan -- 2024-08-23

   Class to generate extensive air showers using Corsika 8

   Copyright 2024, Stephen Fegan <sfegan@llr.in2p3.fr>
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

#include <Eigen/Dense>

#include <calin_global_config.hpp>
#include <math/rng.hpp>
#include <simulation/corsika8_shower_generator.hpp>
#include <provenance/chronicle.hpp>
#include <calin_global_definitions.hpp>

using namespace calin::simulation::corsika8_shower_generator;
using namespace calin::util::log;

CORSIKA8ShowerGenerator::~CORSIKA8ShowerGenerator()
{
  // nothing to see here
}

void CORSIKA8ShowerGenerator::
generate_showers(calin::simulation::tracker::TrackVisitor* visitor,
                unsigned num_events,
                calin::simulation::tracker::ParticleType type,
                double total_energy,
                const Eigen::Vector3d& x0,
                const Eigen::Vector3d& u0,
                double weight)
{
  // don't do anything
}

CORSIKA8ShowerGenerator::config_type CORSIKA8ShowerGenerator::default_config()
{
  using namespace calin::ix::simulation::corsika8_shower_generator;
  
  config_type config;
  config.set_atmospheric_model(ATM_LinsleyUSStd);
  config.set_atmospheric_fraction_n2(0.78084);
  config.set_atmospheric_fraction_o2(0.20946);
  config.set_atmospheric_fraction_ar(0.009340);
  config.set_atmospheric_fraction_co2(0.000417);
  config.set_num_atm_layers(1000);

  config.set_earth_radius(6371.0 * 1E5);
  config.set_zground(0.0);
  config.set_ztop(112.8 * 1E5);
  // calin::vec_to_xyz(config.mutable_uniform_magnetic_field(), Eigen::Vector3d{10.0,12.0,13.0});

  config.set_seed(0);
  config.set_verbosity(C8_WARNING);

  config.set_electrion_photon_cut(0.5);
  config.set_hadronic_cut(300.0);
  config.set_muon_cut(300.0);
  config.set_tau_cut(300.0);
  config.set_he_hadronic_model(SIBYLL);
  config.set_he_hadronic_transition_energy(79400.0 /* 10^1.9 GeV = 79.4 GeV */);

  return config;
}

// #ifndef CALIN_HAVE_CORSIKA8
CORSIKA8ShowerGenerator* CORSIKA8ShowerGenerator::new_instance(const config_type& config)
{
  throw std::runtime_error("CORSIKA8ShowerGenerator::new_instance not implemented");
  return nullptr;
}
// #endif