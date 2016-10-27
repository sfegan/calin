/*

   calin/simulation/vso_quadrature_iact_array_integration.hpp
                                              -- Stephen Fegan -- 2016-07-27

   IACT array tracker that does quadrature integration over the cherenkov
   cone - instance that uses VSOptics ray tracer

   Copyright 2016, Stephen Fegan <sfegan@llr.in2p3.fr>
   LLR, Ecole polytechnique, CNRS/IN2P3, Universite Paris-Saclay

   This file is part of "calin"

   "calin" is free software: you can redistribute it and/or modify it
   under the terms of the GNU General Public License version 2 or
   later, as published by the Free Software Foundation.

   "calin" is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.

*/

#pragma once

#include <math/rng.hpp>
#include <simulation/vso_array.hpp>
#include <simulation/vso_raytracer.hpp>
#include <simulation/quadrature_iact_array_integration.hpp>

namespace calin { namespace simulation { namespace quadrature_iact_array_integration {

class VSO_QuadratureIACTArrayIntegrationHitVisitor;

class VSO_IACTDetectorSphereHitProcessor:
  public calin::simulation::iact_array_tracker::IACTDetectorSphereHitProcessor
{
public:
  VSO_IACTDetectorSphereHitProcessor(
    VSO_QuadratureIACTArrayIntegrationHitVisitor* quadrature,
    calin::simulation::vs_optics::VSOTelescope* scope);
  virtual ~VSO_IACTDetectorSphereHitProcessor();
  virtual void process_hit(
    const calin::simulation::iact_array_tracker::IACTDetectorSphereHit& hit);
private:
  VSO_QuadratureIACTArrayIntegrationHitVisitor* quadrature_ = nullptr;
  calin::simulation::vs_optics::VSOTelescope* scope_ = nullptr;
};

class VSO_QuadratureIACTArrayIntegrationHitVisitor:
  public calin::simulation::iact_array_tracker::HitIACTVisitor
{
public:
  VSO_QuadratureIACTArrayIntegrationHitVisitor(
    const calin::ix::simulation::tracker::QuadratureIACTArrayIntegrationConfig& config,
    calin::simulation::vs_optics::VSOArray* array,
    QuadratureIACTArrayPEProcessor* visitor,
    calin::math::rng::RNG* rng = nullptr,
    bool adopt_array = false, bool adopt_visitor = false,
    bool adopt_rng = false);
  ~VSO_QuadratureIACTArrayIntegrationHitVisitor();
  std::vector<calin::simulation::iact_array_tracker::IACTDetectorSphere> spheres() override;
  void visit_event(const calin::simulation::tracker::Event& event,
    bool& kill_event) override;
  void visit_cherenkov_track(
    const calin::simulation::air_cherenkov_tracker::AirCherenkovTrack& cherenkov_track,
    bool& kill_track) override;
  void leave_cherenkov_track() override;
  void leave_event() override;

  unsigned num_hit(unsigned iscope) const;
  unsigned num_miss(unsigned iscope) const;
  const std::vector<unsigned> raytracer_status(unsigned iscope) const;
protected:
  friend class VSO_IACTDetectorSphereHitProcessor;

  void process_test_ray(
    const calin::simulation::iact_array_tracker::IACTDetectorSphereHit& hit,
    calin::simulation::vs_optics::VSOTelescope* scope,
    double cos_phi, double sin_phi, double dphi);
  void process_hit(
    const calin::simulation::iact_array_tracker::IACTDetectorSphereHit& hit,
    calin::simulation::vs_optics::VSOTelescope* scope);

  double ray_spacing_linear_;
  double ray_spacing_angular_;
  calin::simulation::vs_optics::VSOArray* array_ = nullptr;
  bool adopt_array_ = false;
  QuadratureIACTArrayPEProcessor* visitor_ = nullptr;
  bool adopt_visitor_ = false;
  calin::math::rng::RNG* rng_ = nullptr;
  bool adopt_rng_ = false;
  calin::simulation::vs_optics::VSORayTracer* ray_tracer_ = nullptr;
  std::vector<unsigned> num_hit_;
  std::vector<unsigned> num_miss_;
  std::vector<std::vector<unsigned>> ray_tracer_status_;
};

} } } // namespace calin::simulation::quadrature_iact_array_integration