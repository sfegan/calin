/*

   calin/simulation/vcl_pe_processor.hpp -- Stephen Fegan -- 2025-10-08

   VCL class for processing PEs

   Copyright 2025, Stephen Fegan <sfegan@llr.in2p3.fr>
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

#pragma once

#include <algorithm>
#include <limits>

#include <util/vcl.hpp>
#include <math/ray_vcl.hpp>
#include <math/rng_vcl.hpp>
#include <math/geometry_vcl.hpp>
#include <simulation/pe_processor.hpp>

namespace calin { namespace simulation { namespace vcl_pe_processor {

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//
// VCLWaveformPEProcessor
//
// A PE processor that uses VCL types and operations to process PEs into a
// binned waveform, convolved with a unit response function, and with noise
//
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template<typename VCLArchitecture> class alignas(VCLArchitecture::vec_bytes) VCLWaveformPEProcessor:
  public calin::simulation::pe_processor::SimpleListPEProcessor
{
public:
  CALIN_TYPEALIAS(double_vt, typename VCLArchitecture::double_vt);
  CALIN_TYPEALIAS(double_at, typename VCLArchitecture::double_at);
  CALIN_TYPEALIAS(int64_vt, typename VCLArchitecture::int64_vt);
  CALIN_TYPEALIAS(int64_at, typename VCLArchitecture::int64_at);
  CALIN_TYPEALIAS(int64_bvt, typename VCLArchitecture::int64_bvt);
  CALIN_TYPEALIAS(uint64_vt, typename VCLArchitecture::uint64_vt);
  CALIN_TYPEALIAS(uint64_at, typename VCLArchitecture::uint64_at);
  CALIN_TYPEALIAS(uint64_bvt, typename VCLArchitecture::uint64_bvt);
  CALIN_TYPEALIAS(bool_vt, typename VCLArchitecture::double_bvt);

  CALIN_TYPEALIAS(VCLReal, calin::util::vcl::VCLDoubleReal<VCLArchitecture>);
  CALIN_TYPEALIAS(RNG, calin::math::rng::VCLRNG<VCLArchitecture>);

  VCLWaveformPEProcessor(unsigned nscope, unsigned npix, unsigned nsample, 
      double time_resolution_ns, unsigned nsample_advance,
      RNG* rng = nullptr, bool auto_clear = true, bool adopt_rng = false):
    calin::simulation::pe_processor::SimpleListPEProcessor(nscope, npix, auto_clear),
    nsample_(round_ndouble_to_vector(nsample)),  nadvance_(nsample_advance),
    time_resolution_ns_(time_resolution_ns), sampling_freq_ghz_(1.0/time_resolution_ns_),
  time_advance_(double(nadvance_)*time_resolution_ns_),
  pe_waveform_(nsample_, npix_), v_waveform_(nsample_, npix_),
    rng_(rng), adopt_rng_(adopt_rng)
  {
    if(rng_ == nullptr) {
      rng_ = new RNG(__PRETTY_FUNCTION__, "VCLWaveformPEProcessor RNG");
      adopt_rng_ = true;
    }
  }

  virtual ~VCLWaveformPEProcessor() 
  {
    if(adopt_rng_) {
      delete rng_;
    }
  }

  void transfer_scope_pes_to_waveform(unsigned iscope)
  {
    // Non-vectorized version. Takes advantage of "approximate" ordering of PEs
    // in time within each pixel to minimize memory accesses. Vectorized version
    // was more complex and was tested to be not much faster.
    validate_iscope_ipix(iscope, 0);
    double t0 = get_t0_for_scope(iscope);    
    for(unsigned ipix=0; ipix<npix_; ++ipix) {
      for(unsigned it=0; it<nsample_; ++it) {
        pe_waveform_(it, ipix) = 0;
      }
      auto pd = scopes_[iscope].pixel_data[ipix];
      if(pd==nullptr) {
        continue;
      }
      int it = int(floor((pd->t[0] - t0) * sampling_freq_ghz_));
      double wt = pd->w[0];
      for(unsigned ipe=1; ipe<pd->npe; ++ipe) {
        int jt = int(floor((pd->t[ipe] - t0) * sampling_freq_ghz_));
        if(it==jt) {
          wt += pd->w[ipe];
        } else {
          if(it>=0 and it<int(nsample_)) {
            pe_waveform_(it, ipix) += wt;
          }
          it = jt;
          wt = pd->w[ipe];
        }
      }
      if(it>=0 and it<int(nsample_)) {
        pe_waveform_(it, ipix) += wt;
      }
    }
  }

  void add_nsb_noise_to_waveform(const Eigen::VectorXd& nsb_freq_per_pixel_ghz,
    calin::simulation::detector_efficiency::SplinePEAmplitudeGenerator* pegen = nullptr)
  {
    double t_max = double(nsample_);
    for(unsigned ipix=0; ipix<npix_; ++ipix) {
      double rate_samples = sampling_freq_ghz_/nsb_freq_per_pixel_ghz[ipix];
      if(rate_samples<=0) {
        continue;
      }

      double t_samples = 0;
      while(t_samples < t_max) {
        double_vt dt_samples = rng_->exponential_double() * rate_samples;
        double_vt charge = 1.0;
        if(pegen) {
          charge = pegen->vcl_generate_amplitude<VCLArchitecture>(*rng_);
        }
        double_at dt_samples_a;
        dt_samples.store(dt_samples_a);
        double_at charge_a;
        charge.store(charge_a);
        for(unsigned insb=0; insb<VCLArchitecture::num_double; ++insb) {
          t_samples += dt_samples_a[insb];
          if(t_samples >= double(nsample_)) {
            goto next_pixel;
          }
          int it = int(floor(t_samples));
          pe_waveform_(it, ipix) += charge_a[insb];
        }
      }
      next_pixel:
      ;
    }
  }

  void convolve_impulse_response(const Eigen::VectorXd& impulse_response)
  {
    for(unsigned ipix=0; ipix<npix_; ++ipix) {
      for(unsigned it=0; it<nsample_; ++it) {
        v_waveform_(it, ipix) = 0;
      }
      for(unsigned it=0; it<nsample_; ++it) {
        double wt = pe_waveform_(it, ipix);
        if(wt==0) {
          continue;
        }
        unsigned jmax = std::min(unsigned(impulse_response.size()), nsample_ - it);
        for(unsigned jt=0; jt<jmax; ++jt) {
          unsigned kt = it + jt;
          v_waveform_(kt, ipix) += wt * impulse_response[jt];
        }
      }
    }
  }

  void clear_waveforms()
  {
    pe_waveform_.setZero();
    v_waveform_.setZero();
  }
  
  const Eigen::MatrixXd& pe_waveform() const { return pe_waveform_; }
  const Eigen::MatrixXd& v_waveform() const { return v_waveform_; }

private:
  static unsigned round_ndouble_to_vector(unsigned n) {
    return n + std::min(n%VCLArchitecture::num_double, 1U);
  } 

  double get_t0_for_scope(unsigned iscope) const {
    return scopes_[iscope].tmin - time_advance_;
  }

  unsigned nsample_;
  unsigned nadvance_;
  double time_resolution_ns_;
  double sampling_freq_ghz_;
  double time_advance_;
  Eigen::MatrixXd pe_waveform_; // shape (nsample_, npix_): access as (it, ipix)
  Eigen::MatrixXd v_waveform_;  // shape (nsample_, npix_): access as (it, ipix)
  RNG* rng_ = nullptr;
  bool adopt_rng_ = false;
};  

} } } // namespace calin::simulations::vcl_pe_processor
