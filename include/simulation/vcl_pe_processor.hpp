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

  VCLWaveformPEProcessor(unsigned nscope, unsigned npix, unsigned nsample, double time_resolution, unsigned nsample_advance,
      RNG* rng = nullptr, bool auto_clear = true, bool adopt_rng = false):
    calin::simulation::pe_processor::SimpleListPEProcessor(nscope, npix, auto_clear),
    nsample_(round_ndouble_to_vector(nsample)),  nadvance_(nsample_advance),
    time_resolution_(time_resolution), sampling_freq_(1.0/time_resolution),
    time_advance_(double(nadvance_)*time_resolution_),
    pe_waveform_(npix, nsample_),
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

  void transfer_scope_pes_to_waveform_slow(unsigned iscope)
  {
    // Non-vectorized version. Takes advantage of "approximate" ordering of PEs
    // in time within each pixel to minimize memory accesses
    validate_iscope_ipix(iscope, 0);
    pe_waveform_.setZero();
    double t0 = get_t0_for_scope(iscope);    
    for(unsigned ipix=0; ipix<npix_; ++ipix) {
      auto pd = scopes_[iscope].pixel_data[ipix];
      if(pd==nullptr) {
        continue;
      }
      int it = int(floor((pd->t[0] - t0) * sampling_freq_));
      double wt = pd->w[0];
      for(unsigned ipe=1; ipe<pd->npe; ++ipe) {
        int jt = int(floor((pd->t[ipe] - t0) * sampling_freq_));
        if(it==jt) {
          wt += pd->w[ipe];
        } else {
          if(it>=0 and it<int(nsample_)) {
            pe_waveform_(ipix, it) += wt;
          }
          it = jt;
          wt = pd->w[ipe];
        }
      }
      if(it>=0 and it<int(nsample_)) {
        pe_waveform_(ipix, it) += wt;
      }
    }
  }

  void transfer_scope_pes_to_waveform(unsigned iscope)
  {
    // Vectorized version. Takes advantage of "approximate" ordering of PEs
    // in time within each pixel to minimize memory accesses
    validate_iscope_ipix(iscope, 0);
    pe_waveform_.setZero();
    double t0 = get_t0_for_scope(iscope);
    for(unsigned ipix=0; ipix<npix_; ++ipix) {
      auto pd = scopes_[iscope].pixel_data[ipix];
      if(pd==nullptr) {
        continue;
      }
      int32_t it = 0;
      double w = 0;
      unsigned ipe = 0;
      unsigned jpe = VCLArchitecture::num_int64;
      for(; jpe<=pd->npe; ipe=jpe,jpe+=VCLArchitecture::num_int64) {
        double_vt t;
        t.load(pd->t + ipe);
        int64_vt jt = truncate_to_int64_limited((t - t0) * sampling_freq_);
        if(horizontal_and(jt == jt[0])) {
          // Fast path : all indices identical
          if(jt[0]<0 or jt[0]>=int64_t(nsample_)) {
            continue;
          }
          double_vt jw;
          jw.load(pd->w + ipe);
          if(jt[0] != it) {
            pe_waveform_(ipix, it) += w;
            it = jt[0];
            w = vcl::horizontal_add(jw);
          } else {
            w += vcl::horizontal_add(jw);
          }
        } else {
          // Slow path : indices not identical
          double_at jta;
          jt.store(jta);
          for(unsigned koffset=0; koffset<VCLArchitecture::num_int64; ++ipe,++koffset) {
            if(jta[koffset]<0 or jta[koffset]>=int64_t(nsample_)) {
              continue;
            }
            if(jta[koffset] == it) {
              w += pd->w[ipe];
            } else {
              pe_waveform_(ipix, it) += w;
              it = jta[koffset];
              w = pd->w[ipe];
            }
          }
        }
      }
      double_vt t;
      t.load(pd->t + ipe);
      int64_vt jt = truncate_to_int64_limited((t - t0) * sampling_freq_);
      double_at jta;
      jt.store(jta);

      for(unsigned koffset=0; ipe<pd->npe; ++ipe,++koffset) {        
        if(jta[koffset]<0 or jta[koffset]>=int64_t(nsample_)) {
          continue;
        }
        if(jta[koffset] == it) {
          w += pd->w[ipe];
        } else {
          pe_waveform_(ipix, it) += w;
          it = jta[koffset];
          w = pd->w[ipe];
        }
      }
      // Finalize last index
      pe_waveform_(ipix, it) += w;
    }
  }

  // void add_nsb_noise_to_waveform(const Eigen::VectorXd& rate_per_pixel_ghz,
  //   calin::simulation::detector_efficiency::PEAmplitudeGenerator* pegen = nullptr);
  // void downsample_waveform(unsigned ipix, const Eigen::VectorXd& impulse_response, double offset);

  const Eigen::MatrixXd& pe_waveform() const { return pe_waveform_; }

private:
  static unsigned round_ndouble_to_vector(unsigned n) {
    return n + std::min(n%VCLArchitecture::num_double, 1U);
  } 

  double get_t0_for_scope(unsigned iscope) const {
    return scopes_[iscope].tmin - time_advance_;
  }

  unsigned nsample_;
  unsigned nadvance_;
  double time_resolution_;
  double sampling_freq_;
  double time_advance_;
  Eigen::MatrixXd pe_waveform_;
  RNG* rng_ = nullptr;
  bool adopt_rng_ = false;
};  

} } } // namespace calin::simulations::vcl_pe_processor
