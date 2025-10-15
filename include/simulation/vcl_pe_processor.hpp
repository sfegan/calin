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
#include <fftw3.h>

#include <util/log.hpp>
#include <util/vcl.hpp>
#include <math/ray_vcl.hpp>
#include <math/rng_vcl.hpp>
#include <math/geometry_vcl.hpp>
#include <math/fftw_util.hpp>
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
    pe_waveform_(nsample_, npix_), pe_transform_(nsample_, npix_), 
    v_transform_(nsample_, npix_), v_waveform_(nsample_, npix_), 
    rng_(rng), adopt_rng_(adopt_rng)
  {
    if(rng_ == nullptr) {
      rng_ = new RNG(__PRETTY_FUNCTION__, "VCLWaveformPEProcessor RNG");
      adopt_rng_ = true;
    }

    int rank = 1;
    int n[] = { int(nsample_) };
    int howmany = npix_;
    int* inembed = n;
    int istride = 1;
    int idist = nsample_;
    int* onembed = n;
    int ostride = 1;
    int odist = nsample_;
    
    double* in = pe_waveform_.data();
    double* out = pe_transform_.data();
    fftw_r2r_kind kind[] = { FFTW_R2HC };

    fftw_plan_pe_fwd_ = fftw_plan_many_r2r(rank, n, howmany,
      in, inembed, istride, idist, out, onembed, ostride, odist, kind, 0);

    in = v_transform_.data();
    out = v_waveform_.data();
    kind[0] = FFTW_HC2R;

    fftw_plan_v_bwd_ = fftw_plan_many_r2r(rank, n, howmany,
      in, inembed, istride, idist, out, onembed, ostride, odist, kind, 0);
  }

  virtual ~VCLWaveformPEProcessor() 
  {
    fftw_destroy_plan(fftw_plan_pe_fwd_);
    fftw_destroy_plan(fftw_plan_v_bwd_);
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
    pe_transform_valid_ = false;
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
    pe_transform_valid_ = false;
  }

  unsigned register_impulse_response(const Eigen::VectorXd& impulse_response)
  {
    unsigned response_size = impulse_response.size();    
    Eigen::VectorXd response(nsample_);
    response.setZero();
    if(response_size>nsample_) {
      calin::util::log::LOG(calin::util::log::WARNING) 
        << "Truncating impulse response of size "
        << impulse_response.size() << " to " << nsample_;
      response_size = nsample_;
      response = impulse_response.head(nsample_);
    } else {
      response.head(response_size) = impulse_response;
    }
    
    Eigen::VectorXd transform(nsample_);
    fftw_plan plan = fftw_plan_r2r_1d(nsample_, &response[0], &transform[0],
        FFTW_R2HC, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    impulse_responses_.emplace_back(response_size, response, transform);
    return impulse_responses_.size()-1;
  }

  void convolve_impulse_response(unsigned impulse_response_id, 
    const Eigen::VectorXd& pedestal = Eigen::VectorXd())
  {
    if(impulse_response_id >= impulse_responses_.size()) {
      calin::util::log::LOG(calin::util::log::ERROR)
        << "Invalid impulse_response_id " << impulse_response_id;
      return;
    }
    const ImpulseResponse& ir = impulse_responses_[impulse_response_id];
  
    for(unsigned ipix=0; ipix<npix_; ++ipix) {
      double ped = (ipix<pedestal.size()) ? pedestal[ipix] : 0.0;
      for(unsigned it=0; it<nsample_; ++it) {
        v_waveform_(it, ipix) = ped;
      }
      for(unsigned it=0; it<nsample_; ++it) {
        double wt = pe_waveform_(it, ipix);
        if(wt==0) {
          continue;
        }
        unsigned jmax = std::min(unsigned(ir.response_size), nsample_ - it);
        double *__restrict__ v_waveform_ptr = &v_waveform_(it, ipix);
        const double *__restrict__ impulse_response_ptr = &ir.response[0];
        for(unsigned jt=0; jt<jmax; ++jt) {
          *(v_waveform_ptr++) += wt * (*(impulse_response_ptr++));
        }
      }
    }
  }

  void convolve_impulse_response_fft(unsigned impulse_response_id,
    const Eigen::VectorXd& pedestal = Eigen::VectorXd())
  {
    if(impulse_response_id >= impulse_responses_.size()) {
      calin::util::log::LOG(calin::util::log::ERROR)
        << "Invalid impulse_response_id " << impulse_response_id;
      return;
    }
    const ImpulseResponse& ir = impulse_responses_[impulse_response_id];

    if(not pe_transform_valid_) {
      fftw_execute(fftw_plan_pe_fwd_);
      pe_transform_valid_ = true;
    }

    for(unsigned ipix=0; ipix<npix_; ++ipix) {
      calin::math::fftw_util::hcvec_scale_and_multiply(
        &v_transform_(0, ipix),
        &pe_transform_(0, ipix), ir.transform.data(),
        nsample_, 1.0/nsample_);      
    }

    if(pedestal.size()>0) {
      v_transform_.row(0).head(pedestal.size()) += pedestal;
    }

    fftw_execute(fftw_plan_v_bwd_);
  }

  void clear_waveforms()
  {
    pe_waveform_.setZero();
    v_waveform_.setZero();
    pe_transform_valid_ = false;
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

  struct ImpulseResponse {
    ImpulseResponse(unsigned response_size_,
        const Eigen::VectorXd& response_,
        const Eigen::VectorXd& transform_):
      response_size(response_size_), response(response_), transform(transform_) {}
    unsigned response_size;
    Eigen::VectorXd response;
    Eigen::VectorXd transform;
  };

  unsigned nsample_;
  unsigned nadvance_;
  double time_resolution_ns_;
  double sampling_freq_ghz_;
  double time_advance_;
  bool pe_transform_valid_ = false;
  fftw_plan fftw_plan_pe_fwd_;
  fftw_plan fftw_plan_v_bwd_;
  Eigen::MatrixXd pe_waveform_;  // shape (nsample_, npix_): access as (it, ipix)
  Eigen::MatrixXd pe_transform_; // shape (nsample_, npix_): access as (it, ipix)
  Eigen::MatrixXd v_transform_;  // shape (nsample_, npix_): access as (it, ipix)
  Eigen::MatrixXd v_waveform_;   // shape (nsample_, npix_): access as (it, ipix)
  std::vector<ImpulseResponse> impulse_responses_;
  RNG* rng_ = nullptr;
  bool adopt_rng_ = false;
};

} } } // namespace calin::simulations::vcl_pe_processor
