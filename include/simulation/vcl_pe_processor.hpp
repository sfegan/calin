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
#include <util/memory.hpp>
#include <util/string.hpp>
#include <math/special.hpp>
#include <math/ray_vcl.hpp>
#include <math/rng_vcl.hpp>
#include <math/geometry_vcl.hpp>
#include <math/fftw_util.hpp>
#include <math/interpolation_1d.hpp>
#include <math/brent.hpp>
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

template<typename VCLReal> class alignas(VCLReal::vec_bytes) VCLWaveformPEProcessor:
  public calin::simulation::pe_processor::SimpleListPEProcessor
{
public:
  CALIN_TYPEALIAS(real_t,    typename VCLReal::real_t);
  CALIN_TYPEALIAS(real_vt,   typename VCLReal::real_vt);
  CALIN_TYPEALIAS(real_at,   typename VCLReal::real_at);
  CALIN_TYPEALIAS(int_vt,    typename VCLReal::int_vt);
  CALIN_TYPEALIAS(vecX_t,    typename VCLReal::vecX_t);
  CALIN_TYPEALIAS(matX_t,    typename VCLReal::matX_t);

  CALIN_TYPEALIAS(VCLArchitecture, typename VCLReal::architecture);

  CALIN_TYPEALIAS(RNG, calin::math::rng::VCLRNG<VCLArchitecture>);

  VCLWaveformPEProcessor(unsigned nscope, unsigned npix, unsigned nsample, 
      double time_resolution_ns, unsigned nsample_advance,
      RNG* rng = nullptr, bool auto_clear = true, bool adopt_rng = false):
    calin::simulation::pe_processor::SimpleListPEProcessor(nscope, npix, auto_clear),
    nsample_(round_nreal_to_vector(nsample)),  nadvance_(nsample_advance),
    time_resolution_ns_(time_resolution_ns), sampling_freq_ghz_(1.0/time_resolution_ns_),
    time_advance_(double(nadvance_)*time_resolution_ns_),
    waveform_size_(nsample_ * round_nreal_to_vector(npix_)),
    pe_waveform_(calin::util::memory::aligned_calloc<real_t>(waveform_size_)), 
    v_waveform_(calin::util::memory::aligned_calloc<real_t>(waveform_size_)), 
    rng_(rng), adopt_rng_(adopt_rng)
  {
    if(nsample_ % VCLReal::num_real != 0) {
      throw std::domain_error("Number of samples " + std::to_string(nsample_) 
        + " must be a multiple of vector size " 
        + std::to_string(VCLReal::num_real));
    }

    if((uintptr_t)(const void *)pe_waveform_ % VCLReal::vec_bytes != 0) {
      calin::util::log::LOG(calin::util::log::WARNING) 
        << "pe_waveform_ not aligned on " << VCLReal::vec_bytes << " boundary.";
    }

    if((uintptr_t)(const void *)v_waveform_ % VCLReal::vec_bytes != 0) {
      calin::util::log::LOG(calin::util::log::WARNING) 
        << "v_waveform_ not aligned on " << VCLReal::vec_bytes << " boundary.";
    }

    if(rng_ == nullptr) {
      rng_ = new RNG(__PRETTY_FUNCTION__, "VCLWaveformPEProcessor RNG");
      adopt_rng_ = true;
    }

    clear_waveforms();
  }

  virtual ~VCLWaveformPEProcessor() 
  {
    ::free(pe_waveform_);
    ::free(v_waveform_);
    if(adopt_rng_) {
      delete rng_;
    }
  }

  void transfer_scope_pes_to_waveform(unsigned iscope, const Eigen::VectorXd& channel_time_offset_ns = Eigen::VectorXd())
  {
    // Non-vectorized version. Takes advantage of "approximate" ordering of PEs
    // in time within each pixel to minimize memory accesses. Vectorized version
    // was more complex and was tested to be not much faster.
    validate_iscope_ipix(iscope, 0);
    if(channel_time_offset_ns.size()!=0 and channel_time_offset_ns.size()!=npix_) {
      throw std::domain_error("Time offset vector length is not equal to number of pixels " 
        + std::to_string(channel_time_offset_ns.size()) + " != " + std::to_string(npix_));
    }

    double t0 = get_t0_for_scope(iscope);
    real_t *__restrict__ pe_waveform_ptr = pe_waveform_;
    for(unsigned ipix=0,vpix=0; ipix<npix_; ++ipix,++vpix) {
      if(vpix == VCLReal::num_real) {
        pe_waveform_ptr += nsample_ * VCLReal::num_real;
        vpix = 0;
      }
      double t0ipix = t0;
      if(channel_time_offset_ns.size()) {
        t0ipix -= channel_time_offset_ns[ipix];
      }
      auto pd = scopes_[iscope].pixel_data[ipix];
      if(pd==nullptr) {
        continue;
      }
      int it = int(floor((pd->t[0] - t0ipix) * sampling_freq_ghz_));
      double wt = pd->w[0];
      for(unsigned ipe=1; ipe<pd->npe; ++ipe) {
        int jt = int(floor((pd->t[ipe] - t0ipix) * sampling_freq_ghz_));
        if(it==jt) {
          wt += pd->w[ipe];
        } else {
          if(it>=0 and it<int(nsample_)) {
            pe_waveform_ptr[it*VCLReal::num_real + vpix] += wt;
          }
          it = jt;
          wt = pd->w[ipe];
        }
      }
      if(it>=0 and it<int(nsample_)) {
        pe_waveform_ptr[it*VCLReal::num_real + vpix] += wt;
      }
    }
  }

  void add_nsb_noise_to_waveform(const Eigen::VectorXd& nsb_freq_per_pixel_ghz,
    calin::simulation::detector_efficiency::SplinePEAmplitudeGenerator* pegen = nullptr)
  {
    if(nsb_freq_per_pixel_ghz.size() != int(npix_)) {
      throw std::domain_error("NSB frequency vector length is not equal to number of pixels " 
        + std::to_string(nsb_freq_per_pixel_ghz.size()) + " != " + std::to_string(npix_));
    }

    // Non-vectorized version - a vectorized version would have to use scatter/gather
    // possibly not worth it
    double t_max = double(nsample_);
    real_t *__restrict__ pe_waveform_ptr = pe_waveform_;
    for(unsigned ipix=0,vpix=0; ipix<npix_; ++ipix,++vpix) {
      if(vpix == VCLReal::num_real) {
        pe_waveform_ptr += nsample_ * VCLReal::num_real;
        vpix = 0;
      }
      double rate_samples = sampling_freq_ghz_/nsb_freq_per_pixel_ghz[ipix];
      if(rate_samples<=0) {
        continue;
      }
      double t_samples = 0;
      while(t_samples < t_max) {
        typename VCLArchitecture::double_vt dt_samples = rng_->exponential_double() * rate_samples;
        typename VCLArchitecture::double_vt charge = 1.0;
        if(pegen) {
          charge = pegen->vcl_generate_amplitude<VCLArchitecture>(*rng_);
        }
        typename VCLArchitecture::double_at dt_samples_a;
        dt_samples.store(dt_samples_a);
        typename VCLArchitecture::double_at charge_a;
        charge.store(charge_a);
        for(unsigned insb=0; insb<VCLArchitecture::num_double; ++insb) {
          t_samples += dt_samples_a[insb];
          if(t_samples >= t_max) {
            goto next_pixel;
          }
          int it = int(round(t_samples));
          pe_waveform_ptr[it*VCLReal::num_real + vpix] += charge_a[insb];
        }
      }
      next_pixel:
      ;
    }
  }

  vecX_t estimate_pes_in_window(unsigned window_size, unsigned window_start=0)
  {
    if(window_size == 0) {
      throw std::out_of_range("Window size must be non-zero");
    }
    if(window_size > nsample_) {
      throw std::out_of_range("Window size longer than samples array: " + 
        std::to_string(window_start) + ">" + std::to_string(nsample_));
    }
    if(window_size+window_start > nsample_) {
      throw std::out_of_range("End of window exceeds samples array: " + 
        std::to_string(window_size+window_start) + ">" + std::to_string(nsample_));
    }
    if(nsample_ % VCLReal::num_real != 0) {
      throw std::domain_error("Number of samples " + std::to_string(nsample_) 
        + " is not a multiple of vector size " 
        + std::to_string(VCLReal::num_real));
    }

    vecX_t pes_per_channel_in_window(npix_);

    real_t *__restrict__ pe_waveform_ptr = pe_waveform_ + window_start*VCLReal::num_real;
    for(unsigned ipix=0; ipix<npix_; ipix+=VCLReal::num_real, pe_waveform_ptr += nsample_*VCLReal::num_real) {
      real_vt max_integral = 0;
      real_vt integral = 0;

      unsigned isample=0;
      for(unsigned zsample=window_size*VCLReal::num_real; isample<zsample; isample+=VCLReal::num_real) {
        real_vt samples;
        samples.load_a(pe_waveform_ptr + isample);
        integral += samples;
      }

      max_integral = integral;
      for(unsigned jsample=0, zsample=(nsample_ - window_size - window_start)*VCLReal::num_real; 
          isample<zsample; isample+=VCLReal::num_real, jsample+=VCLReal::num_real) {
        real_vt samples;
        samples.load_a(pe_waveform_ptr + isample);
        integral += samples;
        samples.load_a(pe_waveform_ptr + jsample);
        integral -= samples;
        max_integral = vcl::max(max_integral, integral);
      }

      real_at max_integral_array;
      max_integral.store(max_integral_array);
      for(unsigned jpix=0, mpix=std::min(npix_-ipix,VCLReal::num_real); jpix<mpix; jpix++) {
        pes_per_channel_in_window(ipix+jpix) = max_integral_array[jpix];
      }
    }

    return pes_per_channel_in_window;
  }

  unsigned register_impulse_response(vecX_t& impulse_response, 
    const std::string& units, double window_fraction=0.6)
  {
    ImpulseResponse ir;

    ir.response.resize(nsample_);
    ir.transform.resize(nsample_);

    auto plan = calin::math::fftw_util::plan_r2hc(nsample_, &ir.response[0], &ir.transform[0], FFTW_MEASURE);

    // 1. Store impulse response, truncating or extending if necessary
    ir.response_size = impulse_response.size();    
    if(ir.response_size>nsample_) {
      calin::util::log::LOG(calin::util::log::WARNING) 
        << "Truncating impulse response of size "
        << impulse_response.size() << " to " << nsample_;
      ir.response_size = nsample_;
      ir.response = impulse_response.head(nsample_);
    } else {
      ir.response.setZero();
      ir.response.head(ir.response_size) = impulse_response;
    }

    // 2. Calculate peak, integral and percentiles
    std::vector<double> x(ir.response_size);
    std::vector<double> y(ir.response_size);
    for(unsigned i=0; i<ir.response_size; ++i) {
      double ri = impulse_response(i);
      ir.response_total_integral += ri;
      x[i] = i;
      y[i] = ir.response_total_integral;
      if(ri > ir.response_peak_value) {
        ir.response_peak_value = ri;
        ir.response_peak_index = i;
      }
      if(ir.response_total_integral > ir.response_integral_peak_value) {
        ir.response_integral_peak_value = ir.response_total_integral;
        ir.response_integral_peak_index = i;
      }
    }

    calin::math::interpolation_1d::InterpLinear1D f_of_x(x,y);
    ir.response_window_frac = window_fraction;
    double target = ir.response_integral_peak_value * (1.0-window_fraction)/2.0;
    ir.response_window_lo = 
      calin::math::brent::brent_zero(0, ir.response_integral_peak_index, 
        [&f_of_x,target](double x) { return f_of_x(x)-target; });
    target = ir.response_integral_peak_value * (1.0+window_fraction)/2.0;
    ir.response_window_hi = 
      calin::math::brent::brent_zero(ir.response_window_lo, ir.response_integral_peak_index, 
        [&f_of_x,target](double x) { return f_of_x(x)-target; });

    // 3. Calculate FFT of response function
    ir.transform.resize(nsample_);
    plan.execute();

    // 4. Store data in IR array
    ir.units = units;

    impulse_responses_.emplace_back(ir);
    return impulse_responses_.size()-1;
  }

  unsigned copy_and_shift_impulse_response(unsigned impulse_response_id, 
    unsigned ishift, double scale = 1.0)
  {
    validate_impulse_response_id(impulse_response_id);
    const ImpulseResponse& iro = impulse_responses_[impulse_response_id];

    vecX_t shifted_impulse_response(std::min(iro.response_size+ishift, nsample_));
    shifted_impulse_response.setZero();
    shifted_impulse_response.tail(std::min(iro.response_size, nsample_-ishift)) =
      iro.response.head(std::min(iro.response_size, nsample_-ishift)) * scale;

    return register_impulse_response(shifted_impulse_response, iro.units, iro.response_window_frac);
  }
  
  vecX_t impulse_response(unsigned impulse_response_id) const
  {
    validate_impulse_response_id(impulse_response_id);
    const ImpulseResponse& ir = impulse_responses_[impulse_response_id];
    return ir.response;
  }

  vecX_t impulse_response_fft(unsigned impulse_response_id) const
  {
    validate_impulse_response_id(impulse_response_id);
    const ImpulseResponse& ir = impulse_responses_[impulse_response_id];
    return ir.transform;
  }

  std::string impulse_response_summary(unsigned impulse_response_id) const
  {
    validate_impulse_response_id(impulse_response_id);
    const ImpulseResponse& ir = impulse_responses_[impulse_response_id];
    std::string s;
    s += "Dur=" + calin::util::string::to_string_with_commas(ir.response_size*time_resolution_ns_,1) + "ns";
    s += " Peak=" + calin::util::string::to_string_with_commas(ir.response_peak_value,1) + ir.units;
    s += "@" + calin::util::string::to_string_with_commas(ir.response_peak_index*time_resolution_ns_,1) + "ns";
    s += " IntMax=" + calin::util::string::to_string_with_commas(ir.response_integral_peak_value*time_resolution_ns_,1) + ir.units + ".ns";
    s += "@" + calin::util::string::to_string_with_commas(ir.response_integral_peak_index*time_resolution_ns_,1) + "ns";
    s += " IntWin(" + calin::util::string::to_string_with_commas(ir.response_window_frac*100.0,0);
    s += "%)=" + calin::util::string::to_string_with_commas(ir.response_window_lo*time_resolution_ns_,1);
    s += "->" + calin::util::string::to_string_with_commas(ir.response_window_hi*time_resolution_ns_,1) + "ns";
    return s;
  }

  void convolve_impulse_response_direct(unsigned impulse_response_id, unsigned first_sample_of_interest = 0,
    const vecX_t& pedestal = vecX_t(), const vecX_t& relative_gain = vecX_t(), double white_noise_rms = 0.0)
  {
    if(first_sample_of_interest >= nsample_) {
      throw std::domain_error("First sample of interest must be less than total number of samples: " 
        + std::to_string(first_sample_of_interest) + " >= " + std::to_string(nsample_));
    }

    if(pedestal.size()!=0 and pedestal.size()!=npix_) {
      throw std::domain_error("Pedestal vector length is not equal to number of pixels " 
        + std::to_string(pedestal.size()) + " != " + std::to_string(npix_));
    }

    if(relative_gain.size()!=0 and relative_gain.size()!=npix_) {
      throw std::domain_error("Relative gain vector length is not equal to number of pixels " 
        + std::to_string(relative_gain.size()) + " != " + std::to_string(npix_));
    }

    validate_impulse_response_id(impulse_response_id);
    const ImpulseResponse& ir = impulse_responses_[impulse_response_id];

    const real_t *__restrict__ pe_waveform_ptr = pe_waveform_;
    real_t *__restrict__ v_waveform_ptr = v_waveform_;
    real_t *__restrict__ a_vec = calin::util::memory::aligned_calloc<real_t>(nsample_);
    first_sample_of_interest -= first_sample_of_interest % VCLReal::num_real;
    for(unsigned ipix=0,vpix=0; ipix<npix_; ++ipix,++vpix) {
      if(vpix == VCLReal::num_real) {
        pe_waveform_ptr += nsample_ * VCLReal::num_real;
        v_waveform_ptr += nsample_ * VCLReal::num_real;
        vpix = 0;
      }

      for(unsigned isample=first_sample_of_interest;isample<nsample_;++isample+=VCLReal::num_real) {
        real_vt x = 0;
        if(pedestal.size()) {
          x = pedestal[ipix];
        }
        if(white_noise_rms != 0.0) {
          x += white_noise_rms * rng_->template normal_real<VCLReal>();
        }
        x.store_a(a_vec + isample*VCLReal::num_real);
      }
      
      real_t gain = 1.0;
      if(relative_gain.size()) {
        gain = relative_gain[ipix];
      }
      unsigned isample0 = std::max(first_sample_of_interest,ir.response_size)-ir.response_size;
      for(unsigned isample=isample0; isample<nsample_; ++isample) {
        real_t wt = gain * pe_waveform_ptr[isample*VCLReal::num_real + vpix];
        if(wt==0) {
          continue;
        }
        unsigned jmin = std::max(first_sample_of_interest, isample) - isample;
        unsigned jmax = std::min(unsigned(ir.response_size), nsample_ - isample);
        unsigned jcount = std::max(jmax, jmin) - jmin;
        real_t *__restrict__ a_vec_ptr = a_vec + isample + jmin;
        const real_t *__restrict__ impulse_response_ptr = ir.response.data() + jmin;
        for(unsigned jsample=0; jsample<jcount; ++jsample) {
          *(a_vec_ptr++) += wt * (*(impulse_response_ptr++));
        }
      }

      for(unsigned isample=first_sample_of_interest; isample<nsample_; ++isample) {
        v_waveform_ptr[isample*VCLReal::num_real + vpix] = a_vec[isample];
      }
    }
    ::free(a_vec);
  }

  void convolve_impulse_response_direct_vec(unsigned impulse_response_id, unsigned first_sample_of_interest = 0,
    const vecX_t& pedestal = vecX_t(), const vecX_t& relative_gain = vecX_t(), double white_noise_rms = 0.0)
  {
    if(first_sample_of_interest >= nsample_) {
      throw std::domain_error("First sample of interest must be less than total number of samples: " 
        + std::to_string(first_sample_of_interest) + " >= " + std::to_string(nsample_));
    }

    if(pedestal.size()!=0 and pedestal.size()!=npix_) {
      throw std::domain_error("Pedestal vector length is not equal to number of pixels " 
        + std::to_string(pedestal.size()) + " != " + std::to_string(npix_));
    }

    if(relative_gain.size()!=0 and relative_gain.size()!=npix_) {
      throw std::domain_error("Relative gain vector length is not equal to number of pixels " 
        + std::to_string(relative_gain.size()) + " != " + std::to_string(npix_));
    }

    validate_impulse_response_id(impulse_response_id);
    const ImpulseResponse& ir = impulse_responses_[impulse_response_id];

    for(unsigned ipix=0; ipix<npix_; ipix += VCLReal::num_real) {
      real_t *__restrict__ pe_waveform_ptr = pe_waveform_ + ipix*nsample_;
      real_t *__restrict__ v_waveform_ptr = v_waveform_ + ipix*nsample_;

      // Load the pedestal for this block of pixels
      real_vt x0 = 0;
      if(pedestal.size()) {
        if(ipix + VCLReal::num_real <= npix_) {
          x0.load(pedestal.data() + ipix);
        } else {
          real_at x0_array;
          x0.store_a(x0_array);
          for(unsigned i=0; i+ipix<npix_; i++) {
            x0_array[i] = pedestal(ipix+i);
          }
          x0.load_a(x0_array);
        }
      }

      // Initialize waveform with pedestal + white noise
      for(unsigned isample=first_sample_of_interest;isample<nsample_;++isample) {
        real_vt x = x0;
        if(white_noise_rms != 0.0) {
          x += white_noise_rms * rng_->template normal_real<VCLReal>();
        }
        x.store_a(v_waveform_ptr + isample*VCLReal::num_real);
      }

      real_vt gain = 1.0;
      if(relative_gain.size()) {
        if(ipix + VCLReal::num_real <= npix_) {
          gain.load(relative_gain.data() + ipix);
        } else {
          real_at gain_array;
          gain.store_a(gain_array);
          for(unsigned i=0; i+ipix<npix_; i++) {
            gain_array[i] = relative_gain(ipix+i);
          }
          gain.load_a(gain_array);
        }
      }

      unsigned isample0 = std::max(first_sample_of_interest,ir.response_size)-ir.response_size;
      for(unsigned isample=isample0; isample<nsample_; ++isample) {
        real_vt wt;
        wt.load_a(pe_waveform_ptr + isample*VCLReal::num_real);
        wt *= gain;
        if(vcl::horizontal_and(wt == 0.0)) {
          continue;
        }
        
        unsigned jmax = std::min(unsigned(ir.response_size), nsample_ - isample);        
        real_t *__restrict__ w_waveform_ptr = v_waveform_ptr + isample*VCLReal::num_real;
        const real_t *__restrict__ impulse_response_ptr = ir.response.data();
        for(unsigned jsample=0; jsample<jmax; ++jsample, w_waveform_ptr+=VCLReal::num_real, ++impulse_response_ptr) {
          real_vt x;
          x.load_a(w_waveform_ptr);
          x += wt * (*impulse_response_ptr);
          x.store_a(w_waveform_ptr);
        }
      }
    }
  }

  void convolve_impulse_response_fft(unsigned impulse_response_id,
    const vecX_t& pedestal = vecX_t(),
    const vecX_t& relative_gain = vecX_t(),
    const vecX_t& noise_spectrum = vecX_t())
  {
    if(pedestal.size()!=0 and pedestal.size()!=npix_) {
      throw std::domain_error("Pedestal vector length is not equal to number of pixels " 
        + std::to_string(pedestal.size()) + " != " + std::to_string(npix_));
    }

    if(relative_gain.size()!=0 and relative_gain.size()!=npix_) {
      throw std::domain_error("Relative gain vector length is not equal to number of pixels " 
        + std::to_string(relative_gain.size()) + " != " + std::to_string(npix_));
    }

    if(noise_spectrum.size()!=0 and noise_spectrum.size()!=nsample_) {
      throw std::domain_error("Noise spectrum vector length is not equal to number of samples " 
        + std::to_string(noise_spectrum.size()) + " != " + std::to_string(nsample_));
    }

    validate_impulse_response_id(impulse_response_id);
    const ImpulseResponse& ir = impulse_responses_[impulse_response_id];

    real_t* a_vec = calin::util::memory::aligned_calloc<real_t>(nsample_);
    real_t* b_vec = calin::util::memory::aligned_calloc<real_t>(nsample_);

    auto fwd = calin::math::fftw_util::plan_r2hc(nsample_, a_vec, b_vec, FFTW_MEASURE);
    auto bwd = calin::math::fftw_util::plan_hc2r(nsample_, a_vec, b_vec, FFTW_MEASURE);
    const real_t scale = 1.0/nsample_;

    const real_t *__restrict__ pe_waveform_ptr = pe_waveform_;
    real_t *__restrict__ v_waveform_ptr = v_waveform_;
    for(unsigned ipix=0,vpix=0; ipix<npix_; ++ipix,++vpix) {
      if(vpix == VCLReal::num_real) {
        pe_waveform_ptr += nsample_ * VCLReal::num_real;
        v_waveform_ptr += nsample_ * VCLReal::num_real;
        vpix = 0;
      }

      for(unsigned isample=0;isample<nsample_;isample++) {
        a_vec[isample] = pe_waveform_ptr[isample*VCLReal::num_real + vpix];
      }

      fwd.execute(); // FFT a_vec -> b_vec

      real_t gain = 1.0;
      if(relative_gain.size()) {
        gain = relative_gain[ipix];
      }

      calin::math::fftw_util::hcvec_scale_and_multiply(a_vec, b_vec, ir.transform.data(),
        nsample_, scale*gain); // b_vec * ir.transform * scale * gain -> a_vec

      if(pedestal.size()) {
        a_vec[0] += pedestal(ipix);
      }

      if(noise_spectrum.size()) {
        for(unsigned isample=0;isample<nsample_;isample+=VCLReal::num_real) {
          real_vt x;
          x.load(&noise_spectrum[isample]);
          if(to_bits(x != 0.0)) {
            x *= rng_->template normal_real<VCLReal>();
            real_vt a;
            a.load_a(a_vec + isample);
            a += x;
            a.store_a(a_vec + isample);
          }
        }
      }

      bwd.execute(); // IFFT a_vec -> b_vec

      for(unsigned isample=0;isample<nsample_;isample++) {
        v_waveform_ptr[isample*VCLReal::num_real + vpix] = b_vec[isample];
      }
    }

    ::free(a_vec);
    ::free(b_vec);
  }

  void convolve_impulse_response_fftw_codelet(unsigned impulse_response_id,
    const vecX_t& pedestal = vecX_t(),
    const vecX_t& relative_gain = vecX_t(),
    const vecX_t& noise_spectrum = vecX_t())
  {
    calin::math::fftw_util::FFTWCodelet<VCLReal> fft;
    if(!fft.has_codelet(nsample_)) {
      throw std::domain_error("FFTW codelet for size " + std::to_string(nsample_) 
        + " not available.");
    }

    if(pedestal.size()!=0 and pedestal.size()!=npix_) {
      throw std::domain_error("Pedestal vector length is not equal to number of pixels " 
        + std::to_string(pedestal.size()) + " != " + std::to_string(npix_));
    }

    if(relative_gain.size()!=0 and relative_gain.size()!=npix_) {
      throw std::domain_error("Relative gain vector length is not equal to number of pixels " 
        + std::to_string(relative_gain.size()) + " != " + std::to_string(npix_));
    }

    if(noise_spectrum.size()!=0 and noise_spectrum.size()!=nsample_) {
      throw std::domain_error("Noise spectrum vector length is not equal to number of samples " 
        + std::to_string(noise_spectrum.size()) + " != " + std::to_string(nsample_));
    }

    validate_impulse_response_id(impulse_response_id);
    const ImpulseResponse& ir = impulse_responses_[impulse_response_id];

    real_vt* a_vec = calin::util::memory::aligned_calloc<real_vt>(nsample_);
    real_vt* b_vec = calin::util::memory::aligned_calloc<real_vt>(nsample_);

    real_t scale = 1.0/nsample_;

    const real_t *__restrict__ pe_waveform_ptr = pe_waveform_;
    real_t *__restrict__ v_waveform_ptr = v_waveform_;
    for(unsigned ipix=0; ipix<npix_; ipix+=VCLReal::num_real) {
      // Load data from "pe_waveform_" to "a_vec"
      for(unsigned isample=0; isample<nsample_; ++isample) {
        a_vec[isample].load_a(pe_waveform_ptr);
        pe_waveform_ptr += VCLReal::num_real;
      }

      // Do FFT of "a_vec" into "b_vec"
      fft.r2hc(nsample_, a_vec, b_vec);

      // Extract relative gain from array
      real_vt gain = 1.0;
      if(relative_gain.size()) {
        if(ipix + VCLReal::num_real <= npix_) {
          gain.load(relative_gain.data() + ipix);
        } else {
          real_at gain_array;
          gain.store_a(gain_array);
          for(unsigned i=0; i+ipix<npix_; i++) {
            gain_array[i] = relative_gain(ipix+i);
          }
          gain.load_a(gain_array);
        }
      }

      // Multiply FFT in "b_vec" by impulse response and gain, tranferring back into "a_vec"
      calin::math::fftw_util::hcvec_scale_and_multiply_non_overlapping(a_vec, b_vec, ir.transform.data(), 
        nsample_, gain*scale);

      // Add pedestal in frequency domain as DC offset before inverse transform
      if(pedestal.size()) {
        real_vt ped = 0;
        if(ipix + VCLReal::num_real <= npix_) {
          ped.load(pedestal.data() + ipix);
        } else {
          real_at ped_array;
          ped.store_a(ped_array);
          for(unsigned i=0; i+ipix<npix_; i++) {
            ped_array[i] = pedestal(ipix+i);
          }
          ped.load_a(ped_array);
        }
        *a_vec += ped;
      }

      // Add Gaussian noise with given spectrum
      if(noise_spectrum.size()) {
        const real_t *__restrict__ noise_spectrum_ptr = noise_spectrum.data();
        int isample = 0;
        for(int zsample = int(nsample_)-3; isample<zsample; isample+=4) {
          // Unroll here and use normal pair function which is (might be) faster
          real_vt x0, x1, x2, x3;
          rng_->normal_pair_real(x0, x1);
          rng_->normal_pair_real(x2, x3);
          a_vec[isample]   += x0 * noise_spectrum_ptr[isample];
          a_vec[isample+1] += x1 * noise_spectrum_ptr[isample+1];
          a_vec[isample+2] += x2 * noise_spectrum_ptr[isample+2];
          a_vec[isample+3] += x3 * noise_spectrum_ptr[isample+3];
        }
        for(int zsample = int(nsample_); isample<zsample; isample++) {
          real_vt x = rng_->template normal_real<VCLReal>();
          a_vec[isample] += x * noise_spectrum_ptr[isample];
        }        
      }

      // Do inverse-FFT of "a_vec" into "b_vec"
      fft.hc2r(nsample_, a_vec, b_vec);

      // Store data from "b_vec" into "v_waveform_"
      for(unsigned isample=0; isample<nsample_; ++isample) {
        b_vec[isample].store_a(v_waveform_ptr);
        v_waveform_ptr += VCLReal::num_real;
      }
    }
    ::free(a_vec);
    ::free(b_vec);
  }

  void convolve_multiple_impulse_response_fftw_codelet(
    const Eigen::VectorXi& impulse_response_id,
    const vecX_t& pedestal = vecX_t(),
    const vecX_t& relative_gain = vecX_t(),
    const matX_t& noise_spectrum = matX_t())
  {
    calin::math::fftw_util::FFTWCodelet<VCLReal> fft;
    if(!fft.has_codelet(nsample_)) {
      throw std::domain_error("FFTW codelet for size " + std::to_string(nsample_) 
        + " not available.");
    }

    if(impulse_response_id.size() != npix_) {
      throw std::domain_error("Impulse response id vector length is not equal to number of pixels " 
        + std::to_string(impulse_response_id.size()) + " != " + std::to_string(npix_));
    }

    if(pedestal.size()!=0 and pedestal.size()!=npix_) {
      throw std::domain_error("Pedestal vector length is not equal to number of pixels " 
        + std::to_string(pedestal.size()) + " != " + std::to_string(npix_));
    }
    
    if(relative_gain.size()!=0 and relative_gain.size()!=npix_) {
      throw std::domain_error("Relative gain vector length is not equal to number of pixels " 
        + std::to_string(relative_gain.size()) + " != " + std::to_string(npix_));
    }

    if(noise_spectrum.size()!=0 and (noise_spectrum.rows()!=nsample_
        or (noise_spectrum.cols()!=1 and noise_spectrum.cols()!=npix_))) {
      throw std::domain_error("Noise spectrum matrix has incoorect size: [" 
        + std::to_string(noise_spectrum.rows()) + "," + std::to_string(noise_spectrum.cols()) + "] != ["
        + std::to_string(nsample_) + ",1] or ["
        + std::to_string(nsample_) + "," + std::to_string(npix_) + "]");
    }

    int iri_max = std::numeric_limits<int>::min();
    int iri_min = std::numeric_limits<int>::max();
    for(unsigned ipix=0; ipix<npix_; ipix++) {
      int iri = impulse_response_id(ipix);
      iri_max = std::max(iri_max, iri);
      iri_min = std::min(iri_min, iri);
    }

    validate_impulse_response_id(iri_max);
    if(iri_min<0) {
      throw std::out_of_range("Invalid impulse response id cannot be negative: " 
        + std::to_string(iri_min));
    }

    real_vt* a_vec = calin::util::memory::aligned_calloc<real_vt>(nsample_);
    real_vt* b_vec = calin::util::memory::aligned_calloc<real_vt>(nsample_);
    real_vt* c_vec = calin::util::memory::aligned_calloc<real_vt>(nsample_);
  
    real_t scale = 1.0/nsample_;

    const real_t *__restrict__ pe_waveform_ptr = pe_waveform_;
    real_t *__restrict__ v_waveform_ptr = v_waveform_;
    for(unsigned ipix=0; ipix<npix_; ipix+=VCLReal::num_real) {
      // Load data from "pe_waveform_" to "a_vec"
      for(unsigned isample=0; isample<nsample_; ++isample) {
        a_vec[isample].load_a(pe_waveform_ptr);
        pe_waveform_ptr += VCLReal::num_real;
      }
    
      // Do FFT of "a_vec" into "b_vec"
      fft.r2hc(nsample_, a_vec, b_vec);

      // Load data from "impulse_respose" into "c_vec", block by block, transposing as we go along
      for(unsigned isample=0; isample<nsample_; isample += VCLReal::num_real) {
        real_vt block[VCLReal::num_real]; // square matrix of doubles
        for(unsigned jpix=0, mpix=std::min(npix_-ipix,VCLReal::num_real); jpix<mpix; jpix++) {
          int irid = impulse_response_id[ipix + jpix];
          block[jpix].load_a(impulse_responses_[irid].transform.data() + isample);
        }
        calin::util::vcl::transpose(block);
        for(unsigned jsample = 0; jsample<VCLReal::num_real; jsample++) {
          c_vec[isample+jsample] = block[jsample];
        }
      }

      // Extract relative gain from array
      real_vt gain = 1.0;
      if(relative_gain.size()) {
        if(ipix + VCLReal::num_real <= npix_) {
          gain.load(relative_gain.data() + ipix);
        } else {
          real_at gain_array;
          gain.store_a(gain_array);
          for(unsigned i=0; i+ipix<npix_; i++) {
            gain_array[i] = relative_gain(ipix+i);
          }
          gain.load_a(gain_array);
        }
      }

      // Multiply FFT in "b_vec" by impulse response in "c_vec" and gain, then transfer back into "a_vec"
      calin::math::fftw_util::hcvec_scale_and_multiply_non_overlapping(a_vec, b_vec, c_vec, 
        nsample_, gain*scale);

      // Add pedestal in frequency domain as DC offset before inverse transform
      if(pedestal.size()) {
        real_vt ped = 0;
        if(ipix + VCLReal::num_real <= npix_) {
          ped.load(pedestal.data() + ipix);
        } else {
          real_at ped_array;
          ped.store_a(ped_array);
          for(unsigned i=0; i+ipix<npix_; i++) {
            ped_array[i] = pedestal(ipix+i);
          }
          ped.load_a(ped_array);
        }
        *a_vec += ped;
      }

      // Add Gaussian noise with given spectrum
      if(noise_spectrum.cols() == 1) {
        // Unique noise spectrum for all channels
        int isample = 0;
        const real_t *__restrict__ noise_spectrum_ptr = noise_spectrum.data();
        for(int zsample = int(nsample_)-3; isample<zsample; isample+=4) {
          // Unroll by two here and use Gaussian pair function to produce 2 deviates
          real_vt x0, x1, x2, x3;
          rng_->normal_pair_real(x0, x1);
          rng_->normal_pair_real(x2, x3);
          a_vec[isample]   += x0 * noise_spectrum_ptr[isample];
          a_vec[isample+1] += x1 * noise_spectrum_ptr[isample+1];
          a_vec[isample+2] += x2 * noise_spectrum_ptr[isample+2];
          a_vec[isample+3] += x3 * noise_spectrum_ptr[isample+3];
        }
        for(int zsample = int(nsample_); isample<zsample; isample++) {
          real_vt x = rng_->template normal_real<VCLReal>();
          a_vec[isample] += x * noise_spectrum_ptr[isample];
        }        
      } else if (noise_spectrum.size()) {
        // Load data from "noise_spectrum", block by block, transposing as we go along
        const real_t *__restrict__ noise_spectrum_ptr = noise_spectrum.data();
        for(unsigned isample=0; isample<nsample_; isample += VCLReal::num_real) {
          real_vt block[VCLReal::num_real]; // square matrix of doubles
          for(unsigned jpix=0, mpix=std::min(npix_-ipix,VCLReal::num_real); jpix<mpix; jpix++) {
            block[jpix].load_a(noise_spectrum_ptr + (ipix+jpix)*nsample_ + isample);
          }
          calin::util::vcl::transpose(block);
          // One of these three possible cases which is chosen at compile time
          if(VCLReal::num_real % 4 == 0) {        
            // If there is multiple of 4 reals in the vector then do an unrolled 
            // loop generating two pairs of Gaussian deviates each iteration.
            // This branch is chose for all AVX/AVX-2 and AVX-512 float and double 
            // architectures, and in the SSE (128 bit) float architecture.
            for(unsigned jsample = 0; jsample<VCLReal::num_real; jsample+=4) {
              real_vt x0, x1, x2, x3;
              rng_->normal_pair_real(x0, x1);
              rng_->normal_pair_real(x2, x3);
              a_vec[isample]   += x0 * block[jsample];
              a_vec[isample+1] += x1 * block[jsample+1];
              a_vec[isample+2] += x2 * block[jsample+2];
              a_vec[isample+3] += x3 * block[jsample+3];
            }
          } else if (VCLReal::num_real % 2 == 0) {
            // If there is multiple of 2 reals in the vector then do loop 
            // generating one pairs of Gaussian deviates each iteration. This 
            // branch is chosen is for the 128-bit SSE double architecture.
            for(unsigned jsample = 0; jsample<VCLReal::num_real; jsample+=2) {
              real_vt x0, x1;
              rng_->normal_pair_real(x0, x1);
              a_vec[isample]   += x0 * block[jsample];
              a_vec[isample+1] += x1 * block[jsample+1];
            }
          } else {
            // Otherwise just generate one Gaussian per iteration. There is
            // no case in VCL where this branch is chosen.
            for(unsigned jsample = 0; jsample<VCLReal::num_real; jsample++) {
              real_vt x = rng_->template normal_real<VCLReal>();
              a_vec[isample] += x * block[jsample];
            }
          }
        }
      }

      // Do inverse-FFT of "a_vec" into "b_vec"
      fft.hc2r(nsample_, a_vec, b_vec);

      // Store data from "b_vec" into "v_waveform_"
      for(unsigned isample=0; isample<nsample_; ++isample) {
        b_vec[isample].store_a(v_waveform_ptr);
        v_waveform_ptr += VCLReal::num_real;
      }
    }
    ::free(a_vec);
    ::free(b_vec);
    ::free(c_vec);
  }

  vecX_t ac_coupling_offset(const Eigen::VectorXi& impulse_response_id, 
    const Eigen::VectorXd& nsb_freq_per_pixel_ghz,
    calin::simulation::detector_efficiency::SplinePEAmplitudeGenerator* pegen = nullptr,
    const vecX_t& relative_gain = vecX_t())
  {
    int iri_max = std::numeric_limits<int>::min();
    int iri_min = std::numeric_limits<int>::max();
    for(unsigned ipix=0; ipix<impulse_response_id.size(); ipix++) {
      int iri = impulse_response_id(ipix);
      iri_max = std::max(iri_max, iri);
      iri_min = std::min(iri_min, iri);
    }

    validate_impulse_response_id(iri_max);
    if(iri_min<0) {
      throw std::out_of_range("Invalid impulse response id cannot be negative: " 
        + std::to_string(iri_min));
    }

    if(nsb_freq_per_pixel_ghz.size() != impulse_response_id.size()) {
      throw std::domain_error("NSB frequency and impulse response vectors must have same size " 
        + std::to_string(nsb_freq_per_pixel_ghz.size()) + " != " + std::to_string(impulse_response_id.size()));
    }

    if(relative_gain.size()!=0 and relative_gain.size()!=impulse_response_id.size()) {
      throw std::domain_error("Relative gain and impulse response vectors must have same lengths " 
        + std::to_string(relative_gain.size()) + " != " + std::to_string(impulse_response_id.size()));
    }

    double pe_amp = 1.0;
    if(pegen) {
      pe_amp = pegen->mean_amplitude();
    }

    vecX_t offset = (nsb_freq_per_pixel_ghz * time_resolution_ns_ * pe_amp).cast<real_t>();

    for(unsigned iir=0; iir<impulse_response_id.size(); iir++) {
      offset[iir] *= impulse_responses_[impulse_response_id[iir]].response.sum();
    }
    
    if(relative_gain.size()) {
      offset *= relative_gain;
    }

    return offset;
  }

  vecX_t ac_coupling_offset(int impulse_response_id, 
    const Eigen::VectorXd& nsb_freq_per_pixel_ghz,
    calin::simulation::detector_efficiency::SplinePEAmplitudeGenerator* pegen = nullptr,
    const vecX_t& relative_gain = vecX_t())
  {
    Eigen::VectorXi all_impulse_response_id(nsb_freq_per_pixel_ghz.size());
    all_impulse_response_id.setConstant(impulse_response_id);
    return ac_coupling_offset(all_impulse_response_id, nsb_freq_per_pixel_ghz, pegen = nullptr,relative_gain);
  }

  void clear_waveforms()
  {
    std::fill(pe_waveform_, pe_waveform_+waveform_size_, 0.0);
    std::fill(v_waveform_, v_waveform_+waveform_size_, 0.0);
  }

  void clear() override
  {
    SimpleListPEProcessor::clear();
    clear_waveforms();
  }

  std::string fftw_plans_summary() const
  {
    real_t* a_vec = calin::util::memory::aligned_calloc<real_t>(nsample_);
    real_t* b_vec = calin::util::memory::aligned_calloc<real_t>(nsample_);

    auto fwd = calin::math::fftw_util::plan_r2hc(nsample_, a_vec, b_vec, FFTW_MEASURE);
    auto bwd = calin::math::fftw_util::plan_hc2r(nsample_, a_vec, b_vec, FFTW_MEASURE);

    std::string os;
    os += "Photo-electron forward transformation:\n";
    os += "--------------------------------------\n";
    os += fwd.print();

    os += "\nAmplitude backward transformation:\n";
    os += "----------------------------------\n";
    os += bwd.print();

    ::free(a_vec);
    ::free(b_vec);

    return os;
  }
  
  void inject_pe(unsigned pixel_id, unsigned t_sample, double amplitude = 1.0) 
  {
    if(pixel_id>=npix_) {
      throw std::out_of_range("Pixel index out of range in inject_pe: " 
        + std::to_string(pixel_id) + " >= " + std::to_string(npix_));
    }
    if(t_sample>=nsample_) {
      throw std::out_of_range("Sample time index out of range in inject_pe: " 
        + std::to_string(t_sample) + " >= " + std::to_string(nsample_));
    }
    unsigned ivec = pixel_id/VCLReal::num_real;
    unsigned vpix = pixel_id - ivec*VCLReal::num_real;
    pe_waveform_[ivec*nsample_*VCLReal::num_real + t_sample*VCLReal::num_real + vpix] += amplitude;
  }

  void inject_n_pes(unsigned pixel_id, unsigned npe, double t0_samples,
    calin::simulation::detector_efficiency::SplinePEAmplitudeGenerator* pegen = nullptr,
    double time_spread_ns = 0.0)
  {
    double time_spread_samples = time_spread_ns*sampling_freq_ghz_;

    unsigned ivec = pixel_id/VCLReal::num_real;
    unsigned vpix = pixel_id - ivec*VCLReal::num_real;
    real_t *__restrict__ pe_waveform_ptr = pe_waveform_ + ivec*nsample_*VCLReal::num_real + vpix;
    while(npe) {
      typename VCLArchitecture::double_vt t = t0_samples;
      if(time_spread_ns > 0) {
        t += rng_->normal_double() * time_spread_samples;
      } else if (time_spread_ns < 0) {
        t -= rng_->uniform_double_zc(time_spread_samples);
      }        
      typename VCLArchitecture::double_vt q;
      if(pegen) {
        q = pegen->vcl_generate_amplitude<VCLArchitecture>(*rng_);
      } else {
        q = 1.0;
      }

      typename VCLArchitecture::double_at q_a;
      q.store_a(q_a);

      typename VCLArchitecture::int64_vt it = round_to_int64_limited(t);
      typename VCLArchitecture::int64_at it_a;
      it.store_a(it_a);
      for(unsigned jvec=0;jvec<VCLArchitecture::num_double && npe; jvec++,npe--) {
        unsigned isample = it_a[jvec];
        if(isample>=0 and isample<nsample_) {
          pe_waveform_ptr[isample * VCLReal::num_real] += q_a[jvec];
        }
      }
    }
  }

  void inject_poisson_pes(unsigned pixel_id, double lambda, double t0_samples,
    calin::simulation::detector_efficiency::SplinePEAmplitudeGenerator* pegen = nullptr,
    double time_spread_ns = 0.0)
  {
    calin::math::rng::VCLToScalarRNGCore scalar_core(rng_->core());
    calin::math::rng::RNG scalar_rng(&scalar_core);
    unsigned npe = scalar_rng.poisson(lambda);
    inject_n_pes(pixel_id, npe, t0_samples, pegen, time_spread_ns);
  }

  void inject_poisson_pes(const Eigen::VectorXd& lambda, const Eigen::VectorXd& t0_samples, 
    calin::simulation::detector_efficiency::SplinePEAmplitudeGenerator* pegen = nullptr,
    const Eigen::VectorXd& time_spread_ns = Eigen::VectorXd())
  {
    if(lambda.size() != npix_) {
      throw std::domain_error("Lambda vector length is not equal to number of pixels " 
        + std::to_string(lambda.size()) + " != " + std::to_string(npix_));
    }
    if(t0_samples.size() != 1 and t0_samples.size() != npix_) {
      throw std::domain_error("T0 vector length must be 1 or equal to number of pixels " 
        + std::to_string(t0_samples.size()) + " != 1 or " + std::to_string(npix_));
    }
    if(time_spread_ns.size() != 0 and time_spread_ns.size() != 1 and time_spread_ns.size() != npix_) {
      throw std::domain_error("Time spread vector length must be 0, 1 or equal to number of pixels " 
        + std::to_string(time_spread_ns.size()) + " != 0, 1 or " + std::to_string(npix_));
    }

    calin::math::rng::RNG& scalar_rng(rng_->scalar_rng());
    for(unsigned ipix=0; ipix<npix_; ipix++) {
      unsigned npe = scalar_rng.poisson(lambda[ipix]);
      double t0 = t0_samples[0];
      double time_spread = 0;
      if(t0_samples.size() > 1) {
        t0 = t0_samples[ipix];
      }
      if(time_spread_ns.size() > 1) {
        time_spread = time_spread_ns[ipix];
      } else if (time_spread_ns.size() == 1) {
        time_spread = time_spread_ns[0];
      }
      inject_n_pes(ipix, npe, t0, pegen, time_spread);
    }
  }

  real_t noise_spectrum_var(const vecX_t& noise_spectrum) const 
  {
    using calin::math::special::SQR;
    if(noise_spectrum.size()!=nsample_) {
      throw std::domain_error("Noise spectrum vector length is not equal to number of samples " 
        + std::to_string(noise_spectrum.size()) + " != " + std::to_string(nsample_));
    }

    const real_t* r = noise_spectrum.data();
    const real_t* c = noise_spectrum.data() + nsample_;
    real_t var = SQR(*r++);
    c--;
    while(r<c) {
      var += 2.0*(SQR(*r++) + SQR(*c--));
    }
    if(r==c) {
      var += SQR(*r++);
    }
    return var;
  }

  Eigen::VectorXd spectral_frequencies_ghz(bool imaginary_negative = false) const 
  {
    Eigen::VectorXd freq(nsample_);
    calin::math::fftw_util::hcvec_fftfreq(freq.data(), nsample_, time_resolution_ns_, imaginary_negative);
    return freq;
  }

  matX_t pe_waveform() const 
  { 
    matX_t wf(nsample_, npix_);
    // Transfer data from "pe_waveform_" into "wf", block by block, transposing as we go along
    real_t *__restrict__ pe_waveform_ptr = pe_waveform_;
    for(unsigned ipix=0; ipix<npix_; ipix+=VCLReal::num_real) {
      for(unsigned isample=0; isample<nsample_; isample += VCLReal::num_real) {
        real_vt block[VCLReal::num_real]; // square matrix of doubles
        for(unsigned jsample = 0; jsample<VCLReal::num_real; jsample++) {
          block[jsample].load_a(pe_waveform_ptr);
          pe_waveform_ptr += VCLReal::num_real;
        }
        calin::util::vcl::transpose(block);
        for(unsigned jpix=0, mpix=std::min(npix_-ipix,VCLReal::num_real); jpix<mpix; jpix++) {
          block[jpix].store_a(wf.data() + (ipix+jpix)*nsample_ + isample);
        }
      }
    }
    return wf; 
  }

  matX_t v_waveform() const
  { 
    matX_t wf(nsample_, npix_);
    // Transfer data from "pe_waveform_" into "wf", block by block, transposing as we go along
    real_t *__restrict__ v_waveform_ptr = v_waveform_;
    for(unsigned ipix=0; ipix<npix_; ipix+=VCLReal::num_real) {
      for(unsigned isample=0; isample<nsample_; isample += VCLReal::num_real) {
        real_vt block[VCLReal::num_real]; // square matrix of doubles
        for(unsigned jsample = 0; jsample<VCLReal::num_real; jsample++) {
          block[jsample].load_a(v_waveform_ptr);
          v_waveform_ptr += VCLReal::num_real;
        }
        calin::util::vcl::transpose(block);
        for(unsigned jpix=0, mpix=std::min(npix_-ipix,VCLReal::num_real); jpix<mpix; jpix++) {
          block[jpix].store_a(wf.data() + (ipix+jpix)*nsample_ + isample);
        }
      }
    }
    return wf; 
  }

  unsigned register_camera_response(const Eigen::VectorXd& nsb_freq_per_pixel_ghz,
    calin::simulation::detector_efficiency::SplinePEAmplitudeGenerator* pegen,
    const Eigen::VectorXi& impulse_response_id,
    bool add_ac_coupling_offset,
    const vecX_t& relative_gain = vecX_t(),
    const vecX_t& pedestal = vecX_t(),
    const matX_t& noise_spectrum = matX_t(),
    bool adopt_pegen = false)
  {
    if(nsb_freq_per_pixel_ghz.size() != int(npix_)) {
      throw std::domain_error("NSB frequency vector length is not equal to number of pixels " 
        + std::to_string(nsb_freq_per_pixel_ghz.size()) + " != " + std::to_string(npix_));
    }

    if(impulse_response_id.size() != 1 and impulse_response_id.size() != npix_) {
      throw std::domain_error("Impulse response id must be scalar or vector of length equal to number of pixels " 
        + std::to_string(impulse_response_id.size()) + " != 1 or " + std::to_string(npix_));
    }

    if(pedestal.size()!=0 and pedestal.size()!=npix_) {
      throw std::domain_error("Pedestal vector length is not equal to number of pixels " 
        + std::to_string(pedestal.size()) + " != " + std::to_string(npix_));
    }
    
    if(relative_gain.size()!=0 and relative_gain.size()!=npix_) {
      throw std::domain_error("Relative gain vector length is not equal to number of pixels " 
        + std::to_string(relative_gain.size()) + " != " + std::to_string(npix_));
    }

    if(noise_spectrum.size()!=0 and (noise_spectrum.rows()!=nsample_
        or (noise_spectrum.cols()!=1 and noise_spectrum.cols()!=npix_))) {
      throw std::domain_error("Noise spectrum matrix has incoorect size: [" 
        + std::to_string(noise_spectrum.rows()) + "," + std::to_string(noise_spectrum.cols()) + "] != ["
        + std::to_string(nsample_) + ",1] or ["
        + std::to_string(nsample_) + "," + std::to_string(npix_) + "]");
    }

    CameraResponse cr;
    cr.nsb_freq_per_pixel_ghz    = nsb_freq_per_pixel_ghz;
    cr.pegen                     = pegen;
    cr.impulse_response_id       = impulse_response_id;
    cr.pedestal                  = pedestal;
    cr.relative_gain             = relative_gain;
    cr.noise_spectrum            = noise_spectrum;
    cr.adopt_pegen               = adopt_pegen;

    if(add_ac_coupling_offset) {
      vecX_t offset;
      if(impulse_response_id.size() == 1) {
        offset = ac_coupling_offset(impulse_response_id[0], nsb_freq_per_pixel_ghz, pegen, relative_gain);
      } else {
        offset = ac_coupling_offset(impulse_response_id, nsb_freq_per_pixel_ghz, pegen, relative_gain);
      }
      if(cr.pedestal.size() == 0) {
        cr.pedestal = -offset;
      } else {
        cr.pedestal -= offset;
      }
    }
    camera_responses_.emplace_back(std::move(cr));
    return camera_responses_.size() - 1;
  }

  void apply_camera_response_fftw_codelet(unsigned camera_response_id, int iscope = -1,
    const Eigen::VectorXd& channel_time_offset_ns = Eigen::VectorXd()) 
  {
    auto& cr = camera_responses_[camera_response_id];
    if(iscope>=0) {
      validate_iscope_ipix(iscope, 0);
      if(channel_time_offset_ns.size()!=0 and channel_time_offset_ns.size()!=npix_) {
        throw std::domain_error("Time offset vector length is not equal to number of pixels " 
          + std::to_string(channel_time_offset_ns.size()) + " != " + std::to_string(npix_));
      }

      transfer_scope_pes_to_waveform(iscope, channel_time_offset_ns);
      add_nsb_noise_to_waveform(cr.nsb_freq_per_pixel_ghz, cr.pegen);
    }

    validate_camera_response_id(camera_response_id);
    if(cr.impulse_response_id.size() == 1) {
      convolve_impulse_response_fftw_codelet(
        cr.impulse_response_id[0], cr.pedestal, cr.relative_gain, cr.noise_spectrum);
    } else {
      convolve_multiple_impulse_response_fftw_codelet(
        cr.impulse_response_id, cr.pedestal, cr.relative_gain, cr.noise_spectrum);
    }
  }

  Eigen::VectorXd zero_nsb() const { return Eigen::VectorXd::Zero(npix_); }
  vecX_t no_pedestal() const { return vecX_t(); }
  vecX_t no_relative_gain() const { return vecX_t(); }
  vecX_t unity_relative_gain() const { return vecX_t::Ones(npix_); }
  matX_t no_noise_spectrum() const { return matX_t(); }

private:
#ifndef SWIG
  static inline unsigned round_nreal_to_vector(unsigned n) {
    return ((n + VCLReal::num_real - 1U)/VCLReal::num_real)*VCLReal::num_real;
  } 

  inline double get_t0_for_scope(unsigned iscope) const {
    return scopes_[iscope].tmin - time_advance_;
  }

  inline void validate_impulse_response_id(unsigned impulse_response_id) const {
    if(impulse_response_id >= impulse_responses_.size()) {
      throw std::out_of_range("Invalid impulse response id: " 
        + std::to_string(impulse_response_id) 
        + " >= " + std::to_string(impulse_responses_.size()));
    }
  }

  inline void validate_camera_response_id(unsigned camera_response_id) const {
    if(camera_response_id >= camera_responses_.size()) {
      throw std::out_of_range("Camera impulse response id: " 
        + std::to_string(camera_response_id) 
        + " >= " + std::to_string(camera_responses_.size()));
    }
  }

  struct ImpulseResponse {
    unsigned response_size;
    vecX_t response;
    vecX_t transform;
    int response_peak_index = 0;
    double response_peak_value = -std::numeric_limits<double>::infinity();
    int response_integral_peak_index = 0;
    double response_integral_peak_value = -std::numeric_limits<double>::infinity();
    double response_window_frac;
    double response_window_lo;
    double response_window_hi;
    double response_total_integral = 0;
    std::string units = "";
  };

  struct CameraResponse {
    CameraResponse() = default;
    CameraResponse(CameraResponse&& o): 
      nsb_freq_per_pixel_ghz(o.nsb_freq_per_pixel_ghz),
      pegen(o.pegen), impulse_response_id(o.impulse_response_id),
      pedestal(std::move(o.pedestal)),
      relative_gain(std::move(o.relative_gain)),
      noise_spectrum(std::move(o.noise_spectrum)),
      adopt_pegen(o.adopt_pegen)
    {
      o.pegen = nullptr;
    }
    CameraResponse& operator=(CameraResponse&& o) {
      if(this != &o) {
        nsb_freq_per_pixel_ghz = o.nsb_freq_per_pixel_ghz;
        pegen = o.pegen;
        impulse_response_id = o.impulse_response_id;
        pedestal = std::move(o.pedestal);
        relative_gain = std::move(o.relative_gain);
        noise_spectrum = std::move(o.noise_spectrum);
        adopt_pegen = o.adopt_pegen;
        o.pegen = nullptr;
      }
      return *this;
    }
    ~CameraResponse() {
      if(adopt_pegen) {
        delete pegen;
      }
    }
    Eigen::VectorXd nsb_freq_per_pixel_ghz;
    calin::simulation::detector_efficiency::SplinePEAmplitudeGenerator* pegen = nullptr;
    Eigen::VectorXi impulse_response_id;
    vecX_t pedestal;
    vecX_t relative_gain;
    matX_t noise_spectrum;
    bool adopt_pegen = false;
  };

  unsigned nsample_;
  unsigned nadvance_;
  double time_resolution_ns_;
  double sampling_freq_ghz_;
  double time_advance_;
  unsigned waveform_size_;
  real_t* pe_waveform_ = nullptr;
  real_t* v_waveform_ = nullptr;
  std::vector<ImpulseResponse> impulse_responses_;
  std::vector<CameraResponse> camera_responses_;
  RNG* rng_ = nullptr;
  bool adopt_rng_ = false;
#endif // ifndef SWIG
};


} } } // namespace calin::simulations::vcl_pe_processor
