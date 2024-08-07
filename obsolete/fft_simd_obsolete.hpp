/*

   calin/math/fft_simd.hpp -- Stephen Fegan -- 2018-02-21

   SIMD FFT functions using codelets from FFTW/genfft

   Copyright 2018, Stephen Fegan <sfegan@llr.in2p3.fr>
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

#include <vector>
#include <algorithm>
#include <fftw3.h>
#include <immintrin.h>

#include <util/memory.hpp>

namespace calin { namespace math { namespace fft_simd {

// ****************************************************************************
// *************************** DFT REAL <-> COMPLEX ***************************
// ****************************************************************************

template<typename T> void shuffle_complex_s2v(T& r, T&c) { }
template<typename T> void shuffle_complex_v2s(T& r, T&c) { }

#if defined(__AVX__)
inline void shuffle_complex_s2v(__m256& r, __m256&c)
{
  __m256 t;
  t = _mm256_insertf128_ps(r, _mm256_extractf128_ps(c,0), 1);
  c = _mm256_insertf128_ps(c, _mm256_extractf128_ps(r,1), 0);
  r = _mm256_shuffle_ps(t, c, 0b10001000U);
  c = _mm256_shuffle_ps(t, c, 0b11011101U);
}

inline void shuffle_complex_v2s(__m256& r, __m256&c)
{
  __m256 t;
  t = _mm256_unpacklo_ps(r, c);
  c = _mm256_unpackhi_ps(r, c);
  r = _mm256_insertf128_ps(t, _mm256_extractf128_ps(c,0), 1);
  c = _mm256_insertf128_ps(c, _mm256_extractf128_ps(t,1), 0);
}

inline void shuffle_complex_s2v(__m256d& r, __m256d&c)
{
  __m256d t;
  t = _mm256_insertf128_pd(r, _mm256_extractf128_pd(c,0), 1);
  c = _mm256_insertf128_pd(c, _mm256_extractf128_pd(r,1), 0);
  r = _mm256_unpacklo_pd(t, c);
  c = _mm256_unpackhi_pd(t, c);
}

inline void shuffle_complex_v2s(__m256d& r, __m256d&c)
{
  __m256d t;
  t = _mm256_unpacklo_pd(r, c);
  c = _mm256_unpackhi_pd(r, c);
  r = _mm256_insertf128_pd(t, _mm256_extractf128_pd(c,0), 1);
  c = _mm256_insertf128_pd(c, _mm256_extractf128_pd(t,1), 0);
}
#endif

template<typename T> class FixedSizeRealToComplexDFT
{
public:
  FixedSizeRealToComplexDFT(unsigned N, unsigned real_stride = 1, unsigned complex_stride = 1):
    N_(N), rs_(real_stride), cs_(complex_stride) { assert(rs_ != 0); assert(cs_ != 0); };
  virtual ~FixedSizeRealToComplexDFT() { }
  virtual void r2c(T* r_in, T* c_out) = 0;
  virtual void c2r(T* r_out, T* c_in) = 0;

  // Utility functions
  int real_stride() const { return rs_; }
  int complex_stride() const { return cs_; }
  unsigned num_real() const { return N_; }
  unsigned num_complex() const { return N_/2+1; }
  unsigned real_array_size() const { return num_real()*rs_; }
  unsigned complex_array_size() const { return 2*num_complex()*cs_; }
  T* alloc_real_array() const {
    return calin::util::memory::aligned_calloc<T>(real_array_size()); }
  T* alloc_complex_array() const {
    return calin::util::memory::aligned_calloc<T>(complex_array_size()); }
protected:
  unsigned N_;
  unsigned rs_;
  unsigned cs_;
};

template<typename T> class FFTW_FixedSizeRealToComplexDFT:
  public FixedSizeRealToComplexDFT<T>
{
public:
  FFTW_FixedSizeRealToComplexDFT(unsigned N, unsigned real_stride = 1, unsigned complex_stride = 1,
    bool allow_c2r_to_modify_dft=false, unsigned flags = FFTW_MEASURE):
      FixedSizeRealToComplexDFT<T>(N, real_stride, complex_stride),
      dont_allow_c2r_to_modify_dft_(!allow_c2r_to_modify_dft) {
    T* xt = FixedSizeRealToComplexDFT<T>::alloc_real_array();
    T* xf = FixedSizeRealToComplexDFT<T>::alloc_complex_array();
    int howmany = sizeof(T)/sizeof(double);
    int n = N;
    plan_r2c_ = fftw_plan_many_dft_r2c(1, &n, howmany,
      (double*)xt, nullptr, FixedSizeRealToComplexDFT<T>::rs_*howmany, 1,
      (fftw_complex*)xf, nullptr, FixedSizeRealToComplexDFT<T>::cs_*howmany, 1,
      flags);
    plan_c2r_ = fftw_plan_many_dft_c2r(1, &n, howmany,
      (fftw_complex*)xf, nullptr, FixedSizeRealToComplexDFT<T>::cs_*howmany, 1,
      (double*)xt, nullptr, FixedSizeRealToComplexDFT<T>::rs_*howmany, 1,
      flags);
    free(xf);
    free(xt);
  }
  virtual ~FFTW_FixedSizeRealToComplexDFT() {
    fftw_destroy_plan(plan_r2c_);
    fftw_destroy_plan(plan_c2r_);
  }
  void r2c(T* r_in, T* c_out) override {
    fftw_execute_dft_r2c(plan_r2c_, (double*)r_in, (fftw_complex*)c_out);
    unsigned nc = FixedSizeRealToComplexDFT<T>::num_complex();
    for(unsigned ic=0; ic<nc; ic++) {
      shuffle_complex_s2v(c_out[2*FixedSizeRealToComplexDFT<T>::cs_*ic],
        c_out[2*FixedSizeRealToComplexDFT<T>::cs_*ic+1]);
    }
  }
  void c2r(T* r_out, T* c_in) override {
    unsigned nc = FixedSizeRealToComplexDFT<T>::num_complex();
    for(unsigned ic=0; ic<nc; ic++) {
      shuffle_complex_v2s(c_in[2*FixedSizeRealToComplexDFT<T>::cs_*ic],
        c_in[2*FixedSizeRealToComplexDFT<T>::cs_*ic+1]);
    }
    fftw_execute_dft_c2r(plan_c2r_, (fftw_complex*)c_in, (double*)r_out);
    if(dont_allow_c2r_to_modify_dft_) {
      for(unsigned ic=0; ic<nc; ic++) {
        shuffle_complex_s2v(c_in[2*FixedSizeRealToComplexDFT<T>::cs_*ic],
          c_in[2*FixedSizeRealToComplexDFT<T>::cs_*ic+1]);
      }
    }
  }
protected:
  fftw_plan plan_r2c_;
  fftw_plan plan_c2r_;
  bool dont_allow_c2r_to_modify_dft_;
};

template<typename T> class FFTWF_FixedSizeRealToComplexDFT:
  public FixedSizeRealToComplexDFT<T>
{
public:
  FFTWF_FixedSizeRealToComplexDFT(unsigned N, unsigned real_stride = 1, unsigned complex_stride = 1,
    bool allow_c2r_to_modify_dft=false, unsigned flags = FFTW_MEASURE):
      FixedSizeRealToComplexDFT<T>(N, real_stride, complex_stride),
      dont_allow_c2r_to_modify_dft_(!allow_c2r_to_modify_dft) {
    T* xt = FixedSizeRealToComplexDFT<T>::alloc_real_array();
    T* xf = FixedSizeRealToComplexDFT<T>::alloc_complex_array();
    int howmany = sizeof(T)/sizeof(float);
    int n = N;
    plan_r2c_ = fftwf_plan_many_dft_r2c(1, &n, howmany,
      (float*)xt, nullptr, FixedSizeRealToComplexDFT<T>::rs_*howmany, 1,
      (fftwf_complex*)xf, nullptr, FixedSizeRealToComplexDFT<T>::cs_*howmany, 1,
      flags);
    plan_c2r_ = fftwf_plan_many_dft_c2r(1, &n, howmany,
      (fftwf_complex*)xf, nullptr, FixedSizeRealToComplexDFT<T>::cs_*howmany, 1,
      (float*)xt, nullptr, FixedSizeRealToComplexDFT<T>::rs_*howmany, 1,
      flags);
    free(xf);
    free(xt);
  }
  virtual ~FFTWF_FixedSizeRealToComplexDFT() {
    fftwf_destroy_plan(plan_r2c_);
    fftwf_destroy_plan(plan_c2r_);
  }
  void r2c(T* r_in, T* c_out) override {
    fftwf_execute_dft_r2c(plan_r2c_, (float*)r_in, (fftwf_complex*)c_out);
    unsigned nc = FixedSizeRealToComplexDFT<T>::num_complex();
#if 0
    for(unsigned ic=0; ic<nc; ic++) {
      std::cout << "I(" << ic << "): "
        << c_out[2*FixedSizeRealToComplexDFT<T>::cs_*ic][0] << ' '
        << c_out[2*FixedSizeRealToComplexDFT<T>::cs_*ic][1] << ' '
        << c_out[2*FixedSizeRealToComplexDFT<T>::cs_*ic][2] << ' '
        << c_out[2*FixedSizeRealToComplexDFT<T>::cs_*ic][3] << ' '
        << c_out[2*FixedSizeRealToComplexDFT<T>::cs_*ic+1][0] << ' '
        << c_out[2*FixedSizeRealToComplexDFT<T>::cs_*ic+1][1] << ' '
        << c_out[2*FixedSizeRealToComplexDFT<T>::cs_*ic+1][2] << ' '
        << c_out[2*FixedSizeRealToComplexDFT<T>::cs_*ic+1][3] << '\n';
    }
#endif
    for(unsigned ic=0; ic<nc; ic++) {
      shuffle_complex_s2v(c_out[2*FixedSizeRealToComplexDFT<T>::cs_*ic],
        c_out[2*FixedSizeRealToComplexDFT<T>::cs_*ic+1]);
    }
#if 0
    for(unsigned ic=0; ic<nc; ic++) {
      std::cout << "O(" << ic << "): "
        << c_out[2*FixedSizeRealToComplexDFT<T>::cs_*ic][0] << ' '
        << c_out[2*FixedSizeRealToComplexDFT<T>::cs_*ic][1] << ' '
        << c_out[2*FixedSizeRealToComplexDFT<T>::cs_*ic][2] << ' '
        << c_out[2*FixedSizeRealToComplexDFT<T>::cs_*ic][3] << ' '
        << c_out[2*FixedSizeRealToComplexDFT<T>::cs_*ic+1][0] << ' '
        << c_out[2*FixedSizeRealToComplexDFT<T>::cs_*ic+1][1] << ' '
        << c_out[2*FixedSizeRealToComplexDFT<T>::cs_*ic+1][2] << ' '
        << c_out[2*FixedSizeRealToComplexDFT<T>::cs_*ic+1][3] << '\n';
    }
#endif
  }
  void c2r(T* r_out, T* c_in) override {
    unsigned nc = FixedSizeRealToComplexDFT<T>::num_complex();
    for(unsigned ic=0; ic<nc; ic++) {
      shuffle_complex_v2s(c_in[2*FixedSizeRealToComplexDFT<T>::cs_*ic],
        c_in[2*FixedSizeRealToComplexDFT<T>::cs_*ic+1]);
    }
    fftwf_execute_dft_c2r(plan_c2r_, (fftwf_complex*)c_in, (float*)r_out);
    if(dont_allow_c2r_to_modify_dft_) {
      for(unsigned ic=0; ic<nc; ic++) {
        shuffle_complex_s2v(c_in[2*FixedSizeRealToComplexDFT<T>::cs_*ic],
          c_in[2*FixedSizeRealToComplexDFT<T>::cs_*ic+1]);
      }
    }
  }
protected:
  fftwf_plan plan_r2c_;
  fftwf_plan plan_c2r_;
  bool dont_allow_c2r_to_modify_dft_;
};

#if defined(__AVX__)
FixedSizeRealToComplexDFT<__m256>* new_m256_r2c_dft(unsigned n,
  unsigned real_stride = 1, unsigned complex_stride = 1);
FixedSizeRealToComplexDFT<__m256d>* new_m256d_r2c_dft(unsigned n,
  unsigned real_stride = 1, unsigned complex_stride = 1);

FixedSizeRealToComplexDFT<__m256>* new_m256_codelet_r2c_dft(unsigned n,
  unsigned real_stride = 1, unsigned complex_stride = 1);
FixedSizeRealToComplexDFT<__m256d>* new_m256d_codelet_r2c_dft(unsigned n,
  unsigned real_stride = 1, unsigned complex_stride = 1);

FixedSizeRealToComplexDFT<__m256>* new_m256_fftw_r2c_dft(unsigned n,
  unsigned real_stride = 1, unsigned complex_stride = 1);
FixedSizeRealToComplexDFT<__m256d>* new_m256d_fftw_r2c_dft(unsigned n,
  unsigned real_stride = 1, unsigned complex_stride = 1);
#endif // defined(__AVX__)

std::vector<float> test_m256_r2c_dft(const std::vector<float>& data);
std::vector<float> test_m256_c2r_dft(const std::vector<float>& fft, unsigned n);
std::vector<float> test_fftw_m256_r2c_dft(const std::vector<float>& data);
std::vector<float> test_fftw_m256_c2r_dft(const std::vector<float>& fft, unsigned n);

std::vector<double> test_m256d_r2c_dft(const std::vector<double>& data);
std::vector<double> test_m256d_c2r_dft(const std::vector<double>& fft, unsigned n);
std::vector<double> test_fftw_m256d_r2c_dft(const std::vector<double>& data);
std::vector<double> test_fftw_m256d_c2r_dft(const std::vector<double>& fft, unsigned n);

// ****************************************************************************
// ************************* DFT REAL <-> HALF-COMPLEX ************************
// ****************************************************************************

template<typename T> class FixedSizeRealToHalfComplexDFT
{
public:
  FixedSizeRealToHalfComplexDFT(unsigned N, unsigned real_stride = 1, unsigned half_complex_stride = 1):
    N_(N), rs_(real_stride), hcs_(half_complex_stride) { assert(rs_ != 0); assert(hcs_ != 0); };
  virtual ~FixedSizeRealToHalfComplexDFT() { }
  virtual void r2hc(T* r_in, T* hc_out) = 0;
  virtual void hc2r(T* r_out, T* hc_in) = 0;

  // Utility functions
  int real_stride() const { return rs_; }
  int half_complex_stride() const { return hcs_; }
  unsigned num_real() const { return N_; }
  unsigned num_half_complex() const { return N_; }
  unsigned real_array_size() const { return num_real()*rs_; }
  unsigned half_complex_array_size() const { return num_half_complex()*hcs_; }
  T* alloc_real_array() const {
    return calin::util::memory::aligned_calloc<T>(real_array_size()); }
  T* alloc_half_complex_array() const {
    return calin::util::memory::aligned_calloc<T>(half_complex_array_size()); }
protected:
  unsigned N_;
  unsigned rs_;
  unsigned hcs_;
};

template<typename T> class FFTW_FixedSizeRealToHalfComplexDFT:
  public FixedSizeRealToHalfComplexDFT<T>
{
public:
  FFTW_FixedSizeRealToHalfComplexDFT(unsigned N, unsigned real_stride = 1, unsigned half_complex_stride = 1,
    unsigned flags = FFTW_MEASURE):
      FixedSizeRealToHalfComplexDFT<T>(N, real_stride, half_complex_stride) {
    T* xt = FixedSizeRealToHalfComplexDFT<T>::alloc_real_array();
    T* xf = FixedSizeRealToHalfComplexDFT<T>::alloc_half_complex_array();
    int howmany = sizeof(T)/sizeof(double);
    fftw_r2r_kind kind = FFTW_R2HC;
    int n = N;
    plan_r2hc_ = fftw_plan_many_r2r(1, &n, howmany,
      (double*)xt, nullptr, FixedSizeRealToHalfComplexDFT<T>::rs_*howmany, 1,
      (double*)xf, nullptr, FixedSizeRealToHalfComplexDFT<T>::hcs_*howmany, 1,
      &kind, flags);
    kind = FFTW_HC2R;
    plan_hc2r_ = fftw_plan_many_r2r(1, &n, howmany,
      (double*)xf, nullptr, FixedSizeRealToHalfComplexDFT<T>::hcs_*howmany, 1,
      (double*)xt, nullptr, FixedSizeRealToHalfComplexDFT<T>::rs_*howmany, 1,
      &kind, flags);
    free(xf);
    free(xt);
  }
  virtual ~FFTW_FixedSizeRealToHalfComplexDFT() {
    fftw_destroy_plan(plan_r2hc_);
    fftw_destroy_plan(plan_hc2r_);
  }
  void r2hc(T* r_in, T* hc_out) override {
    fftw_execute_r2r(plan_r2hc_, (double*)r_in, (double*)hc_out);
  }
  void hc2r(T* r_out, T* hc_in) override {
    fftw_execute_r2r(plan_hc2r_, (double*)hc_in, (double*)r_out);
  }
protected:
  fftw_plan plan_r2hc_;
  fftw_plan plan_hc2r_;
};

template<typename T> class FFTWF_FixedSizeRealToHalfComplexDFT:
  public FixedSizeRealToHalfComplexDFT<T>
{
public:
  FFTWF_FixedSizeRealToHalfComplexDFT(unsigned N, unsigned real_stride = 1, unsigned half_complex_stride = 1,
    unsigned flags = FFTW_MEASURE):
      FixedSizeRealToHalfComplexDFT<T>(N, real_stride, half_complex_stride) {
    T* xt = FixedSizeRealToHalfComplexDFT<T>::alloc_real_array();
    T* xf = FixedSizeRealToHalfComplexDFT<T>::alloc_half_complex_array();
    int howmany = sizeof(T)/sizeof(float);
    fftw_r2r_kind kind = FFTW_R2HC;
    int n = N;
    plan_r2hc_ = fftwf_plan_many_r2r(1, &n, howmany,
      (float*)xt, nullptr, FixedSizeRealToHalfComplexDFT<T>::rs_*howmany, 1,
      (float*)xf, nullptr, FixedSizeRealToHalfComplexDFT<T>::hcs_*howmany, 1,
      &kind, flags);
    kind = FFTW_HC2R;
    plan_hc2r_ = fftwf_plan_many_r2r(1, &n, howmany,
      (float*)xf, nullptr, FixedSizeRealToHalfComplexDFT<T>::hcs_*howmany, 1,
      (float*)xt, nullptr, FixedSizeRealToHalfComplexDFT<T>::rs_*howmany, 1,
      &kind, flags);
    free(xf);
    free(xt);
  }
  virtual ~FFTWF_FixedSizeRealToHalfComplexDFT() {
    fftwf_destroy_plan(plan_r2hc_);
    fftwf_destroy_plan(plan_hc2r_);
  }
  void r2hc(T* r_in, T* hc_out) override {
    fftwf_execute_r2r(plan_r2hc_, (float*)r_in, (float*)hc_out);
  }
  void hc2r(T* r_out, T* hc_in) override {
    fftwf_execute_r2r(plan_hc2r_, (float*)hc_in, (float*)r_out);
  }
protected:
  fftwf_plan plan_r2hc_;
  fftwf_plan plan_hc2r_;
};

#if defined(__AVX__)
FixedSizeRealToHalfComplexDFT<__m256>* new_m256_r2hc_dft(unsigned n,
  unsigned real_stride = 1, unsigned half_complex_stride = 1);
FixedSizeRealToHalfComplexDFT<__m256d>* new_m256d_r2hc_dft(unsigned n,
  unsigned real_stride = 1, unsigned half_complex_stride = 1);

FixedSizeRealToHalfComplexDFT<__m256>* new_m256_codelet_r2hc_dft(unsigned n,
  unsigned real_stride = 1, unsigned half_complex_stride = 1);
FixedSizeRealToHalfComplexDFT<__m256d>* new_m256d_codelet_r2hc_dft(unsigned n,
  unsigned real_stride = 1, unsigned half_complex_stride = 1);

FixedSizeRealToHalfComplexDFT<__m256>* new_m256_fftw_r2hc_dft(unsigned n,
  unsigned real_stride = 1, unsigned half_complex_stride = 1);
FixedSizeRealToHalfComplexDFT<__m256d>* new_m256d_fftw_r2hc_dft(unsigned n,
  unsigned real_stride = 1, unsigned half_complex_stride = 1);
#endif // defined(__AVX__)

std::vector<float> test_m256_r2hc_dft(const std::vector<float>& data);
std::vector<float> test_m256_hc2r_dft(const std::vector<float>& fft);
std::vector<float> test_fftw_m256_r2hc_dft(const std::vector<float>& data);
std::vector<float> test_fftw_m256_hc2r_dft(const std::vector<float>& fft);

std::vector<double> test_m256d_r2hc_dft(const std::vector<double>& data);
std::vector<double> test_m256d_hc2r_dft(const std::vector<double>& fft);
std::vector<double> test_fftw_m256d_r2hc_dft(const std::vector<double>& data);
std::vector<double> test_fftw_m256d_hc2r_dft(const std::vector<double>& fft);

// ****************************************************************************
// ************************** OTHER UTILITY FUNCTIONS *************************
// ****************************************************************************

std::vector<unsigned> list_available_m256_codelets();
std::vector<unsigned> list_available_m256d_codelets();


} } } // namespace calin::math::fft_simd
