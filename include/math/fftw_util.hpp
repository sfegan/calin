/*

   calin/math/fftw_util.hpp -- Stephen Fegan -- 2016-03-29

   Utility functions for fftw HC types

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

#pragma once

#include<cmath>
#include<algorithm>

#include<util/log.hpp>
#include<util/vcl.hpp>
#include<math/fftw_util.pb.h>
#include<math/special.hpp>
#include<fftw3.h>

namespace calin { namespace math { namespace fftw_util {

inline unsigned hcvec_num_real(unsigned nsample) {
  // 1 -> 1
  // 2 -> 2
  // 3 -> 2
  // 4 -> 3
  // 5 -> 3
  return std::min(nsample/2+1, nsample); // Min to handle zero case
}

inline unsigned hcvec_num_imag(unsigned nsample) {
  // 1 -> 0
  // 2 -> 0
  // 3 -> 1
  // 4 -> 1
  // 5 -> 2
  return (std::max(nsample,1U)-1)/2; // Max to handle zero case
}

void hcvec_fftfreq(double* ovec, unsigned nsample, double d=1.0,  bool imaginary_negative = false);
void hcvec_fftindex(int* ovec, unsigned nsample, bool imaginary_negative = false);

#ifndef SWIG

template<typename T>
void hcvec_scale_and_multiply(T* ovec, const T* ivec1,
  const T* ivec2, unsigned nsample, T scale = 1.0)
{
  T *ro = ovec;
  T *co = ovec + nsample-1;
  const T *ri1 = ivec1;
  const T *ci1 = ivec1 + nsample-1;
  const T *ri2 = ivec2;
  const T *ci2 = ivec2 + nsample-1;

  (*ro++) = (*ri1++) * (*ri2++) * scale;
  if(ro==ri1 or ro==ri2)
  {
    while(ro < co)
    {
      T vri1 = *ri1++;
      T vci1 = *ci1--;
      T vri2 = *ri2++;
      T vci2 = *ci2--;
      (*ro++) = (vri1*vri2 - vci1*vci2)*scale;
      (*co--) = (vri1*vci2 + vci1*vri2)*scale;
    }
   }
  else
  {
    while(ro < co)
    {
      (*ro++) = ((*ri1)*(*ri2) - (*ci1)*(*ci2)) * scale;
      (*co--) = ((*ri1++)*(*ci2--) + (*ci1--)*(*ri2++)) * scale;
    }
  }
  if(ro==co)(*ro) = (*ri1) * (*ri2) * scale;
}

template<typename VCLReal>
void hcvec_scale_and_multiply_vcl(typename VCLReal::real_t* ovec,
  const typename VCLReal::real_t* ivec1, const typename VCLReal::real_t* ivec2,
  unsigned nsample, typename VCLReal::real_t scale = 1.0)
{
  typename VCLReal::real_t* ro = ovec;
  typename VCLReal::real_t* co = ovec + nsample;
  const typename VCLReal::real_t* ri1 = ivec1;
  const typename VCLReal::real_t* ci1 = ivec1 + nsample;
  const typename VCLReal::real_t* ri2 = ivec2;
  const typename VCLReal::real_t* ci2 = ivec2 + nsample;

  (*ro++) = (*ri1++) * (*ri2++) * scale;

  while(co - ro >= 2*VCLReal::num_real)
  {
    co -= VCLReal::num_real;
    ci1 -= VCLReal::num_real;
    ci2 -= VCLReal::num_real;
    typename VCLReal::real_vt vri1; vri1.load(ri1);
    typename VCLReal::real_vt vci1; vci1.load(ci1);
    typename VCLReal::real_vt vri2; vri2.load(ri2);
    typename VCLReal::real_vt vci2; vci2.load(ci2);

    typename VCLReal::real_vt vro = (vri1*vri2 - calin::util::vcl::reverse(vci1*vci2)) * scale;
    typename VCLReal::real_vt vco = (calin::util::vcl::reverse(vri1)*vci2 + vci1*calin::util::vcl::reverse(vri2)) * scale;

    vro.store(ro);
    vco.store(co);

    ro += VCLReal::num_real;
    ri1 += VCLReal::num_real;
    ri2 += VCLReal::num_real;
  }

  --co;
  --ci1;
  --ci2;
  while(ro < co)
  {
    double vri1 = *ri1++;
    double vci1 = *ci1--;
    double vri2 = *ri2++;
    double vci2 = *ci2--;
    (*ro++) = (vri1*vri2 - vci1*vci2) * scale;
    (*co--) = (vri1*vci2 + vci1*vri2) * scale;
  }
  if(ro==co)(*ro) = (*ri1) * (*ri2) * scale;
}
// NOTE : This function overrides the template for systems with AVX !!!
#if INSTRSET >= 7
void hcvec_scale_and_multiply(double* ovec, const double* ivec1,
  const double* ivec2, unsigned nsample, double scale = 1.0);
#endif

#if 0
template<typename T>
void hcvec_scale_and_multiply_with_stride(T* ovec, int ostride,
  const T* ivec1, int istride1, const T* ivec2, int istride2, unsigned nsample, T scale = 1.0)
{
  T *ro = ovec;
  T *co = ovec + (nsample-1)*ostride;
  const T *ri1 = ivec1;
  const T *ci1 = ivec1 + (nsample-1)*istride1;
  const T *ri2 = ivec2;
  const T *ci2 = ivec2 + (nsample-1)*istride2;

  *ro = *ri1 * *ri2 * scale;
  ro += ostride, ri1 += istride1, ri2 += istride2;
  if(ro==ri1 or ro==ri2)
  {
    while(ro < co)
    {
      T vri1 = *ri1;
      T vci1 = *ci1;
      T vri2 = *ri2;
      T vci2 = *ci2;
      *ro = (vri1*vri2 - vci1*vci2)*scale;
      *co = (vri1*vci2 + vci1*vri2)*scale;
      ro += ostride, ri1 += istride1, ri2 += istride2;
      co -= ostride, ci1 -= istride1, ci2 -= istride2;
    }
   }
  else
  {
    while(ro < co)
    {
      *ro = ((*ri1)*(*ri2) - (*ci1)*(*ci2)) * scale;
      *co = ((*ri1)*(*ci2) + (*ci1)*(*ri2)) * scale;
      ro += ostride, ri1 += istride1, ri2 += istride2;
      co -= ostride, ci1 -= istride1, ci2 -= istride2;
    }
  }
  if(ro==co)(*ro) = (*ri1) * (*ri2) * scale;
}
#endif

template<typename T>
void hcvec_scale_and_multiply_conj(T* ovec, const T* ivec1,
  const T* ivec2_conj, unsigned nsample, T scale = 1.0)
{
  T *ro = ovec;
  T *co = ovec + nsample-1;
  const T *ri1 = ivec1;
  const T *ci1 = ivec1 + nsample-1;
  const T *ri2 = ivec2_conj;
  const T *ci2 = ivec2_conj + nsample-1;

  (*ro++) = (*ri1++) * (*ri2++) * scale;
  if(ro==ri1 or ro==ri2)
  {
    while(ro < co)
    {
      T vri1 = *ri1++;
      T vci1 = *ci1--;
      T vri2 = *ri2++;
      T vci2 = *ci2--;
      (*ro++) = (vri1*vri2 + vci1*vci2)*scale;
      (*co--) = (vci1*vri2 - vri1*vci2)*scale;
    }
   }
  else
  {
    while(ro < co)
    {
      (*ro++) = ((*ri1)*(*ri2) + (*ci1)*(*ci2)) * scale;
      (*co--) = ((*ci1--)*(*ri2++) - (*ri1++)*(*ci2--)) * scale;
    }
  }
  if(ro==co)(*ro) = (*ri1) * (*ri2) * scale;
}

template<typename T>
void hcvec_scale(T* ovec, unsigned nsample,
  T scale = 1.0)
{
  for(unsigned i=0; i<nsample; ++i) {
    ovec[i] *= scale;
  }
}

template<typename T>
void hcvec_add_scaled(T* ovec, const T* ivec, unsigned nsample,
  T scale = 1.0)
{
  for(unsigned i=0; i<nsample; ++i) {
    ovec[i] += ivec[i]*scale;
  }
}

template<typename T>
void hcvec_set_real(T* ovec, T real_value, unsigned nsample)
{
  const unsigned nreal = hcvec_num_real(nsample);
  for(unsigned i=0; i<nreal; ++i) {
    ovec[i] = real_value;
  }
  for(unsigned i=nreal; i<nsample; ++i) {
    ovec[i] = T(0);
  }
}

template<typename T>
void hcvec_add_real(T* ovec, T real_addand, unsigned nsample)
{
  const unsigned nreal = hcvec_num_real(nsample);
  for(unsigned i=0; i<nreal; ++i) {
    ovec[i] += real_addand;
  }
}

template<typename T>
void hcvec_copy_with_scale_and_add_real(T* ovec, const T* ivec,
  T scale, T real_addand, unsigned nsample)
{
  const unsigned nreal = hcvec_num_real(nsample);
  for(unsigned i=0; i<nreal; ++i) {
    ovec[i] = ivec[i] * scale + real_addand;
  }
  for(unsigned i=nreal; i<nsample; ++i) {
    ovec[i] = ivec[i] * scale;
  }
}

template<typename T>
void hcvec_scale_and_add_real(T* ovec, T scale, T real_addand, unsigned nsample)
{
  const unsigned nreal = hcvec_num_real(nsample);
  for(unsigned i=0; i<nreal; ++i) {
    ovec[i] = ovec[i] * scale + real_addand;
  }
  for(unsigned i=nreal; i<nsample; ++i) {
    ovec[i] = ovec[i] * scale;
  }
}

template<typename T>
T hcvec_sum_real(const T* ivec, unsigned nsample)
{
  const unsigned nreal = hcvec_num_real(nsample);
  T sum_real = T(0);
  for(unsigned i=0; i<nreal; ++i) {
    sum_real += ivec[i];
  }
  return sum_real;
}

template<typename T>
T hcvec_avg_real(const T* ivec, unsigned nsample)
{
  const unsigned nreal = hcvec_num_real(nsample);
  T sum_real = T(0);
  for(unsigned i=0; i<nreal; ++i) {
    sum_real += ivec[i];
  }
  return sum_real/double(nreal);
}

template<typename T>
void hcvec_psd_weight(T* ovec, unsigned nsample)
{
  using calin::math::special::SQR;
  T* r = ovec;
  T* c = ovec + nsample;
  *r++ = 1.0;
  c--;
  while(r<c) {
    *r++ = 2.0;
    *c-- = 2.0;
  }
  if(r==c) {
    *r++ = 1.0;
  }
}

template<typename T>
void hcvec_to_psd(T* ovec, const T* ivec, unsigned nsample, T dc_cpt = 0)
{
  using calin::math::special::SQR;
  const T* r = ivec;
  const T* c = ivec + nsample;
  *ovec++ = SQR(*r++ + dc_cpt);
  c--;
  while(r<c) {
    *ovec++ = 2.0*(SQR(*r++) + SQR(*c--));
  }
  if(r==c) {
    *ovec++ = SQR(*r++);
  }
}

template<typename T>
void hcvec_to_psd_no_square(T* ovec, const T* ivec, unsigned nsample)
{
  const T* r = ivec;
  const T* c = ivec + nsample;
  *ovec++ = *r++;
  c--;
  while(r<c) {
    *ovec++ = 2.0*(*r++ + *c--);
  }
  if(r==c) {
    *ovec++ = *r++;
  }
}

template<typename T>
void hcvec_multiply_and_add_real(T* ovec, const T* ivec1,
  const T* ivec2, T real_addand, unsigned nsample)
{
  T *ro = ovec;
  T *co = ovec + nsample-1;
  const T *ri1 = ivec1;
  const T *ci1 = ivec1 + nsample-1;
  const T *ri2 = ivec2;
  const T *ci2 = ivec2 + nsample-1;

  (*ro++) = (*ri1++) * (*ri2++) + real_addand;
  if(ro==ri1 or ro==ri2)
  {
    while(ro < co)
    {
      T vri1 = *ri1++;
      T vci1 = *ci1--;
      T vri2 = *ri2++;
      T vci2 = *ci2--;
      (*ro++) = (vri1*vri2 - vci1*vci2) + real_addand;
      (*co--) = (vri1*vci2 + vci1*vri2);
    }
   }
  else
  {
    while(ro < co)
    {
      (*ro++) = ((*ri1)*(*ri2) - (*ci1)*(*ci2)) + real_addand;
      (*co--) = ((*ri1++)*(*ci2--) + (*ci1--)*(*ri2++));
    }
  }
  if(ro==co)(*ro) = (*ri1) * (*ri2) + real_addand;
}

template<typename VCLReal>
void hcvec_multiply_and_add_real_vcl(typename VCLReal::real_t* ovec,
  const typename VCLReal::real_t* ivec1, const typename VCLReal::real_t* ivec2,
  typename VCLReal::real_t real_addand, unsigned nsample)
{
  typename VCLReal::real_t* ro = ovec;
  typename VCLReal::real_t* co = ovec + nsample;
  const typename VCLReal::real_t* ri1 = ivec1;
  const typename VCLReal::real_t* ci1 = ivec1 + nsample;
  const typename VCLReal::real_t* ri2 = ivec2;
  const typename VCLReal::real_t* ci2 = ivec2 + nsample;

  (*ro++) = (*ri1++) * (*ri2++) + real_addand;

  while(co - ro >= 2*VCLReal::num_real)
  {
    co -= VCLReal::num_real;
    ci1 -= VCLReal::num_real;
    ci2 -= VCLReal::num_real;
    typename VCLReal::real_vt vri1; vri1.load(ri1);
    typename VCLReal::real_vt vci1; vci1.load(ci1);
    typename VCLReal::real_vt vri2; vri2.load(ri2);
    typename VCLReal::real_vt vci2; vci2.load(ci2);

    typename VCLReal::real_vt vro = (vri1*vri2 - calin::util::vcl::reverse(vci1*vci2)) + real_addand;
    typename VCLReal::real_vt vco = (calin::util::vcl::reverse(vri1)*vci2 + vci1*calin::util::vcl::reverse(vri2));

    vro.store(ro);
    vco.store(co);

    ro += VCLReal::num_real;
    ri1 += VCLReal::num_real;
    ri2 += VCLReal::num_real;
  }

  --co;
  --ci1;
  --ci2;
  while(ro < co)
  {
    double vri1 = *ri1++;
    double vci1 = *ci1--;
    double vri2 = *ri2++;
    double vci2 = *ci2--;
    (*ro++) = (vri1*vri2 - vci1*vci2) + real_addand;
    (*co--) = (vri1*vci2 + vci1*vri2);
  }
  if(ro==co)(*ro) = (*ri1) * (*ri2) + real_addand;
}

// NOTE : This function overrides the template for systems with AVX !!!
#if INSTRSET >= 7
void hcvec_multiply_and_add_real(double* ovec, const double* ivec1,
  const double* ivec2, double real_addand, unsigned nsample);
#endif

template<typename T>
void hcvec_polynomial(T* ovec, const T* ivec, const std::vector<T>& p, unsigned nsample)
{
  if(p.empty()) {
    hcvec_set_real(ovec, T(0), nsample);
    return;
  }

  T *ro = ovec;
  T *co = ovec + nsample;
  const T *ri = ivec;
  const T *ci = ivec + nsample;

  auto pi = p.end();
  --pi;
  T vro = *pi;
  T vco;
  T vri = *ri;
  T vci;
  while(pi != p.begin()) {
    --pi;
    vro = vro * vri + (*pi);
  }
  *ro = vro;

  ++ro, ++ri;
  --co, --ci;

  while(ro < co)
  {
    pi = p.end();
    --pi;
    vro = *pi;
    vco = T(0);

    vri = *ri;
    vci = *ci;

    while(pi != p.begin()) {
      --pi;
      T vvro = vro * vri - vco * vci + (*pi);
      vco    = vro * vci + vco * vri;
      vro   = vvro;
    }
    *ro = vro;
    *co = vco;

    ++ro, ++ri;
    --co, --ci;
  }

  if(ro==co) {
    pi = p.end();
    --pi;
    vro = *pi;
    vri = *ri;
    while(pi != p.begin()) {
      --pi;
      vro = vro * vri + (*pi);
    }
    *ro = vro;
  }
}

// More complex version with vectorization and unrolling of vector loop. In this
// version the order of the inner loops are inverted so that each frequency is
// passed through the full polynomial before moving to the next (to improve
// cache performance)
template<typename VCLReal>
void hcvec_polynomial_vcl(typename VCLReal::real_t* ovec,
  const typename VCLReal::real_t* ivec,
  const std::vector<typename VCLReal::real_t>& p, unsigned nsample)
{
  // No user servicable parts inside

  if(p.empty()) {
    hcvec_set_real(ovec, typename VCLReal::real_t(0), nsample);
    return;
  }

  typename VCLReal::real_t* ro = ovec;
  typename VCLReal::real_t* co = ovec + nsample;
  const typename VCLReal::real_t* ri = ivec;
  const typename VCLReal::real_t* ci = ivec + nsample;

  // Evaluate the zero frequency (real-only) component
  auto pi = p.end();
  --pi;
  *ro = *pi;
  while(pi != p.begin()) {
    --pi;
    *ro = (*ro) * (*ri) + (*pi);
  }

  ++ro, ++ri;

  // Evaluate two AVX vectors of real and complex compnents (i.e. 2 * num_real
  // frequencies) using vector types
  while(co - ro >= 4*VCLReal::num_real)
  {
    pi = p.end();
    typename VCLReal::real_vt vpi = *(--pi);

    typename VCLReal::real_vt vro_a = vpi;
    typename VCLReal::real_vt vco_a = typename VCLReal::real_t(0);
    typename VCLReal::real_vt vro_b = vpi;
    typename VCLReal::real_vt vco_b = typename VCLReal::real_t(0);

    typename VCLReal::real_vt vri_a; vri_a.load(ri);
    ri += VCLReal::num_real;
    typename VCLReal::real_vt vri_b; vri_b.load(ri);
    ri += VCLReal::num_real;

    ci -= VCLReal::num_real;
    typename VCLReal::real_vt vci_a; vci_a.load(ci);
    ci -= VCLReal::num_real;
    typename VCLReal::real_vt vci_b; vci_b.load(ci);

    while(pi != p.begin()) {
      vpi = *(--pi);

      typename VCLReal::real_vt vro_t;

      vro_t  = vro_a*vri_a - calin::util::vcl::reverse(vco_a*vci_a) + vpi;
      vco_a  = calin::util::vcl::reverse(vro_a)*vci_a + vco_a*calin::util::vcl::reverse(vri_a);
      vro_a  = vro_t;

      vro_t  = vro_b*vri_b - calin::util::vcl::reverse(vco_b*vci_b) + vpi;
      vco_b  = calin::util::vcl::reverse(vro_b)*vci_b + vco_b*calin::util::vcl::reverse(vri_b);
      vro_b  = vro_t;
    }

    vro_a.store(ro);
    ro += VCLReal::num_real;
    vro_b.store(ro);
    ro += VCLReal::num_real;

    co -= VCLReal::num_real;
    vco_a.store(co);
    co -= VCLReal::num_real;
    vco_b.store(co);
  }

  // Evaluate any remaining real & complex frequencies that don't fit into a vector
  --co, --ci;

  while(ro < co)
  {
    pi = p.end();
    --pi;
    typename VCLReal::real_t vro = *pi;
    typename VCLReal::real_t vco = typename VCLReal::real_t(0);

    typename VCLReal::real_t vri = *ri;
    typename VCLReal::real_t vci = *ci;

    while(pi != p.begin()) {
      --pi;
      typename VCLReal::real_t vrt;
      vrt = vro * vri - vco * vci + (*pi);
      vco = vro * vci + vco * vri;
      vro = vrt;
    }
    *ro = vro;
    *co = vco;

    ++ro, ++ri;
    --co, --ci;
  }

  // For even numbers of frequencies, finish with final real-only component
  if(ro==co) {
    pi = p.end();
    --pi;
    typename VCLReal::real_t vro = *pi;
    typename VCLReal::real_t vri = *ri;
    while(pi != p.begin()) {
      --pi;
      vro = vro * vri + (*pi);
    }
    *ro = vro;
  }
}

template<typename T>
void hcvec_polynomial_old(T* ovec, const T* ivec, const std::vector<T>& p, unsigned nsample)
{
  if(p.empty()) {
    hcvec_set_real(ovec, T(0), nsample);
    return;
  }

  auto pi = p.end();

  --pi;
  hcvec_set_real(ovec, *pi, nsample);

  while(pi != p.begin()) {
    --pi;
    hcvec_multiply_and_add_real(ovec, ovec, ivec, *pi, nsample);
  }
}

// NOTE : This function overrides the template for systems with AVX !!!
#if INSTRSET >= 7
void hcvec_polynomial(double* ovec, const double* ivec,
  const std::vector<double>& p, unsigned nsample);
#endif

// *****************************************************************************
// *****************************************************************************
//
// Multi stage polynomial
//
// *****************************************************************************
// *****************************************************************************

template<typename T>
void hcvec_multi_stage_polynomial(T* ovec, const T* ivec,
  const std::vector<const std::vector<T>*>& stage_p, unsigned nsample)
{
  T *ro = ovec;
  T *co = ovec + nsample;
  const T *ri = ivec;
  const T *ci = ivec + nsample;

  T vri = *ri;
  T vci;

  for(auto p: stage_p)
  {
    auto pi = p->end();
    --pi;
    T vro = *pi;
    while(pi != p->begin()) {
      --pi;
      vro = vro * vri + (*pi);
    }
    vri = vro;
  }
  *ro = vri;

  ++ro, ++ri;
  --co, --ci;

  while(ro < co)
  {
    vri = *ri;
    vci = *ci;

    for(auto p: stage_p)
    {
      auto pi = p->end();
      --pi;

      T vro = *pi;
      T vco = T(0);

      while(pi != p->begin()) {
        --pi;
        T vvro = vro * vri - vco * vci + (*pi);
        vco    = vro * vci + vco * vri;
        vro   = vvro;
      }
      vri = vro;
      vci = vco;
    }
    *ro = vri;
    *co = vci;

    ++ro, ++ri;
    --co, --ci;
  }

  if(ro==co) {
    vri = *ri;
    for(auto p: stage_p)
    {
      auto pi = p->end();
      --pi;
      T vro = *pi;
      while(pi != p->begin()) {
        --pi;
        vro = vro * vri + (*pi);
      }
      vri = vro;
    }
    *ro = vri;
  }
}

// Extremely complex version with vectorization and unrolling of vector loop. In
// this version the order of the inner loops are inverted so that each frequency
// is passed through all the polynomials before moving to the next (to improve
// cache performance)
template<typename VCLReal>
void hcvec_multi_stage_polynomial_vcl(typename VCLReal::real_t* ovec,
  const typename VCLReal::real_t* ivec,
  const std::vector<const std::vector<typename VCLReal::real_t>*>& stage_p, unsigned nsample)
{
  // No user servicable parts inside

  typename VCLReal::real_t* ro = ovec;
  typename VCLReal::real_t* co = ovec + nsample;
  const typename VCLReal::real_t* ri = ivec;
  const typename VCLReal::real_t* ci = ivec + nsample;

  // Evaluate the zero frequency (real-only) component

  typename VCLReal::real_t sri = *ri;
  typename VCLReal::real_t sci;
  for(auto p: stage_p)
  {
    auto pi = p->end();
    --pi;
    typename VCLReal::real_t sro = *pi;
    while(pi != p->begin()) {
      --pi;
      sro = sro * sri + (*pi);
    }
    sri = sro;
  }
  *ro = sri;
  ++ro, ++ri;

  // Evaluate three AVX vectors of real and complex compnents (i.e. 3 * num_real
  // frequencies) using vector types
  while(co - ro >= 6*VCLReal::num_real)
  {
    typename VCLReal::real_vt vri_a; vri_a.load(ri);
    ri += VCLReal::num_real;
    typename VCLReal::real_vt vri_b; vri_b.load(ri);
    ri += VCLReal::num_real;
    typename VCLReal::real_vt vri_c; vri_c.load(ri);
    ri += VCLReal::num_real;

    ci -= VCLReal::num_real;
    typename VCLReal::real_vt vci_a; vci_a.load(ci);
    ci -= VCLReal::num_real;
    typename VCLReal::real_vt vci_b; vci_b.load(ci);
    ci -= VCLReal::num_real;
    typename VCLReal::real_vt vci_c; vci_c.load(ci);

    for(auto p: stage_p)
    {
      auto pi = p->end();

      typename VCLReal::real_vt vpi = *(--pi);

      typename VCLReal::real_vt vro_a = vpi;
      typename VCLReal::real_vt vco_a = typename VCLReal::real_t(0);
      typename VCLReal::real_vt vro_b = vpi;
      typename VCLReal::real_vt vco_b = typename VCLReal::real_t(0);
      typename VCLReal::real_vt vro_c = vpi;
      typename VCLReal::real_vt vco_c = typename VCLReal::real_t(0);

      while(pi != p->begin()) {
        vpi = *(--pi);

        typename VCLReal::real_vt vro_t;

        vro_t  = vro_a*vri_a - calin::util::vcl::reverse(vco_a*vci_a) + vpi;
        vco_a  = calin::util::vcl::reverse(vro_a)*vci_a + vco_a*calin::util::vcl::reverse(vri_a);
        vro_a  = vro_t;

        vro_t  = vro_b*vri_b - calin::util::vcl::reverse(vco_b*vci_b) + vpi;
        vco_b  = calin::util::vcl::reverse(vro_b)*vci_b + vco_b*calin::util::vcl::reverse(vri_b);
        vro_b  = vro_t;

        vro_t  = vro_c*vri_c - calin::util::vcl::reverse(vco_c*vci_c) + vpi;
        vco_c  = calin::util::vcl::reverse(vro_c)*vci_c + vco_c*calin::util::vcl::reverse(vri_c);
        vro_c  = vro_t;
      }

      vri_a = vro_a;
      vci_a = vco_a;
      vri_b = vro_b;
      vci_b = vco_b;
      vri_c = vro_c;
      vci_c = vco_c;
    }

    vri_a.store(ro);
    ro += VCLReal::num_real;
    vri_b.store(ro);
    ro += VCLReal::num_real;
    vri_c.store(ro);
    ro += VCLReal::num_real;

    co -= VCLReal::num_real;
    vci_a.store(co);
    co -= VCLReal::num_real;
    vci_b.store(co);
    co -= VCLReal::num_real;
    vci_c.store(co);
  }

  // Evaluate any remaining real & complex frequencies that don't fit into a vector
  --co, --ci;

  while(ro < co)
  {
    sri = *ri;
    sci = *ci;

    for(auto p: stage_p)
    {
      auto pi = p->end();
      --pi;

      typename VCLReal::real_t sro = *pi;
      typename VCLReal::real_t sco = 0;

      while(pi != p->begin()) {
        --pi;
        typename VCLReal::real_t sro_t = sro * sri - sco * sci + (*pi);
        sco     = sro * sci + sco * sri;
        sro     = sro_t;
      }
      sri = sro;
      sci = sco;
    }
    *ro = sri;
    *co = sci;

    ++ro, ++ri;
    --co, --ci;
  }

  // For even numbers of frequencies, finish with final real-only component
  if(ro==co) {
    sri = *ri;
    for(auto p: stage_p)
    {
      auto pi = p->end();
      --pi;
      typename VCLReal::real_t sro = *pi;
      while(pi != p->begin()) {
        --pi;
        sro = sro * sri + (*pi);
      }
      sri = sro;
    }
    *ro = sri;
  }
}

// NOTE : This function overrides the template for systems with AVX !!!
#if INSTRSET >= 7
void hcvec_multi_stage_polynomial(double* ovec, double* ivec,
  const std::vector<const std::vector<double>*>& stage_p, unsigned nsample);
#endif

// *****************************************************************************
// *****************************************************************************
//
// Analytic DFT of Gaussian
//
// *****************************************************************************
// *****************************************************************************

template<typename T>
void hcvec_gaussian_dft(T* ovec, T mean, T sigma, unsigned nsample)
{
  T *ro = ovec;
  T *co = ovec + nsample-1;

  T nsample_inv = 1.0/T(nsample);
  T scale = 2.0*calin::math::special::SQR(M_PI*sigma*nsample_inv);
  T phase = 2*M_PI*mean*nsample_inv;

  (*ro++) = 1.0;

  while(ro < co)
  {
    T x = T(ro-ovec);
    T amp = std::exp(-calin::math::special::SQR(x) * scale);
    x *= phase;
    T c = std::cos(x);
    T s = std::sin(x);
    (*ro++) = amp*c;
    (*co--) = -amp*s;
  }

  if(ro==co) {
    T x = T(ro-ovec);
    T amp = std::exp(-calin::math::special::SQR(x) * scale);
    x *= phase;
    T c = std::cos(x);
    (*ro++) = amp*c;
  }
}

template<typename VCLReal>
void hcvec_gaussian_dft_vcl(typename VCLReal::real_t* ovec,
  typename VCLReal::real_t mean, typename VCLReal::real_t sigma, unsigned nsample)
{
  // No user servicable parts inside

  typename VCLReal::real_t* ro = ovec;
  typename VCLReal::real_t* co = ovec + nsample;

  typename VCLReal::real_t nsample_inv = 1.0/typename VCLReal::real_t(nsample);
  typename VCLReal::real_t scale = 2.0*calin::math::special::SQR(M_PI*sigma*nsample_inv);
  typename VCLReal::real_t phase = -2*M_PI*mean*nsample_inv;

  // Evaluate the zero frequency (real-only) component
  (*ro++) = 1.0;

  // Evaluate AVX vectors of real and complex compnents (i.e. num_real
  // frequencies) using vector types
  typename VCLReal::real_vt x = VCLReal::iota() + 1.0;
  while(co - ro >= 2*VCLReal::num_real)
  {
    typename VCLReal::real_vt amp = vcl::exp(-x*x*scale);
    typename VCLReal::real_vt c;
    typename VCLReal::real_vt s = vcl::sincos(&c, x*phase);

    c *= amp;
    s *= amp;

    s = calin::util::vcl::reverse(s);

    x += VCLReal::num_real;

    c.store(ro);
    ro += VCLReal::num_real;

    co -= VCLReal::num_real;
    s.store(co);
  }

  // Evaluate any remaining real & complex frequencies that don't fit into a vector
  --co;

  while(ro < co)
  {
    typename VCLReal::real_t x = typename VCLReal::real_t(ro-ovec);
    typename VCLReal::real_t amp = std::exp(-calin::math::special::SQR(x) * scale);
    x *= phase;
    typename VCLReal::real_t c = std::cos(x);
    typename VCLReal::real_t s = std::sin(x);
    (*ro++) = amp*c;
    (*co--) = amp*s;
  }

  if(ro==co) {
    typename VCLReal::real_t x = double(ro-ovec);
    typename VCLReal::real_t amp = std::exp(-calin::math::special::SQR(x) * scale);
    x *= phase;
    typename VCLReal::real_t c = std::cos(x);
    (*ro++) = amp*c;
  }
}

// NOTE : This function overrides the template for systems with AVX !!!
#if INSTRSET >= 7
void hcvec_gaussian_dft(double* ovec, double mean, double sigma, unsigned nsample);
#endif

// *****************************************************************************
// *****************************************************************************
//
// Analytic DFT of Double-Gaussian
//
// *****************************************************************************
// *****************************************************************************

template<typename T>
void hcvec_2gaussian_dft(T* ovec, T mean, T sigma, T split, unsigned nsample)
{
  T *ro = ovec;
  T *co = ovec + nsample-1;

  T nsample_inv = 1.0/T(nsample);
  T scale = 2.0*calin::math::special::SQR(M_PI*sigma*nsample_inv);
  T phase1 = -2*M_PI*(mean-0.5*split)*nsample_inv;
  T phase2 = -2*M_PI*(mean+0.5*split)*nsample_inv;

  (*ro++) = 1.0;

  while(ro < co)
  {
    T x = T(ro-ovec);
    T amp = 0.5*std::exp(-calin::math::special::SQR(x) * scale);
    T x1 = x*phase1;
    T c = std::cos(x1);
    T s = std::sin(x1);
    T x2 = x*phase2;
    c += std::cos(x2);
    s += std::sin(x2);
    (*ro++) = amp*c;
    (*co--) = amp*s;
  }

  if(ro==co) {
    T x = T(ro-ovec);
    T amp = 0.5*std::exp(-calin::math::special::SQR(x) * scale);
    T x1 = x*phase1;
    T c = std::cos(x1);
    T x2 = x*phase2;
    c += std::cos(x2);
    (*ro++) = amp*c;
  }
}

template<typename VCLReal>
void hcvec_2gaussian_dft_vcl(typename VCLReal::real_t* ovec,
  typename VCLReal::real_t mean, typename VCLReal::real_t sigma, typename VCLReal::real_t split, unsigned nsample)
{
  // No user servicable parts inside

  typename VCLReal::real_t* ro = ovec;
  typename VCLReal::real_t* co = ovec + nsample;

  typename VCLReal::real_t nsample_inv = 1.0/typename VCLReal::real_t(nsample);
  typename VCLReal::real_t scale = 2.0*calin::math::special::SQR(M_PI*sigma*nsample_inv);
  typename VCLReal::real_t phase1 = -2*M_PI*(mean-0.5*split)*nsample_inv;
  typename VCLReal::real_t phase2 = -2*M_PI*(mean+0.5*split)*nsample_inv;

  // Evaluate the zero frequency (real-only) component
  (*ro++) = 1.0;

  // Evaluate AVX vectors of real and complex compnents (i.e. num_real
  // frequencies) using vector types
  typename VCLReal::real_vt x = VCLReal::iota() + 1.0;
  while(co - ro >= 2*VCLReal::num_real)
  {
    typename VCLReal::real_vt amp = 0.5*vcl::exp(-x*x*scale);
    typename VCLReal::real_vt c1;
    typename VCLReal::real_vt s1 = vcl::sincos(&c1, x*phase1);
    typename VCLReal::real_vt c2;
    typename VCLReal::real_vt s2 = vcl::sincos(&c2, x*phase2);

    c1 = amp*(c1 + c2);
    s1 = amp*(s1 + s2);
    s1 = calin::util::vcl::reverse(s1);

    x += VCLReal::num_real;

    c1.store(ro);
    ro += VCLReal::num_real;

    co -= VCLReal::num_real;
    s1.store(co);
  }

  // Evaluate any remaining real & complex frequencies that don't fit into a vector
  --co;

  while(ro < co)
  {
    typename VCLReal::real_t x = typename VCLReal::real_t(ro-ovec);
    typename VCLReal::real_t amp = 0.5*std::exp(-calin::math::special::SQR(x) * scale);
    typename VCLReal::real_t x1 = x*phase1;
    typename VCLReal::real_t c = std::cos(x1);
    typename VCLReal::real_t s = std::sin(x1);
    typename VCLReal::real_t x2 = x*phase2;
    c += std::cos(x2);
    s += std::sin(x2);
    (*ro++) = amp*c;
    (*co--) = amp*s;
  }

  if(ro==co) {
    typename VCLReal::real_t x = typename VCLReal::real_t(ro-ovec);
    typename VCLReal::real_t amp = 0.5*std::exp(-calin::math::special::SQR(x) * scale);
    typename VCLReal::real_t x1 = x*phase1;
    typename VCLReal::real_t c = std::cos(x1);
    typename VCLReal::real_t x2 = x*phase2;
    c += std::cos(x2);
    (*ro++) = amp*c;
  }
}

// NOTE : This function overrides the template for systems with AVX !!!
#if INSTRSET >= 7
void hcvec_2gaussian_dft(double* ovec, double mean, double sigma, double split, unsigned nsample);
#endif

// *****************************************************************************
// *****************************************************************************
//
// Analytic DFT of Delta(x-x0)
//
// *****************************************************************************
// *****************************************************************************

template<typename T>
void hcvec_delta_dft(T* ovec, T x0, unsigned nsample)
{
  T *ro = ovec;
  T *co = ovec + nsample-1;

  T nsample_inv = 1.0/T(nsample);
  T phase = 2*M_PI*x0*nsample_inv;

  (*ro++) = 1.0;

  while(ro < co)
  {
    T x = T(ro-ovec) * phase;
    T c = std::cos(x);
    T s = std::sin(x);
    (*ro++) = c;
    (*co--) = -s;
  }

  if(ro==co) {
    T x = T(ro-ovec) * phase;
    T c = std::cos(x);
    (*ro++) = c;
  }
}

template<typename VCLReal>
void hcvec_delta_dft_vcl(typename VCLReal::real_t* ovec, typename VCLReal::real_t x0, unsigned nsample)
{
  // No user servicable parts inside

  typename VCLReal::real_t* ro = ovec;
  typename VCLReal::real_t* co = ovec + nsample;

  typename VCLReal::real_t nsample_inv = 1.0/typename VCLReal::real_t(nsample);
  typename VCLReal::real_t phase = 2*M_PI*x0*nsample_inv;

  // Evaluate the zero frequency (real-only) component
  (*ro++) = 1.0;

  // Evaluate three AVX vectors of real and complex compnents (i.e. 3 * num_real
  // frequencies) using vector types
  typename VCLReal::real_vt x = VCLReal::iota() + 1.0;
  while(co - ro >= 2*VCLReal::num_real)
  {
    typename VCLReal::real_vt c;
    typename VCLReal::real_vt s = vcl::sincos(&c, x*phase);

    s = -calin::util::vcl::reverse(s);
    x += VCLReal::num_real;

    c.store(ro);
    ro += VCLReal::num_real;

    co -= VCLReal::num_real;
    s.store(co);
  }

  // Evaluate any remaining real & complex frequencies that don't fit into a vector
  --co;

  while(ro < co)
  {
    typename VCLReal::real_t x = typename VCLReal::real_t(ro-ovec) * phase;
    typename VCLReal::real_t c = std::cos(x);
    typename VCLReal::real_t s = std::sin(x);
    (*ro++) = c;
    (*co--) = -s;
  }

  if(ro==co) {
    typename VCLReal::real_t x = typename VCLReal::real_t(ro-ovec) * phase;
    typename VCLReal::real_t c = std::cos(x);
    (*ro++) = c;
  }
}

// NOTE : This function overrides the template for systems with AVX !!!
#if INSTRSET >= 7
void hcvec_delta_dft(double* ovec, double x0, unsigned nsample);
#endif

// *****************************************************************************
// *****************************************************************************
//
// Analytic Inverse DFT of Delta(k-k0)
//
// *****************************************************************************
// *****************************************************************************

template<typename T>
void hcvec_delta_idft(T* ovec, T k0, T phase0, unsigned nsample)
{
  T *ro = ovec;
  T *co = ovec + nsample;

  T nsample_inv = 1.0/T(nsample);
  T phase = 2*M_PI*k0*nsample_inv;

  bool unity_amplitude = (k0==0) || (fabs(k0) == T(nsample)/2);
  const T amp = unity_amplitude ? 1.0 : 2.0;

  while(ro < co)
  {
    T k = T(ro-ovec) * phase + phase0;
    T c = std::cos(k);
    (*ro++) = amp*c;
  }
}

template<typename VCLReal>
void hcvec_delta_idft_vcl(typename VCLReal::real_t* ovec, typename VCLReal::real_t k0,
  typename VCLReal::real_t phase0, unsigned nsample)
{
  // No user servicable parts inside

  typename VCLReal::real_t* ro = ovec;
  typename VCLReal::real_t* co = ovec + nsample;

  typename VCLReal::real_t nsample_inv = 1.0/typename VCLReal::real_t(nsample);
  typename VCLReal::real_t phase = 2*M_PI*k0*nsample_inv;

  bool unity_amplitude = (k0==0) || (fabs(k0) == typename VCLReal::real_t(nsample)/2);
  typename VCLReal::real_t amp = unity_amplitude ? 1.0 : 2.0;

  // Evaluate one AVX vectors of points (i.e. num_real points) using vector types
  typename VCLReal::real_vt k = VCLReal::iota();
  while(co - ro >= VCLReal::num_real)
  {
    typename VCLReal::real_vt c;
    vcl::sincos(&c, k*phase + phase0);
    c *= amp;

    k += VCLReal::num_real;

    c.store(ro);
    ro += VCLReal::num_real;
  }

  // Evaluate any remaining points that don't fit into a vector
  while(ro < co)
  {
    typename VCLReal::real_t k = typename VCLReal::real_t(ro-ovec) * phase + phase0;
    typename VCLReal::real_t c = std::cos(k);
    (*ro++) = amp*c;
  }
}

template<typename T>
void hcvec_delta_iq_idft(T* oivec, T* oqvec, T k0, T phase0, unsigned nsample)
{
  T nsample_inv = 1.0/T(nsample);
  T phase = 2*M_PI*k0*nsample_inv;

  bool unity_amplitude = (k0==0) || (fabs(k0) == T(nsample)/2);
  T iamp = unity_amplitude ? 1.0 : 2.0;
  T qamp = unity_amplitude ? 0.0 : -2.0;

  for(unsigned ik=0; ik<nsample; ++ik)
  {
    T k = T(ik) * phase + phase0;
    T c = std::cos(k);
    T s = std::sin(k);
    (*oivec++) = iamp*c;
    (*oqvec++) = qamp*s;
  }
}

template<typename VCLReal>
void hcvec_delta_iq_idft_vcl(
  typename VCLReal::real_t* oivec, typename VCLReal::real_t* oqvec,
  typename VCLReal::real_t k0, typename VCLReal::real_t phase0, unsigned nsample)
{
  // No user servicable parts inside

  typename VCLReal::real_t nsample_inv = 1.0/typename VCLReal::real_t(nsample);
  typename VCLReal::real_t phase = 2*M_PI*k0*nsample_inv;

  bool unity_amplitude = (k0==0) || (fabs(k0) == typename VCLReal::real_t(nsample)/2);
  typename VCLReal::real_t iamp = unity_amplitude ? 1.0 : 2.0;
  typename VCLReal::real_t qamp = unity_amplitude ? 0.0 : -2.0;

  // Evaluate one AVX vectors of points (i.e. num_real points) using vector types
  typename VCLReal::real_vt ikv = VCLReal::iota();
  unsigned ik = 0;
  while(nsample - ik >= VCLReal::num_real)
  {
    typename VCLReal::real_vt c;
    typename VCLReal::real_vt s;
    s = vcl::sincos(&c, ikv*phase + phase0);
    c *= iamp;
    s *= qamp;

    ikv += VCLReal::num_real;
    ik += VCLReal::num_real;

    c.store(oivec);
    oivec += VCLReal::num_real;
    s.store(oqvec);
    oqvec += VCLReal::num_real;
  }

  // Evaluate any remaining points that don't fit into a vector
  while(ik < nsample)
  {
    typename VCLReal::real_t k = typename VCLReal::real_t(ik++) * phase + phase0;
    typename VCLReal::real_t c = std::cos(k);
    typename VCLReal::real_t s = std::sin(k);
    (*oivec++) = iamp*c;
    (*oqvec++) = qamp*s;
  }
}

// NOTE : This function overrides the template for systems with AVX !!!
#if INSTRSET >= 7
void hcvec_delta_idft(double* ovec, double k0, double phase0, unsigned nsample);
void hcvec_delta_iq_idft(double* oivec, double* oqvec, double k0, double phase0, unsigned nsample);
#endif

// *****************************************************************************
// *****************************************************************************
//
// FFTW codelet aceess
//
// *****************************************************************************
// *****************************************************************************

template<typename VCLReal> class alignas(VCLReal::vec_bytes) FFTWCodelet
{
public:
  using float_type        = typename VCLReal::real_vt;

  bool has_codelet(unsigned size) {
    switch(size) {
#define ADD_CASE(n) case n : return(true)
    ADD_CASE(8);
    ADD_CASE(12);
    ADD_CASE(15);
    ADD_CASE(16);
    ADD_CASE(18);
    ADD_CASE(20);
    ADD_CASE(24);
    ADD_CASE(28);
    ADD_CASE(30);
    ADD_CASE(32);
    ADD_CASE(36);
    ADD_CASE(40);
    ADD_CASE(48);
    ADD_CASE(56);
    ADD_CASE(60);
    ADD_CASE(64);
#undef ADD_CASE
    default:
      return false;
    }
  }

  void r2hc(unsigned size, const float_type* r, float_type* c) {
    switch(size) {
#define ADD_CASE(n) case n : return r2hc_##n(r,c)
    ADD_CASE(8);
    ADD_CASE(12);
    ADD_CASE(15);
    ADD_CASE(16);
    ADD_CASE(18);
    ADD_CASE(20);
    ADD_CASE(24);
    ADD_CASE(28);
    ADD_CASE(30);
    ADD_CASE(32);
    ADD_CASE(36);
    ADD_CASE(40);
    ADD_CASE(48);
    ADD_CASE(56);
    ADD_CASE(60);
    ADD_CASE(64);
#undef ADD_CASE
    default:
      throw std::runtime_error("No FFT codelet available for array size : " + std::to_string(size));
    }
  }

  void hc2r(unsigned size, float_type* r, const float_type* c) {
    switch(size) {
#define ADD_CASE(n) case n : return hc2r_##n(r,c)
    ADD_CASE(8);
    ADD_CASE(12);
    ADD_CASE(15);
    ADD_CASE(16);
    ADD_CASE(18);
    ADD_CASE(20);
    ADD_CASE(24);
    ADD_CASE(28);
    ADD_CASE(30);
    ADD_CASE(32);
    ADD_CASE(36);
    ADD_CASE(40);
    ADD_CASE(48);
    ADD_CASE(56);
    ADD_CASE(60);
    ADD_CASE(64);
#undef ADD_CASE
    default:
      throw std::runtime_error("No FFT codelet available for array size : " + std::to_string(size));
    }
  }

#define MAKE_R2HC(n) \
  void r2hc_##n(const float_type* r, float_type* c) { \
    dft_codelet_r2cf_##n(const_cast<float_type*>(r), const_cast<float_type*>(r+1), c, c+n, 2, 1, -1, 1, 0, 0); \
  }

#define MAKE_HC2R(n) \
  void hc2r_##n(float_type* r, const float_type* c) { \
    dft_codelet_r2cb_##n(r, r+1, const_cast<float_type*>(c), const_cast<float_type*>(c+n), 2, 1, -1, 1, 0, 0); \
  }

  MAKE_R2HC(8);
  MAKE_HC2R(8);
  MAKE_R2HC(12);
  MAKE_HC2R(12);
  MAKE_R2HC(15);
  MAKE_HC2R(15);
  MAKE_R2HC(16);
  MAKE_HC2R(16);
  MAKE_R2HC(18);
  MAKE_HC2R(18);
  MAKE_R2HC(20);
  MAKE_HC2R(20);
  MAKE_R2HC(24);
  MAKE_HC2R(24);
  MAKE_R2HC(28);
  MAKE_HC2R(28);
  MAKE_R2HC(30);
  MAKE_HC2R(30);
  MAKE_R2HC(32);
  MAKE_HC2R(32);
  MAKE_R2HC(36);
  MAKE_HC2R(36);
  MAKE_R2HC(40);
  MAKE_HC2R(40);
  MAKE_R2HC(48);
  MAKE_HC2R(48);
  MAKE_R2HC(56);
  MAKE_HC2R(56);
  MAKE_R2HC(60);
  MAKE_HC2R(60);
  MAKE_R2HC(64);
  MAKE_HC2R(64);

private:
  using E                 = typename VCLReal::real_vt;
  using R                 = typename VCLReal::real_vt;
  using INT               = int;
  using stride            = int;

  inline int WS(const stride s, const stride i) { return s*i; }

  inline E ADD(const E& a, const E& b) { return a+b; }
  inline E SUB(const E& a, const E& b) { return a-b; }
  inline E MUL(const E& a, const E& b) { return a*b; }
  inline E NEG(const E& a) { return -a; }

  inline E FMA(const E& a, const E& b, const E& c) { return vcl::mul_add(a,b,c); }
  inline E FMS(const E& a, const E& b, const E& c) { return vcl::mul_sub(a,b,c); }
  // Note: inconsistency between FFTW and Intel intrinsics definitions of FNMA/S
  inline E FNMA(const E& a, const E& b, const E& c) { return -vcl::mul_add(a,b,c); }
  inline E FNMS(const E& a, const E& b, const E& c) { return vcl::nmul_add(a,b,c); }

  inline void MAKE_VOLATILE_STRIDE(int a, int b) { }

#define DK(name, val) const E name = val;

  inline E ZERO() { return 0; }

#include "../../src/math/genfft_codelets/dft_r2cf_8.c"
#include "../../src/math/genfft_codelets/dft_r2cb_8.c"
#include "../../src/math/genfft_codelets/dft_r2cf_12.c"
#include "../../src/math/genfft_codelets/dft_r2cb_12.c"
#include "../../src/math/genfft_codelets/dft_r2cf_15.c"
#include "../../src/math/genfft_codelets/dft_r2cb_15.c"
#include "../../src/math/genfft_codelets/dft_r2cf_16.c"
#include "../../src/math/genfft_codelets/dft_r2cb_16.c"
#include "../../src/math/genfft_codelets/dft_r2cf_18.c"
#include "../../src/math/genfft_codelets/dft_r2cb_18.c"
#include "../../src/math/genfft_codelets/dft_r2cf_20.c"
#include "../../src/math/genfft_codelets/dft_r2cb_20.c"
#include "../../src/math/genfft_codelets/dft_r2cf_24.c"
#include "../../src/math/genfft_codelets/dft_r2cb_24.c"
#include "../../src/math/genfft_codelets/dft_r2cf_28.c"
#include "../../src/math/genfft_codelets/dft_r2cb_28.c"
#include "../../src/math/genfft_codelets/dft_r2cf_30.c"
#include "../../src/math/genfft_codelets/dft_r2cb_30.c"
#include "../../src/math/genfft_codelets/dft_r2cf_32.c"
#include "../../src/math/genfft_codelets/dft_r2cb_32.c"
#include "../../src/math/genfft_codelets/dft_r2cf_36.c"
#include "../../src/math/genfft_codelets/dft_r2cb_36.c"
#include "../../src/math/genfft_codelets/dft_r2cf_40.c"
#include "../../src/math/genfft_codelets/dft_r2cb_40.c"
#include "../../src/math/genfft_codelets/dft_r2cf_48.c"
#include "../../src/math/genfft_codelets/dft_r2cb_48.c"
#include "../../src/math/genfft_codelets/dft_r2cf_56.c"
#include "../../src/math/genfft_codelets/dft_r2cb_56.c"
#include "../../src/math/genfft_codelets/dft_r2cf_60.c"
#include "../../src/math/genfft_codelets/dft_r2cb_60.c"
#include "../../src/math/genfft_codelets/dft_r2cf_64.c"
#include "../../src/math/genfft_codelets/dft_r2cb_64.c"

};

#undef DK
#undef MAKE_R2HC
#undef MAKE_HC2R

// *****************************************************************************
// *****************************************************************************
//
// Various other functions and SWIG definitions
//
// *****************************************************************************
// *****************************************************************************

using uptr_fftw_plan = std::unique_ptr<fftw_plan_s,void(*)(fftw_plan_s*)>;
using uptr_fftw_data = std::unique_ptr<double,void(*)(void*)>;

#endif // defined SWIG

// Expose some of these functions in more SWIG friendly way
Eigen::VectorXd hcvec_fftfreq(unsigned nsample, double d=1.0,  bool imaginary_negative = false);
Eigen::VectorXi hcvec_fftindex(unsigned nsample, bool imaginary_negative = false);
double hcvec_sum_real(const Eigen::VectorXd& ivec);
double hcvec_avg_real(const Eigen::VectorXd& ivec);
Eigen::VectorXd hcvec_multiply_and_add_real(const Eigen::VectorXd& ivec1, const Eigen::VectorXd& ivec2, double real_addand);
Eigen::VectorXd hcvec_scale_and_add_real(const Eigen::VectorXd& ivec, double scale, double real_addand);
Eigen::VectorXd hcvec_gaussian_dft(double mean, double sigma, unsigned nsample, bool vcl = true);
Eigen::VectorXd hcvec_2gaussian_dft(double mean, double sigma, double split, unsigned nsample, bool vcl = true);
Eigen::VectorXd hcvec_delta_dft(double x0, unsigned nsample, bool vcl = true);
Eigen::VectorXd hcvec_delta_idft(double k0, double phase0, unsigned nsample, bool vcl = true);
Eigen::VectorXd hcvec_delta_idft_by_index(unsigned index, unsigned nsample, bool vcl = true);

void hcvec_delta_iq_idft(Eigen::VectorXd& oivec, Eigen::VectorXd& oqvec,
  double k0, double phase0, unsigned nsample, bool vcl = true);
void hcvec_delta_iq_idft_by_index(Eigen::VectorXd& oivec, Eigen::VectorXd& oqvec,
  unsigned index, unsigned nsample, bool vcl = true);

Eigen::VectorXd hcvec_psd_weight(unsigned nsample);
Eigen::VectorXd hcvec_to_psd(const Eigen::VectorXd& ivec, double dc_cpt=0);
Eigen::VectorXd hcvec_to_psd_no_square(const Eigen::VectorXd& ivec);

Eigen::VectorXd fftw_r2hc(const Eigen::VectorXd& x,
  calin::ix::math::fftw_util::FFTWPlanningRigor fftw_rigor = calin::ix::math::fftw_util::ESTIMATE);
Eigen::VectorXd fftw_hc2r(const Eigen::VectorXd& f,
  calin::ix::math::fftw_util::FFTWPlanningRigor fftw_rigor = calin::ix::math::fftw_util::ESTIMATE);

int proto_planning_enum_to_fftw_flag(calin::ix::math::fftw_util::FFTWPlanningRigor x);

bool load_wisdom_from_file(std::string filename = "~/.calin_fft_wisdom");
bool load_wisdom_from_proto(const calin::ix::math::fftw_util::FFTWWisdom& proto);

bool save_wisdom_to_file(std::string filename = "~/.calin_fft_wisdom");
bool save_wisdom_to_proto(calin::ix::math::fftw_util::FFTWWisdom& proto);

Eigen::VectorXd fftw_codelet_r2hc(const Eigen::VectorXd& x);
Eigen::VectorXd fftw_codelet_hc2r(const Eigen::VectorXd& f);
Eigen::VectorXd fftw_codelet_r2hc_float(const Eigen::VectorXd& x);
Eigen::VectorXd fftw_codelet_hc2r_float(const Eigen::VectorXd& f);

} } } // namespace calin::math::fftw_util
