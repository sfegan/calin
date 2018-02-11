/*

   calin/unit_tests/math/test_simd.cpp -- Stephen Fegan -- 2018-01-05

   Unit tests for SIMD

   Copyright 2018, Stephen Fegan <sfegan@llr.in2p3.fr>
   LLR, Ecole Polytechnique, CNRS/IN2P3

   This file is part of "calin"

   "calin" is free software: you can redistribute it and/or modify it
   under the terms of the GNU General Public License version 2 or
   later, as published by the Free Software Foundation.

   "calin" is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.

*/

#include <iostream>
#include <iomanip>
#include <gtest/gtest.h>
#include <vector>
#include <tuple>
#include <cmath>

#include <math/rng.hpp>
#include <math/simd.hpp>
#include <util/memory.hpp>

using calin::math::rng::NR3_AVX2_RNGCore;
using calin::math::rng::NR3RNGCore;
using namespace calin::math::simd;

static constexpr uint64_t NSIM_RANDSINCOS = 100000000ULL;
static constexpr unsigned NSIM_TRACEANAL = 10000;
static constexpr unsigned NSIM_TRACECOV = 1024;

TEST(TestSIMD, SpeedTest100M_Random64SinCos32)
{
  float sum_s = 0;
  float sum_c = 0;
  uint64_t seed = calin::math::rng::RNG::uint64_from_random_device();
  NR3RNGCore core(seed);
  for(unsigned i=0;i<NSIM_RANDSINCOS/2;i++) {
    uint64_t x64 = core.uniform_uint64();
    float x = 2.328306437e-10 * 2.0 * M_PI * unsigned(x64);
    sum_s += ::sinf(x);
    sum_c += ::cosf(x);
    x = 2.328306437e-10 * 2.0 * M_PI * unsigned(x64>>32);
    sum_s += ::sinf(x);
    sum_c += ::cosf(x);
  }

  EXPECT_GE(sum_s, -float(NSIM_RANDSINCOS));
  EXPECT_LE(sum_s, float(NSIM_RANDSINCOS));
  EXPECT_GE(sum_c, -float(NSIM_RANDSINCOS));
  EXPECT_LE(sum_c, float(NSIM_RANDSINCOS));
}

#ifdef CALIN_HAS_NR3_AVX2_RNGCORE
TEST(TestSIMD, SpeedTest100M_Random256SinCos32)
{
  float sum_s = 0;
  float sum_c = 0;
  uint64_t seed = calin::math::rng::RNG::uint64_from_random_device();
  NR3_AVX2_RNGCore core(seed);
  for(unsigned i=0;i<NSIM_RANDSINCOS/8;i++) {
    float x[8] __attribute__((aligned(32)));
    __m256 vx = core.uniform_zc_psvec256(2*M_PI);
    _mm256_store_ps(x, vx);
    sum_s += ::sinf(x[0]);
    sum_c += ::cosf(x[0]);
    sum_s += ::sinf(x[1]);
    sum_c += ::cosf(x[1]);
    sum_s += ::sinf(x[2]);
    sum_c += ::cosf(x[2]);
    sum_s += ::sinf(x[3]);
    sum_c += ::cosf(x[3]);
    sum_s += ::sinf(x[4]);
    sum_c += ::cosf(x[4]);
    sum_s += ::sinf(x[5]);
    sum_c += ::cosf(x[5]);
    sum_s += ::sinf(x[6]);
    sum_c += ::cosf(x[6]);
    sum_s += ::sinf(x[7]);
    sum_c += ::cosf(x[7]);
  }

  EXPECT_GE(sum_s, -float(NSIM_RANDSINCOS));
  EXPECT_LE(sum_s, float(NSIM_RANDSINCOS));
  EXPECT_GE(sum_c, -float(NSIM_RANDSINCOS));
  EXPECT_LE(sum_c, float(NSIM_RANDSINCOS));
}

TEST(TestSIMD, SpeedTest100M_Random256SinCos256)
{
  __m256 sum_s = _mm256_setzero_ps();
  __m256 sum_c = _mm256_setzero_ps();
  uint64_t seed = calin::math::rng::RNG::uint64_from_random_device();
  NR3_AVX2_RNGCore core(seed);
  for(unsigned i=0;i<NSIM_RANDSINCOS/8;i++) {
    __m256 x = core.uniform_zc_psvec256(8.0);
    __m256 s;
    __m256 c;
    avx2_sincosf_domain_pi_poly3(x, s, c);
    sum_s = _mm256_add_ps(sum_s, s);
    sum_c = _mm256_add_ps(sum_c, c);
  }
  EXPECT_GE(reinterpret_cast<float*>(&sum_s)[0], -float(NSIM_RANDSINCOS));
  EXPECT_LE(reinterpret_cast<float*>(&sum_s)[0], float(NSIM_RANDSINCOS));
  EXPECT_GE(reinterpret_cast<float*>(&sum_c)[0], -float(NSIM_RANDSINCOS));
  EXPECT_LE(reinterpret_cast<float*>(&sum_c)[0], float(NSIM_RANDSINCOS));
}

constexpr unsigned nchan = 2048;
constexpr unsigned nsamp = 60;

int all_imax[nchan];
int all_max[nchan];
int all_bkg[nchan];
int all_sig[nchan];
int all_sum_q[nchan];
int all_sum_qt[nchan];
float all_mean_t[nchan];

TEST(TestTraceAnalysis, Scalar)
{
  NR3_AVX2_RNGCore core(12345);
  uint16_t* hg = new uint16_t[nchan*nsamp];
  // uint16_t* lg = new uint16_t[nchan*nsamp];
  for(unsigned iloop=0;iloop<NSIM_TRACEANAL;iloop++)
  {
    const __m256i mask_12bit = _mm256_set1_epi16((1<<12)-1);
    for(unsigned i=0;i<nchan*nsamp/16;i++) {
      __m256i x = _mm256_and_si256(core.uniform_uivec256(), mask_12bit);
      _mm256_storeu_si256((__m256i*)(hg+i*16), x);
      // x = _mm256_and_si256(core.uniform_uivec256(), mask_12bit);
      // _mm256_store_si256((__m256i*)(lg+i*16), x);
    }

    int imax = 0;
    int max = 0;
    int bkg = 0;
    int sig = 0;
    int sum_q = 0;
    int sum_qt = 0;
    int win = 0;
    int isamp = 0;
    int samp[nsamp];

    for(unsigned ichan=0;ichan<nchan;ichan++) {
      samp[0] = hg[ichan*nsamp];
      imax = 0;
      max = samp[0];
      bkg = max;
      sum_qt = 0;
      for(isamp = 1;isamp<16;isamp++) {
        const unsigned _samp = hg[ichan*nsamp+isamp];
        samp[isamp] = _samp;
        bkg += _samp;
        if(_samp > max) {
          imax = isamp;
          max = _samp;
        }
        sum_qt += _samp*isamp;
      }
      sig = bkg;
      win = bkg;
      sum_q = bkg;
      for(;isamp<nsamp;isamp++) {
        const unsigned _samp = hg[ichan*nsamp+isamp];
        samp[isamp] = _samp;
        sum_q += _samp;
        sum_qt += _samp*isamp;
        win += _samp - samp[isamp-16];
        sig = std::max(sig, win);
        if(_samp > max) {
          imax = isamp;
          max = _samp;
        }
      }

#if 0
      if(iloop==0 and ichan<16) {
        for(unsigned isamp=0; isamp<nsamp; isamp++) {
          std::cout << samp[isamp] << ' ';
        }
        std::cout << '\n';
      }
#endif

      all_imax[ichan] = imax;
      all_max[ichan] = max;
      all_bkg[ichan] = bkg;
      all_sig[ichan] = sig;
      all_sum_q[ichan] = sum_q;
      all_sum_qt[ichan] = sum_qt;
      all_mean_t[ichan] =
        (double(sum_qt) - double(bkg*nsamp*(nsamp-1)/2)/16.0)/(double(sum_q) - double(bkg*nsamp)/16.0);
    }

    if(iloop==0) {
      for(unsigned ichan=0; ichan<32; ichan++) {
        std::cout << ichan << ' ' << all_imax[ichan] << ' ' << all_max[ichan] << ' '
          << all_bkg[ichan] << ' ' << all_sig[ichan] << ' '
          << all_sum_q[ichan] << ' ' << all_sum_qt[ichan] << ' '
          << all_mean_t[ichan] << '\n';
      }
    }
  }

  delete[] hg;
}

#define DO_ONE_SWIZZLE_U16(i) \
  tmp =  _mm256_unpackhi_epi16(*(data+i), *(data+i+1)); \
  *(data+i) =  _mm256_unpacklo_epi16(*(data+i), *(data+i+1)); \
  *(data+i+1) = tmp

#define DO_ONE_SWIZZLE_U32(i) \
  tmp =  _mm256_unpackhi_epi32(*(data+i), *(data+i+2)); \
  *(data+i) =  _mm256_unpacklo_epi32(*(data+i), *(data+i+2)); \
  *(data+i+2) = tmp

#define DO_ONE_SWIZZLE_U64(i) \
  tmp =  _mm256_unpackhi_epi64(*(data+i), *(data+i+4)); \
  *(data+i) =  _mm256_unpacklo_epi64(*(data+i), *(data+i+4)); \
  *(data+i+4) = tmp

#define DO_ONE_SWIZZLE_U128(i) \
  tmp =  _mm256_permute2x128_si256(*(data+i), *(data+i+8), 0x31); \
  *(data+i) =  _mm256_permute2x128_si256(*(data+i), *(data+i+8), 0x20); \
  *(data+i+8) = tmp

inline void swizzle_u16(__m256i* data)
{
  __m256i tmp;
  DO_ONE_SWIZZLE_U16(0);
  DO_ONE_SWIZZLE_U16(2);
  DO_ONE_SWIZZLE_U16(4);
  DO_ONE_SWIZZLE_U16(6);
  DO_ONE_SWIZZLE_U16(8);
  DO_ONE_SWIZZLE_U16(10);
  DO_ONE_SWIZZLE_U16(12);
  DO_ONE_SWIZZLE_U16(14);

  DO_ONE_SWIZZLE_U32(0);
  DO_ONE_SWIZZLE_U32(1);
  DO_ONE_SWIZZLE_U32(4);
  DO_ONE_SWIZZLE_U32(5);
  DO_ONE_SWIZZLE_U32(8);
  DO_ONE_SWIZZLE_U32(9);
  DO_ONE_SWIZZLE_U32(12);
  DO_ONE_SWIZZLE_U32(13);

  DO_ONE_SWIZZLE_U64(0);
  DO_ONE_SWIZZLE_U64(1);
  DO_ONE_SWIZZLE_U64(2);
  DO_ONE_SWIZZLE_U64(3);
  DO_ONE_SWIZZLE_U64(8);
  DO_ONE_SWIZZLE_U64(9);
  DO_ONE_SWIZZLE_U64(10);
  DO_ONE_SWIZZLE_U64(11);

  DO_ONE_SWIZZLE_U128(0);
  DO_ONE_SWIZZLE_U128(1);
  DO_ONE_SWIZZLE_U128(2);
  DO_ONE_SWIZZLE_U128(3);
  DO_ONE_SWIZZLE_U128(4);
  DO_ONE_SWIZZLE_U128(5);
  DO_ONE_SWIZZLE_U128(6);
  DO_ONE_SWIZZLE_U128(7);

  std::swap(data[1], data[4]);
  std::swap(data[3], data[6]);
  std::swap(data[9], data[12]);
  std::swap(data[11], data[14]);
}

TEST(TestTraceAnalysis, AVX2)
{
  NR3_AVX2_RNGCore core(12345);
  uint16_t* hg = new uint16_t[(nchan+1)*nsamp];
  // uint16_t* lg = new uint16_t[nchan*nsamp];
  for(unsigned iloop=0;iloop<NSIM_TRACEANAL;iloop++)
  {
    const __m256i mask_12bit = _mm256_set1_epi16((1<<12)-1);
    for(unsigned i=0;i<nchan*nsamp/16;i++) {
      __m256i x = _mm256_and_si256(core.uniform_uivec256(), mask_12bit);
#if 0
      __m256i x = _mm256_setr_epi16(
        (i*16+0)/60,(i*16+1)/60,(i*16+2)/60,(i*16+3)/60,
        (i*16+4)/60,(i*16+5)/60,(i*16+6)/60,(i*16+7)/60,
        (i*16+8)/60,(i*16+9)/60,(i*16+10)/60,(i*16+11)/60,
        (i*16+12)/60,(i*16+13)/60,(i*16+14)/60,(i*16+15)/60);
#endif
      _mm256_storeu_si256((__m256i*)(hg+i*16), x);
      // x = _mm256_and_si256(core.uniform_uivec256(), mask_12bit);
      // _mm256_store_si256((__m256i*)(lg+i*16), x);
    }

    const unsigned nv_trace = (nsamp+15)/16;
    const unsigned nv_block = nv_trace*16;
    __m256i* samples = new __m256i[nv_block];

    const unsigned nblock = nchan/16;
    for(unsigned iblock=0;iblock<nblock;iblock++)
    {
      uint16_t* base = hg + iblock*nsamp*16;
      __m256i* vp = samples;
      for(unsigned iv_trace=0; iv_trace<nv_trace; iv_trace++) {
        for(unsigned ivec=0; ivec<16; ivec++) {
          *(vp++) = _mm256_loadu_si256((__m256i*)(base + iv_trace*16 + nsamp*ivec));
        }
        swizzle_u16(samples + iv_trace*16);
      }

      __m256i max = samples[0];
      __m256i imax = _mm256_set1_epi16(0);

      __m256i bkg_l = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(samples[0],0));
      __m256i bkg_u = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(samples[0],1));

      __m256i sum_qt_l = _mm256_set1_epi32(0);
      __m256i sum_qt_u = _mm256_set1_epi32(0);

      unsigned isamp;
      for(isamp = 1;isamp<16;isamp++) {
        __m256i visamp = _mm256_set1_epi16(isamp);
        __m256i mask = _mm256_cmpgt_epi16(samples[isamp], max);
        max = _mm256_blendv_epi8(max, samples[isamp], mask);
        imax = _mm256_blendv_epi8(imax, visamp, mask);

        __m256i samp_l = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(samples[isamp],0));
        __m256i samp_u = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(samples[isamp],1));

        bkg_l = _mm256_add_epi32(bkg_l, samp_l);
        bkg_u = _mm256_add_epi32(bkg_u, samp_u);

        visamp = _mm256_set1_epi32(isamp);
        sum_qt_l = _mm256_add_epi32(sum_qt_l, _mm256_mullo_epi32(visamp, samp_l));
        sum_qt_u = _mm256_add_epi32(sum_qt_u, _mm256_mullo_epi32(visamp, samp_u));
      }

      __m256i sig_l = bkg_l;
      __m256i sig_u = bkg_u;
      __m256i win_l = bkg_l;
      __m256i win_u = bkg_u;
      __m256i sum_q_l = bkg_l;
      __m256i sum_q_u = bkg_u;

      for(;isamp<nsamp;isamp++) {
        __m256i visamp = _mm256_set1_epi16(isamp);
        __m256i mask = _mm256_cmpgt_epi16(samples[isamp], max);
        max = _mm256_blendv_epi8(max, samples[isamp], mask);
        imax = _mm256_blendv_epi8(imax, visamp, mask);

        __m256i samp_l = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(samples[isamp],0));
        __m256i samp_u = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(samples[isamp],1));

        win_l = _mm256_add_epi32(win_l, samp_l);
        win_u = _mm256_add_epi32(win_u, samp_u);

        sum_q_l = _mm256_add_epi32(sum_q_l, samp_l);
        sum_q_u = _mm256_add_epi32(sum_q_u, samp_u);

        visamp = _mm256_set1_epi32(isamp);
        sum_qt_l = _mm256_add_epi32(sum_qt_l, _mm256_mullo_epi32(visamp, samp_l));
        sum_qt_u = _mm256_add_epi32(sum_qt_u, _mm256_mullo_epi32(visamp, samp_u));

        samp_l = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(samples[isamp-16],0));
        samp_u = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(samples[isamp-16],1));

        win_l = _mm256_sub_epi32(win_l, samp_l);
        win_u = _mm256_sub_epi32(win_u, samp_u);

        sig_l = _mm256_max_epu32(sig_l, win_l);
        sig_u = _mm256_max_epu32(sig_u, win_u);
      }

      _mm256_storeu_si256((__m256i*)(all_imax+iblock*16), _mm256_cvtepu16_epi32(_mm256_extracti128_si256(imax,0)));
      _mm256_storeu_si256((__m256i*)(all_imax+iblock*16+8), _mm256_cvtepu16_epi32(_mm256_extracti128_si256(imax,1)));

      _mm256_storeu_si256((__m256i*)(all_max+iblock*16), _mm256_cvtepu16_epi32(_mm256_extracti128_si256(max,0)));
      _mm256_storeu_si256((__m256i*)(all_max+iblock*16+8), _mm256_cvtepu16_epi32(_mm256_extracti128_si256(max,1)));

      _mm256_storeu_si256((__m256i*)(all_bkg+iblock*16), bkg_l);
      _mm256_storeu_si256((__m256i*)(all_bkg+iblock*16+8), bkg_u);

      _mm256_storeu_si256((__m256i*)(all_sig+iblock*16), sig_l);
      _mm256_storeu_si256((__m256i*)(all_sig+iblock*16+8), sig_u);

      _mm256_storeu_si256((__m256i*)(all_sum_q+iblock*16), sum_q_l);
      _mm256_storeu_si256((__m256i*)(all_sum_q+iblock*16+8), sum_q_u);

      _mm256_storeu_si256((__m256i*)(all_sum_qt+iblock*16), sum_qt_l);
      _mm256_storeu_si256((__m256i*)(all_sum_qt+iblock*16+8), sum_qt_u);

      __m256 tmean_l = _mm256_div_ps(
        _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_slli_epi32(sum_qt_l, 4), _mm256_mullo_epi32(bkg_l, _mm256_set1_epi32(nsamp*(nsamp-1)/2)))),
        _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_slli_epi32(sum_q_l, 4), _mm256_mullo_epi32(bkg_l, _mm256_set1_epi32(nsamp)))));
      __m256 tmean_u = _mm256_div_ps(
        _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_slli_epi32(sum_qt_u, 4), _mm256_mullo_epi32(bkg_u, _mm256_set1_epi32(nsamp*(nsamp-1)/2)))),
        _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_slli_epi32(sum_q_u, 4), _mm256_mullo_epi32(bkg_u, _mm256_set1_epi32(nsamp)))));

      _mm256_storeu_ps(all_mean_t+iblock*16, tmean_l);
      _mm256_storeu_ps(all_mean_t+iblock*16+8, tmean_u);


#if 0
      if(iloop==0 and iblock==0) {
        uint16_t _samples[16] __attribute__((aligned(32)));
        for(unsigned ichan=0; ichan<16; ichan++) {
          for(unsigned isamp=0; isamp<nsamp; isamp++) {
            _mm256_storeu_si256((__m256i *)_samples, samples[isamp]);
            std::cout << _samples[ichan] << ' ';
          }
          std::cout << '\n';
        }
      }
#endif
    }

    if(iloop==0) {
      for(unsigned ichan=0; ichan<32; ichan++) {
        std::cout << ichan << ' ' << all_imax[ichan] << ' ' << all_max[ichan] << ' '
          << all_bkg[ichan] << ' ' << all_sig[ichan] << ' '
          << all_sum_q[ichan] << ' ' << all_sum_qt[ichan] << ' '
          << all_mean_t[ichan] << '\n';
      }
    }
  }

  delete[] hg;
}

uint32_t cov[nchan*nsamp*(nsamp+1)/2] __attribute__((aligned(32)));

TEST(TestTraceCov, Scalar)
{
  NR3_AVX2_RNGCore core(12345);
  uint16_t* hg = new uint16_t[nchan*nsamp];
  std::fill(cov, cov+nchan*nsamp*(nsamp+1)/2, 0);
  // uint16_t* lg = new uint16_t[nchan*nsamp];
  for(unsigned iloop=0;iloop<NSIM_TRACECOV;iloop++)
  {
    const __m256i mask_12bit = _mm256_set1_epi16((1<<12)-1);
    for(unsigned i=0;i<nchan*nsamp/16;i++) {
      __m256i x = _mm256_and_si256(core.uniform_uivec256(), mask_12bit);
      _mm256_storeu_si256((__m256i*)(hg+i*16), x);
      // x = _mm256_and_si256(core.uniform_uivec256(), mask_12bit);
      // _mm256_store_si256((__m256i*)(lg+i*16), x);
    }

    uint32_t*__restrict__ cov_base = cov;
    uint16_t*__restrict__ samp_base = hg;
    for(unsigned ichan=0;ichan<nchan;ichan++) {
      for(unsigned isamp=0;isamp<nsamp;isamp++) {
        const uint16_t sampi = samp_base[isamp];
        for(unsigned jsamp=isamp;jsamp<nsamp;jsamp++) {
          const uint16_t sampj = samp_base[jsamp];
          *(cov_base++) += sampi * sampj;
        }
      }
      samp_base += nsamp;
    }
  }
  for(unsigned ichan=0;ichan<20;ichan++) {
    for(unsigned isamp=0;isamp<5;isamp++)
      std::cout << ' ' << cov[ichan*nsamp*(nsamp+1)/2 + isamp];
    std::cout << '\n';
  }
}

TEST(TestTraceCov, AVX2)
{
  NR3_AVX2_RNGCore core(12345);
  uint16_t* hg;
  calin::util::memory::safe_aligned_calloc(hg, nchan*nsamp);
  std::fill(cov, cov+nchan*nsamp*(nsamp+1)/2, 0);
  const unsigned nv_trace = (nsamp+15)/16;
  const unsigned nv_block = nv_trace*16;
  __m256i* samples;
  calin::util::memory::safe_aligned_calloc(samples, nv_block);
  const unsigned nblock = nchan/16;

  for(unsigned iloop=0;iloop<NSIM_TRACECOV;iloop++)
  {
    const __m256i mask_12bit = _mm256_set1_epi16((1<<12)-1);
    for(unsigned i=0;i<nchan*nsamp/16;i++) {
      __m256i x = _mm256_and_si256(core.uniform_uivec256(), mask_12bit);
      _mm256_storeu_si256((__m256i*)(hg+i*16), x);
      // x = _mm256_and_si256(core.uniform_uivec256(), mask_12bit);
      // _mm256_store_si256((__m256i*)(lg+i*16), x);
    }
#if 1
    __m256i*__restrict__ cov_base = (__m256i*__restrict__)cov;
    for(unsigned iblock=0;iblock<nblock;iblock++)
    {
#if 1
      uint16_t* base = hg + iblock*nsamp*16;
      __m256i* vp = samples;
      for(unsigned iv_trace=0; iv_trace<nv_trace; iv_trace++) {
        for(unsigned ivec=0; ivec<16; ivec++) {
//          std::cerr << iblock << ' ' << iv_trace << ' ' << ivec << ' ' << (iblock*nsamp*16 + iv_trace*16 + nsamp*ivec) <<  '\n';
          *(vp++) = _mm256_loadu_si256((__m256i*)(base + iv_trace*16 + nsamp*ivec));
        }
        calin::math::simd::avx2_m256_swizzle_u16(samples + iv_trace*16);
      }
#endif
      for(unsigned isamp=0;isamp<nsamp;isamp++) {
        const __m256i sampi = samples[isamp];
        for(unsigned jsamp=isamp;jsamp<nsamp;jsamp++) {
          const __m256i sampj = samples[jsamp];
          const __m256i prod_lo = _mm256_mullo_epi16(sampi, sampj);
          const __m256i prod_hi = _mm256_mulhi_epu16(sampi, sampj);
          *cov_base = _mm256_add_epi32(*cov_base, _mm256_unpacklo_epi16(prod_lo, prod_hi));
          cov_base++;
          *cov_base = _mm256_add_epi32(*cov_base, _mm256_unpackhi_epi16(prod_lo, prod_hi));
          cov_base++;
        }
      }
    }
#endif
  }
  for(unsigned ichan=0;ichan<20;ichan++) {
    unsigned iblock = ichan/16;
    unsigned irem = ichan%16;
    for(unsigned isamp=0;isamp<5;isamp++)
      std::cout << ' ' << cov[iblock*16*nsamp*(nsamp+1)/2 + isamp*16 + irem];
    std::cout << '\n';
  }

  free(hg);
  free(samples);
}

TEST(TestTraceCov, AVX2_Unroll)
{
  NR3_AVX2_RNGCore core(12345);
  uint16_t* hg;
  calin::util::memory::safe_aligned_calloc(hg, nchan*nsamp);
  std::fill(cov, cov+nchan*nsamp*(nsamp+1)/2, 0);
  const unsigned nv_trace = (nsamp+15)/16;
  const unsigned nv_block = nv_trace*16;
  __m256i* samples;
  calin::util::memory::safe_aligned_calloc(samples, nv_block);
  const unsigned nblock = nchan/16;

  for(unsigned iloop=0;iloop<NSIM_TRACECOV;iloop++)
  {
    const __m256i mask_12bit = _mm256_set1_epi16((1<<12)-1);
    for(unsigned i=0;i<nchan*nsamp/16;i++) {
      __m256i x = _mm256_and_si256(core.uniform_uivec256(), mask_12bit);
      _mm256_storeu_si256((__m256i*)(hg+i*16), x);
      // x = _mm256_and_si256(core.uniform_uivec256(), mask_12bit);
      // _mm256_store_si256((__m256i*)(lg+i*16), x);
    }

    __m256i*__restrict__ cov_base = (__m256i*__restrict__)cov;
    for(unsigned iblock=0;iblock<nblock;iblock++)
    {
#if 1
      uint16_t* base = hg + iblock*nsamp*16;
      __m256i* vp = samples;
      for(unsigned iv_trace=0; iv_trace<nv_trace; iv_trace++) {
        for(unsigned ivec=0; ivec<16; ivec++) {
          *(vp++) = _mm256_loadu_si256((__m256i*)(base + iv_trace*16 + nsamp*ivec));
        }
        calin::math::simd::avx2_m256_swizzle_u16(samples + iv_trace*16);
      }
#endif
      for(unsigned isamp=0;isamp<nsamp;isamp+=2) {
        __m256i*__restrict__ cov_base_ipo = cov_base + (nsamp - isamp)*2;

        const __m256i sampi = samples[isamp];

        const __m256i prod_lo = _mm256_mullo_epi16(sampi, sampi);
        const __m256i prod_hi = _mm256_mulhi_epu16(sampi, sampi);
        *cov_base = _mm256_add_epi32(*cov_base, _mm256_unpacklo_epi16(prod_lo, prod_hi));
        cov_base++;
        *cov_base = _mm256_add_epi32(*cov_base, _mm256_unpackhi_epi16(prod_lo, prod_hi));
        cov_base++;

        const __m256i sampipo = samples[isamp+1];
        for(unsigned jsamp=isamp+1;jsamp<nsamp;jsamp++) {
          const __m256i sampj = samples[jsamp];

          const __m256i prod_lo = _mm256_mullo_epi16(sampi, sampj);
          const __m256i prod_hi = _mm256_mulhi_epu16(sampi, sampj);
          *cov_base = _mm256_add_epi32(*cov_base, _mm256_unpacklo_epi16(prod_lo, prod_hi));
          cov_base++;
          *cov_base = _mm256_add_epi32(*cov_base, _mm256_unpackhi_epi16(prod_lo, prod_hi));
          cov_base++;

          const __m256i prod_lo1 = _mm256_mullo_epi16(sampipo, sampj);
          const __m256i prod_hi1 = _mm256_mulhi_epu16(sampipo, sampj);
          *cov_base_ipo = _mm256_add_epi32(*cov_base_ipo, _mm256_unpacklo_epi16(prod_lo1, prod_hi1));
          cov_base_ipo++;
          *cov_base_ipo = _mm256_add_epi32(*cov_base_ipo, _mm256_unpackhi_epi16(prod_lo1, prod_hi1));
          cov_base_ipo++;
        }
        cov_base = cov_base_ipo;
      }
    }
  }
  for(unsigned ichan=0;ichan<20;ichan++) {
    unsigned iblock = ichan/16;
    unsigned irem = ichan%16;
    for(unsigned isamp=0;isamp<5;isamp++)
      std::cout << ' ' << cov[iblock*16*nsamp*(nsamp+1)/2 + isamp*16 + irem];
    std::cout << '\n';
  }

  free(hg);
  free(samples);
}

TEST(TestTraceCov, AVX2_2Event)
{
  NR3_AVX2_RNGCore core(12345);
  uint16_t* hg1;
  uint16_t* hg2;
  calin::util::memory::safe_aligned_calloc(hg1, nchan*nsamp);
  calin::util::memory::safe_aligned_calloc(hg2, nchan*nsamp);

  std::fill(cov, cov+nchan*nsamp*(nsamp+1)/2, 0);

  const unsigned nv_trace = (nsamp+15)/16;
  const unsigned nv_block = nv_trace*16;
  __m256i* samples1;
  __m256i* samples2;
  calin::util::memory::safe_aligned_calloc(samples1, nv_block);
  calin::util::memory::safe_aligned_calloc(samples2, nv_block);
  const unsigned nblock = nchan/16;

  for(unsigned iloop=0;iloop<NSIM_TRACECOV/2;iloop++)
  {
    const __m256i mask_12bit = _mm256_set1_epi16((1<<12)-1);
    for(unsigned i=0;i<nchan*nsamp/16;i++) {
      __m256i x = _mm256_and_si256(core.uniform_uivec256(), mask_12bit);
      _mm256_storeu_si256((__m256i*)(hg1+i*16), x);
    }
    for(unsigned i=0;i<nchan*nsamp/16;i++) {
      __m256i x = _mm256_and_si256(core.uniform_uivec256(), mask_12bit);
      _mm256_storeu_si256((__m256i*)(hg2+i*16), x);
    }

    __m256i*__restrict__ cov_base = (__m256i*__restrict__)cov;
    for(unsigned iblock=0;iblock<nblock;iblock++)
    {
      uint16_t* base = hg1 + iblock*nsamp*16;
      __m256i* vp = samples1;
      for(unsigned iv_trace=0; iv_trace<nv_trace; iv_trace++) {
        for(unsigned ivec=0; ivec<16; ivec++) {
          *(vp++) = _mm256_loadu_si256((__m256i*)(base + iv_trace*16 + nsamp*ivec));
        }
        calin::math::simd::avx2_m256_swizzle_u16(samples1 + iv_trace*16);
      }

      base = hg2 + iblock*nsamp*16;
      vp = samples2;
      for(unsigned iv_trace=0; iv_trace<nv_trace; iv_trace++) {
        for(unsigned ivec=0; ivec<16; ivec++) {
          *(vp++) = _mm256_loadu_si256((__m256i*)(base + iv_trace*16 + nsamp*ivec));
        }
        calin::math::simd::avx2_m256_swizzle_u16(samples2 + iv_trace*16);
      }

      for(unsigned isamp=0;isamp<nsamp;isamp++) {
        const __m256i sampi1 = samples1[isamp];
        const __m256i sampi2 = samples2[isamp];
        for(unsigned jsamp=isamp;jsamp<nsamp;jsamp++) {
          __m256i sampj = samples1[jsamp];
          __m256i prod_lo = _mm256_mullo_epi16(sampi1, sampj);
          __m256i prod_hi = _mm256_mulhi_epu16(sampi1, sampj);
          *cov_base = _mm256_add_epi32(*cov_base, _mm256_unpacklo_epi16(prod_lo, prod_hi));
          *(cov_base+1) = _mm256_add_epi32(*(cov_base+1), _mm256_unpackhi_epi16(prod_lo, prod_hi));

          sampj = samples2[jsamp];
          prod_lo = _mm256_mullo_epi16(sampi2, sampj);
          prod_hi = _mm256_mulhi_epu16(sampi2, sampj);
          *cov_base = _mm256_add_epi32(*cov_base, _mm256_unpacklo_epi16(prod_lo, prod_hi));
          *(cov_base+1) = _mm256_add_epi32(*(cov_base+1), _mm256_unpackhi_epi16(prod_lo, prod_hi));

          cov_base+=2;
        }
      }
    }
  }
  for(unsigned ichan=0;ichan<20;ichan++) {
    unsigned iblock = ichan/16;
    unsigned irem = ichan%16;
    for(unsigned isamp=0;isamp<5;isamp++)
      std::cout << ' ' << cov[iblock*16*nsamp*(nsamp+1)/2 + isamp*16 + irem];
    std::cout << '\n';
  }

  free(hg1);
  free(hg2);
  free(samples1);
  free(samples2);
}

TEST(TestTraceCov, AVX2_4Event)
{
  NR3_AVX2_RNGCore core(12345);
  uint16_t* hg1;
  uint16_t* hg2;
  uint16_t* hg3;
  uint16_t* hg4;
  calin::util::memory::safe_aligned_calloc(hg1, nchan*nsamp);
  calin::util::memory::safe_aligned_calloc(hg2, nchan*nsamp);
  calin::util::memory::safe_aligned_calloc(hg3, nchan*nsamp);
  calin::util::memory::safe_aligned_calloc(hg4, nchan*nsamp);

  std::fill(cov, cov+nchan*nsamp*(nsamp+1)/2, 0);

  const unsigned nv_trace = (nsamp+15)/16;
  const unsigned nv_block = nv_trace*16;
  __m256i* samples1;
  __m256i* samples2;
  __m256i* samples3;
  __m256i* samples4;
  calin::util::memory::safe_aligned_calloc(samples1, nv_block);
  calin::util::memory::safe_aligned_calloc(samples2, nv_block);
  calin::util::memory::safe_aligned_calloc(samples3, nv_block);
  calin::util::memory::safe_aligned_calloc(samples4, nv_block);
  const unsigned nblock = nchan/16;

  for(unsigned iloop=0;iloop<NSIM_TRACECOV;iloop+=4)
  {
    const __m256i mask_12bit = _mm256_set1_epi16((1<<12)-1);
    for(unsigned i=0;i<nchan*nsamp/16;i++) {
      __m256i x = _mm256_and_si256(core.uniform_uivec256(), mask_12bit);
      _mm256_storeu_si256((__m256i*)(hg1+i*16), x);
    }
    for(unsigned i=0;i<nchan*nsamp/16;i++) {
      __m256i x = _mm256_and_si256(core.uniform_uivec256(), mask_12bit);
      _mm256_storeu_si256((__m256i*)(hg2+i*16), x);
    }
    for(unsigned i=0;i<nchan*nsamp/16;i++) {
      __m256i x = _mm256_and_si256(core.uniform_uivec256(), mask_12bit);
      _mm256_storeu_si256((__m256i*)(hg3+i*16), x);
    }
    for(unsigned i=0;i<nchan*nsamp/16;i++) {
      __m256i x = _mm256_and_si256(core.uniform_uivec256(), mask_12bit);
      _mm256_storeu_si256((__m256i*)(hg4+i*16), x);
    }

    __m256i*__restrict__ cov_base = (__m256i*__restrict__)cov;
    for(unsigned iblock=0;iblock<nblock;iblock++)
    {
      uint16_t* base = hg1 + iblock*nsamp*16;
      __m256i* vp = samples1;
      for(unsigned iv_trace=0; iv_trace<nv_trace; iv_trace++) {
        for(unsigned ivec=0; ivec<16; ivec++) {
          *(vp++) = _mm256_loadu_si256((__m256i*)(base + iv_trace*16 + nsamp*ivec));
        }
        calin::math::simd::avx2_m256_swizzle_u16(samples1 + iv_trace*16);
      }

      base = hg2 + iblock*nsamp*16;
      vp = samples2;
      for(unsigned iv_trace=0; iv_trace<nv_trace; iv_trace++) {
        for(unsigned ivec=0; ivec<16; ivec++) {
          *(vp++) = _mm256_loadu_si256((__m256i*)(base + iv_trace*16 + nsamp*ivec));
        }
        calin::math::simd::avx2_m256_swizzle_u16(samples2 + iv_trace*16);
      }

      base = hg3 + iblock*nsamp*16;
      vp = samples3;
      for(unsigned iv_trace=0; iv_trace<nv_trace; iv_trace++) {
        for(unsigned ivec=0; ivec<16; ivec++) {
          *(vp++) = _mm256_loadu_si256((__m256i*)(base + iv_trace*16 + nsamp*ivec));
        }
        calin::math::simd::avx2_m256_swizzle_u16(samples3 + iv_trace*16);
      }

      base = hg4 + iblock*nsamp*16;
      vp = samples4;
      for(unsigned iv_trace=0; iv_trace<nv_trace; iv_trace++) {
        for(unsigned ivec=0; ivec<16; ivec++) {
          *(vp++) = _mm256_loadu_si256((__m256i*)(base + iv_trace*16 + nsamp*ivec));
        }
        calin::math::simd::avx2_m256_swizzle_u16(samples4 + iv_trace*16);
      }

      for(unsigned isamp=0;isamp<nsamp;isamp++) {
        const __m256i sampi1 = samples1[isamp];
        const __m256i sampi2 = samples2[isamp];
        const __m256i sampi3 = samples3[isamp];
        const __m256i sampi4 = samples4[isamp];

        __m256i prod_lo = _mm256_mullo_epi16(sampi1, sampi1);
        __m256i prod_hi = _mm256_mulhi_epu16(sampi1, sampi1);
        __m256i sum_l = _mm256_unpacklo_epi16(prod_lo, prod_hi);
        __m256i sum_u = _mm256_unpackhi_epi16(prod_lo, prod_hi);

        prod_lo = _mm256_mullo_epi16(sampi2, sampi2);
        prod_hi = _mm256_mulhi_epu16(sampi2, sampi2);
        sum_l = _mm256_add_epi32(sum_l, _mm256_unpacklo_epi16(prod_lo, prod_hi));
        sum_u = _mm256_add_epi32(sum_u, _mm256_unpackhi_epi16(prod_lo, prod_hi));

        prod_lo = _mm256_mullo_epi16(sampi3, sampi3);
        prod_hi = _mm256_mulhi_epu16(sampi3, sampi3);
        sum_l = _mm256_add_epi32(sum_l, _mm256_unpacklo_epi16(prod_lo, prod_hi));
        sum_u = _mm256_add_epi32(sum_u, _mm256_unpackhi_epi16(prod_lo, prod_hi));

        prod_lo = _mm256_mullo_epi16(sampi4, sampi4);
        prod_hi = _mm256_mulhi_epu16(sampi4, sampi4);
        sum_l = _mm256_add_epi32(sum_l, _mm256_unpacklo_epi16(prod_lo, prod_hi));
        sum_u = _mm256_add_epi32(sum_u, _mm256_unpackhi_epi16(prod_lo, prod_hi));

        *cov_base = _mm256_add_epi32(*cov_base, sum_l);
        cov_base++;
        *cov_base = _mm256_add_epi32(*cov_base, sum_u);
        cov_base++;

        for(unsigned jsamp=isamp+1;jsamp<nsamp;jsamp++) {
          __m256i sampj = samples1[jsamp];
          prod_lo = _mm256_mullo_epi16(sampi1, sampj);
          prod_hi = _mm256_mulhi_epu16(sampi1, sampj);
          sum_l = _mm256_unpacklo_epi16(prod_lo, prod_hi);
          sum_u = _mm256_unpackhi_epi16(prod_lo, prod_hi);

          sampj = samples2[jsamp];
          prod_lo = _mm256_mullo_epi16(sampi2, sampj);
          prod_hi = _mm256_mulhi_epu16(sampi2, sampj);
          sum_l = _mm256_add_epi32(sum_l, _mm256_unpacklo_epi16(prod_lo, prod_hi));
          sum_u = _mm256_add_epi32(sum_u, _mm256_unpackhi_epi16(prod_lo, prod_hi));

          sampj = samples3[jsamp];
          prod_lo = _mm256_mullo_epi16(sampi3, sampj);
          prod_hi = _mm256_mulhi_epu16(sampi3, sampj);
          sum_l = _mm256_add_epi32(sum_l, _mm256_unpacklo_epi16(prod_lo, prod_hi));
          sum_u = _mm256_add_epi32(sum_u, _mm256_unpackhi_epi16(prod_lo, prod_hi));

          sampj = samples4[jsamp];
          prod_lo = _mm256_mullo_epi16(sampi4, sampj);
          prod_hi = _mm256_mulhi_epu16(sampi4, sampj);
          sum_l = _mm256_add_epi32(sum_l, _mm256_unpacklo_epi16(prod_lo, prod_hi));
          sum_u = _mm256_add_epi32(sum_u, _mm256_unpackhi_epi16(prod_lo, prod_hi));

          *cov_base = _mm256_add_epi32(*cov_base, sum_l);
          cov_base++;
          *cov_base = _mm256_add_epi32(*cov_base, sum_u);
          cov_base++;
        }
      }
    }
  }
  for(unsigned ichan=0;ichan<20;ichan++) {
    unsigned iblock = ichan/16;
    unsigned irem = ichan%16;
    for(unsigned isamp=0;isamp<5;isamp++)
      std::cout << ' ' << cov[iblock*16*nsamp*(nsamp+1)/2 + isamp*16 + irem];
    std::cout << '\n';
  }

  free(hg1);
  free(hg2);
  free(hg3);
  free(hg4);
  free(samples1);
  free(samples2);
  free(samples3);
  free(samples4);
}

TEST(TestTraceCov, AVX2_8Event)
{
  NR3_AVX2_RNGCore core(12345);
  uint16_t* hg1;
  uint16_t* hg2;
  uint16_t* hg3;
  uint16_t* hg4;
  uint16_t* hg5;
  uint16_t* hg6;
  uint16_t* hg7;
  uint16_t* hg8;
  calin::util::memory::safe_aligned_calloc(hg1, nchan*nsamp);
  calin::util::memory::safe_aligned_calloc(hg2, nchan*nsamp);
  calin::util::memory::safe_aligned_calloc(hg3, nchan*nsamp);
  calin::util::memory::safe_aligned_calloc(hg4, nchan*nsamp);
  calin::util::memory::safe_aligned_calloc(hg5, nchan*nsamp);
  calin::util::memory::safe_aligned_calloc(hg6, nchan*nsamp);
  calin::util::memory::safe_aligned_calloc(hg7, nchan*nsamp);
  calin::util::memory::safe_aligned_calloc(hg8, nchan*nsamp);

  std::fill(cov, cov+nchan*nsamp*(nsamp+1)/2, 0);

  const unsigned nv_trace = (nsamp+15)/16;
  const unsigned nv_block = nv_trace*16;
  __m256i* samples1;
  __m256i* samples2;
  __m256i* samples3;
  __m256i* samples4;
  __m256i* samples5;
  __m256i* samples6;
  __m256i* samples7;
  __m256i* samples8;
  calin::util::memory::safe_aligned_calloc(samples1, nv_block);
  calin::util::memory::safe_aligned_calloc(samples2, nv_block);
  calin::util::memory::safe_aligned_calloc(samples3, nv_block);
  calin::util::memory::safe_aligned_calloc(samples4, nv_block);
  calin::util::memory::safe_aligned_calloc(samples5, nv_block);
  calin::util::memory::safe_aligned_calloc(samples6, nv_block);
  calin::util::memory::safe_aligned_calloc(samples7, nv_block);
  calin::util::memory::safe_aligned_calloc(samples8, nv_block);
  const unsigned nblock = nchan/16;

  for(unsigned iloop=0;iloop<NSIM_TRACECOV;iloop+=8)
  {
    const __m256i mask_12bit = _mm256_set1_epi16((1<<12)-1);
    for(unsigned i=0;i<nchan*nsamp/16;i++) {
      __m256i x = _mm256_and_si256(core.uniform_uivec256(), mask_12bit);
      _mm256_storeu_si256((__m256i*)(hg1+i*16), x);
    }
    for(unsigned i=0;i<nchan*nsamp/16;i++) {
      __m256i x = _mm256_and_si256(core.uniform_uivec256(), mask_12bit);
      _mm256_storeu_si256((__m256i*)(hg2+i*16), x);
    }
    for(unsigned i=0;i<nchan*nsamp/16;i++) {
      __m256i x = _mm256_and_si256(core.uniform_uivec256(), mask_12bit);
      _mm256_storeu_si256((__m256i*)(hg3+i*16), x);
    }
    for(unsigned i=0;i<nchan*nsamp/16;i++) {
      __m256i x = _mm256_and_si256(core.uniform_uivec256(), mask_12bit);
      _mm256_storeu_si256((__m256i*)(hg4+i*16), x);
    }
    for(unsigned i=0;i<nchan*nsamp/16;i++) {
      __m256i x = _mm256_and_si256(core.uniform_uivec256(), mask_12bit);
      _mm256_storeu_si256((__m256i*)(hg5+i*16), x);
    }
    for(unsigned i=0;i<nchan*nsamp/16;i++) {
      __m256i x = _mm256_and_si256(core.uniform_uivec256(), mask_12bit);
      _mm256_storeu_si256((__m256i*)(hg6+i*16), x);
    }
    for(unsigned i=0;i<nchan*nsamp/16;i++) {
      __m256i x = _mm256_and_si256(core.uniform_uivec256(), mask_12bit);
      _mm256_storeu_si256((__m256i*)(hg7+i*16), x);
    }
    for(unsigned i=0;i<nchan*nsamp/16;i++) {
      __m256i x = _mm256_and_si256(core.uniform_uivec256(), mask_12bit);
      _mm256_storeu_si256((__m256i*)(hg8+i*16), x);
    }


    __m256i*__restrict__ cov_base = (__m256i*__restrict__)cov;
    for(unsigned iblock=0;iblock<nblock;iblock++)
    {
      uint16_t* base = hg1 + iblock*nsamp*16;
      __m256i* vp = samples1;
      for(unsigned iv_trace=0; iv_trace<nv_trace; iv_trace++) {
        for(unsigned ivec=0; ivec<16; ivec++) {
          *(vp++) = _mm256_loadu_si256((__m256i*)(base + iv_trace*16 + nsamp*ivec));
        }
        calin::math::simd::avx2_m256_swizzle_u16(samples1 + iv_trace*16);
      }

      base = hg2 + iblock*nsamp*16;
      vp = samples2;
      for(unsigned iv_trace=0; iv_trace<nv_trace; iv_trace++) {
        for(unsigned ivec=0; ivec<16; ivec++) {
          *(vp++) = _mm256_loadu_si256((__m256i*)(base + iv_trace*16 + nsamp*ivec));
        }
        calin::math::simd::avx2_m256_swizzle_u16(samples2 + iv_trace*16);
      }

      base = hg3 + iblock*nsamp*16;
      vp = samples3;
      for(unsigned iv_trace=0; iv_trace<nv_trace; iv_trace++) {
        for(unsigned ivec=0; ivec<16; ivec++) {
          *(vp++) = _mm256_loadu_si256((__m256i*)(base + iv_trace*16 + nsamp*ivec));
        }
        calin::math::simd::avx2_m256_swizzle_u16(samples3 + iv_trace*16);
      }

      base = hg4 + iblock*nsamp*16;
      vp = samples4;
      for(unsigned iv_trace=0; iv_trace<nv_trace; iv_trace++) {
        for(unsigned ivec=0; ivec<16; ivec++) {
          *(vp++) = _mm256_loadu_si256((__m256i*)(base + iv_trace*16 + nsamp*ivec));
        }
        calin::math::simd::avx2_m256_swizzle_u16(samples4 + iv_trace*16);
      }

      base = hg5 + iblock*nsamp*16;
      vp = samples5;
      for(unsigned iv_trace=0; iv_trace<nv_trace; iv_trace++) {
        for(unsigned ivec=0; ivec<16; ivec++) {
          *(vp++) = _mm256_loadu_si256((__m256i*)(base + iv_trace*16 + nsamp*ivec));
        }
        calin::math::simd::avx2_m256_swizzle_u16(samples5 + iv_trace*16);
      }

      base = hg6 + iblock*nsamp*16;
      vp = samples6;
      for(unsigned iv_trace=0; iv_trace<nv_trace; iv_trace++) {
        for(unsigned ivec=0; ivec<16; ivec++) {
          *(vp++) = _mm256_loadu_si256((__m256i*)(base + iv_trace*16 + nsamp*ivec));
        }
        calin::math::simd::avx2_m256_swizzle_u16(samples6 + iv_trace*16);
      }

      base = hg7 + iblock*nsamp*16;
      vp = samples7;
      for(unsigned iv_trace=0; iv_trace<nv_trace; iv_trace++) {
        for(unsigned ivec=0; ivec<16; ivec++) {
          *(vp++) = _mm256_loadu_si256((__m256i*)(base + iv_trace*16 + nsamp*ivec));
        }
        calin::math::simd::avx2_m256_swizzle_u16(samples7 + iv_trace*16);
      }

      base = hg8 + iblock*nsamp*16;
      vp = samples8;
      for(unsigned iv_trace=0; iv_trace<nv_trace; iv_trace++) {
        for(unsigned ivec=0; ivec<16; ivec++) {
          *(vp++) = _mm256_loadu_si256((__m256i*)(base + iv_trace*16 + nsamp*ivec));
        }
        calin::math::simd::avx2_m256_swizzle_u16(samples8 + iv_trace*16);
      }

      for(unsigned isamp=0;isamp<nsamp;isamp++) {
        const __m256i sampi1 = samples1[isamp];
        const __m256i sampi2 = samples2[isamp];
        const __m256i sampi3 = samples3[isamp];
        const __m256i sampi4 = samples4[isamp];
        const __m256i sampi5 = samples5[isamp];
        const __m256i sampi6 = samples6[isamp];
        const __m256i sampi7 = samples7[isamp];
        const __m256i sampi8 = samples8[isamp];

        __m256i prod_lo = _mm256_mullo_epi16(sampi1, sampi1);
        __m256i prod_hi = _mm256_mulhi_epu16(sampi1, sampi1);
        __m256i sum_l = _mm256_unpacklo_epi16(prod_lo, prod_hi);
        __m256i sum_u = _mm256_unpackhi_epi16(prod_lo, prod_hi);

        prod_lo = _mm256_mullo_epi16(sampi2, sampi2);
        prod_hi = _mm256_mulhi_epu16(sampi2, sampi2);
        sum_l = _mm256_add_epi32(sum_l, _mm256_unpacklo_epi16(prod_lo, prod_hi));
        sum_u = _mm256_add_epi32(sum_u, _mm256_unpackhi_epi16(prod_lo, prod_hi));

        prod_lo = _mm256_mullo_epi16(sampi3, sampi3);
        prod_hi = _mm256_mulhi_epu16(sampi3, sampi3);
        sum_l = _mm256_add_epi32(sum_l, _mm256_unpacklo_epi16(prod_lo, prod_hi));
        sum_u = _mm256_add_epi32(sum_u, _mm256_unpackhi_epi16(prod_lo, prod_hi));

        prod_lo = _mm256_mullo_epi16(sampi4, sampi4);
        prod_hi = _mm256_mulhi_epu16(sampi4, sampi4);
        sum_l = _mm256_add_epi32(sum_l, _mm256_unpacklo_epi16(prod_lo, prod_hi));
        sum_u = _mm256_add_epi32(sum_u, _mm256_unpackhi_epi16(prod_lo, prod_hi));

        prod_lo = _mm256_mullo_epi16(sampi5, sampi5);
        prod_hi = _mm256_mulhi_epu16(sampi5, sampi5);
        sum_l = _mm256_add_epi32(sum_l, _mm256_unpacklo_epi16(prod_lo, prod_hi));
        sum_u = _mm256_add_epi32(sum_u, _mm256_unpackhi_epi16(prod_lo, prod_hi));

        prod_lo = _mm256_mullo_epi16(sampi6, sampi6);
        prod_hi = _mm256_mulhi_epu16(sampi6, sampi6);
        sum_l = _mm256_add_epi32(sum_l, _mm256_unpacklo_epi16(prod_lo, prod_hi));
        sum_u = _mm256_add_epi32(sum_u, _mm256_unpackhi_epi16(prod_lo, prod_hi));

        prod_lo = _mm256_mullo_epi16(sampi7, sampi7);
        prod_hi = _mm256_mulhi_epu16(sampi7, sampi7);
        sum_l = _mm256_add_epi32(sum_l, _mm256_unpacklo_epi16(prod_lo, prod_hi));
        sum_u = _mm256_add_epi32(sum_u, _mm256_unpackhi_epi16(prod_lo, prod_hi));

        prod_lo = _mm256_mullo_epi16(sampi8, sampi8);
        prod_hi = _mm256_mulhi_epu16(sampi8, sampi8);
        sum_l = _mm256_add_epi32(sum_l, _mm256_unpacklo_epi16(prod_lo, prod_hi));
        sum_u = _mm256_add_epi32(sum_u, _mm256_unpackhi_epi16(prod_lo, prod_hi));

        *cov_base = _mm256_add_epi32(*cov_base, sum_l);
        cov_base++;
        *cov_base = _mm256_add_epi32(*cov_base, sum_u);
        cov_base++;

        for(unsigned jsamp=isamp+1;jsamp<nsamp;jsamp++) {
          __m256i sampj = samples1[jsamp];
          prod_lo = _mm256_mullo_epi16(sampi1, sampj);
          prod_hi = _mm256_mulhi_epu16(sampi1, sampj);
          sum_l = _mm256_unpacklo_epi16(prod_lo, prod_hi);
          sum_u = _mm256_unpackhi_epi16(prod_lo, prod_hi);

          sampj = samples2[jsamp];
          prod_lo = _mm256_mullo_epi16(sampi2, sampj);
          prod_hi = _mm256_mulhi_epu16(sampi2, sampj);
          sum_l = _mm256_add_epi32(sum_l, _mm256_unpacklo_epi16(prod_lo, prod_hi));
          sum_u = _mm256_add_epi32(sum_u, _mm256_unpackhi_epi16(prod_lo, prod_hi));

          sampj = samples3[jsamp];
          prod_lo = _mm256_mullo_epi16(sampi3, sampj);
          prod_hi = _mm256_mulhi_epu16(sampi3, sampj);
          sum_l = _mm256_add_epi32(sum_l, _mm256_unpacklo_epi16(prod_lo, prod_hi));
          sum_u = _mm256_add_epi32(sum_u, _mm256_unpackhi_epi16(prod_lo, prod_hi));

          sampj = samples4[jsamp];
          prod_lo = _mm256_mullo_epi16(sampi4, sampj);
          prod_hi = _mm256_mulhi_epu16(sampi4, sampj);
          sum_l = _mm256_add_epi32(sum_l, _mm256_unpacklo_epi16(prod_lo, prod_hi));
          sum_u = _mm256_add_epi32(sum_u, _mm256_unpackhi_epi16(prod_lo, prod_hi));

          sampj = samples5[jsamp];
          prod_lo = _mm256_mullo_epi16(sampi5, sampj);
          prod_hi = _mm256_mulhi_epu16(sampi5, sampj);
          sum_l = _mm256_add_epi32(sum_l, _mm256_unpacklo_epi16(prod_lo, prod_hi));
          sum_u = _mm256_add_epi32(sum_u, _mm256_unpackhi_epi16(prod_lo, prod_hi));

          sampj = samples6[jsamp];
          prod_lo = _mm256_mullo_epi16(sampi6, sampj);
          prod_hi = _mm256_mulhi_epu16(sampi6, sampj);
          sum_l = _mm256_add_epi32(sum_l, _mm256_unpacklo_epi16(prod_lo, prod_hi));
          sum_u = _mm256_add_epi32(sum_u, _mm256_unpackhi_epi16(prod_lo, prod_hi));

          sampj = samples7[jsamp];
          prod_lo = _mm256_mullo_epi16(sampi7, sampj);
          prod_hi = _mm256_mulhi_epu16(sampi7, sampj);
          sum_l = _mm256_add_epi32(sum_l, _mm256_unpacklo_epi16(prod_lo, prod_hi));
          sum_u = _mm256_add_epi32(sum_u, _mm256_unpackhi_epi16(prod_lo, prod_hi));

          sampj = samples8[jsamp];
          prod_lo = _mm256_mullo_epi16(sampi8, sampj);
          prod_hi = _mm256_mulhi_epu16(sampi8, sampj);
          sum_l = _mm256_add_epi32(sum_l, _mm256_unpacklo_epi16(prod_lo, prod_hi));
          sum_u = _mm256_add_epi32(sum_u, _mm256_unpackhi_epi16(prod_lo, prod_hi));

          *cov_base = _mm256_add_epi32(*cov_base, sum_l);
          cov_base++;
          *cov_base = _mm256_add_epi32(*cov_base, sum_u);
          cov_base++;
        }
      }
    }
  }
  for(unsigned ichan=0;ichan<20;ichan++) {
    unsigned iblock = ichan/16;
    unsigned irem = ichan%16;
    for(unsigned isamp=0;isamp<5;isamp++)
      std::cout << ' ' << cov[iblock*16*nsamp*(nsamp+1)/2 + isamp*16 + irem];
    std::cout << '\n';
  }

  free(hg1);
  free(hg2);
  free(hg3);
  free(hg4);
  free(hg5);
  free(hg6);
  free(hg7);
  free(hg8);
  free(samples1);
  free(samples2);
  free(samples3);
  free(samples4);
  free(samples5);
  free(samples6);
  free(samples7);
  free(samples8);
}





class NEvent : public ::testing::TestWithParam<unsigned>
{
  // nothing to see here
};

TEST_P(NEvent, AVX2)
{
  NR3_AVX2_RNGCore core(12345);
  uint16_t* hg[GetParam()];
  __m256i* samples[GetParam()];

  std::fill(cov, cov+nchan*nsamp*(nsamp+1)/2, 0);
  const unsigned nv_trace = (nsamp+15)/16;
  const unsigned nv_block = nv_trace*16;
  const unsigned nblock = nchan/16;

  for(unsigned ievent=0; ievent<GetParam(); ievent++) {
    calin::util::memory::safe_aligned_calloc(hg[ievent], nchan*nsamp);
    calin::util::memory::safe_aligned_calloc(samples[ievent], nv_block);
  }

  const __m256i mask_12bit = _mm256_set1_epi16((1<<12)-1);
  for(unsigned iloop=0;iloop<NSIM_TRACECOV;iloop+=GetParam())
  {
    for(unsigned ievent=0; ievent<GetParam(); ievent++) {
      for(unsigned i=0;i<nchan*nsamp/16;i++) {
        __m256i x = _mm256_and_si256(core.uniform_uivec256(), mask_12bit);
        _mm256_storeu_si256((__m256i*)(hg[ievent]+i*16), x);
      }
    }

    __m256i*__restrict__ cov_base = (__m256i*__restrict__)cov;
    for(unsigned iblock=0;iblock<nblock;iblock++)
    {
      for(unsigned ievent=0; ievent<GetParam(); ievent++)
      {
        uint16_t* base = hg[ievent] + iblock*nsamp*16;
        __m256i* vp = samples[ievent];
        for(unsigned iv_trace=0; iv_trace<nv_trace; iv_trace++) {
          for(unsigned ivec=0; ivec<16; ivec++) {
            *(vp++) = _mm256_loadu_si256((__m256i*)(base + iv_trace*16 + nsamp*ivec));
          }
          calin::math::simd::avx2_m256_swizzle_u16(samples[ievent] + iv_trace*16);
        }
      }

      if(GetParam() == 1)
      {
        for(unsigned isamp=0;isamp<nsamp;isamp++)
        {
          for(unsigned ievent=0; ievent<GetParam(); ievent++)
          {
            const __m256i sampi = samples[ievent][isamp];
            __m256i*__restrict__ cov_element = (__m256i*__restrict__)cov_base;
            for(unsigned jsamp=isamp;jsamp<nsamp;jsamp++) {
              __m256i sampj = samples[ievent][jsamp];
              __m256i prod_lo = _mm256_mullo_epi16(sampi, sampj);
              __m256i prod_hi = _mm256_mulhi_epu16(sampi, sampj);
              *cov_element = _mm256_add_epi32(*cov_element, _mm256_unpacklo_epi16(prod_lo, prod_hi));
              cov_element++;
              *cov_element = _mm256_add_epi32(*cov_element, _mm256_unpackhi_epi16(prod_lo, prod_hi));
              cov_element++;
            }
          }
          cov_base += 2*(nsamp-isamp);
        }
      }
      else if(GetParam() == 2)
      {
        for(unsigned isamp=0;isamp<nsamp;isamp++)
        {
          for(unsigned ievent=0; ievent<GetParam(); ievent+=2)
          {
            const __m256i sampi1 = samples[ievent][isamp];
            const __m256i sampi2 = samples[ievent+1][isamp];
            __m256i*__restrict__ cov_element = (__m256i*__restrict__)cov_base;
            for(unsigned jsamp=isamp;jsamp<nsamp;jsamp++) {
              __m256i sampj = samples[ievent][jsamp];
              __m256i prod_lo = _mm256_mullo_epi16(sampi1, sampj);
              __m256i prod_hi = _mm256_mulhi_epu16(sampi1, sampj);
              *cov_element = _mm256_add_epi32(*cov_element, _mm256_unpacklo_epi16(prod_lo, prod_hi));
              *(cov_element+1) = _mm256_add_epi32(*(cov_element+1), _mm256_unpackhi_epi16(prod_lo, prod_hi));

              sampj = samples[ievent+1][jsamp];
              prod_lo = _mm256_mullo_epi16(sampi2, sampj);
              prod_hi = _mm256_mulhi_epu16(sampi2, sampj);
              *cov_element = _mm256_add_epi32(*cov_element, _mm256_unpacklo_epi16(prod_lo, prod_hi));
              *(cov_element+1) = _mm256_add_epi32(*(cov_element+1), _mm256_unpackhi_epi16(prod_lo, prod_hi));

              cov_element += 2;
            }
          }
          cov_base += 2*(nsamp-isamp);
        }
      }
      else
      {
        for(unsigned isamp=0;isamp<nsamp;isamp++)
        {
          for(unsigned ievent=0; ievent<GetParam(); ievent+=4)
          {
            const __m256i sampi1 = samples[ievent][isamp];
            const __m256i sampi2 = samples[ievent+1][isamp];
            const __m256i sampi3 = samples[ievent+2][isamp];
            const __m256i sampi4 = samples[ievent+3][isamp];
            __m256i*__restrict__ cov_element = (__m256i*__restrict__)cov_base;
            for(unsigned jsamp=isamp;jsamp<nsamp;jsamp++) {
              __m256i sampj = samples[ievent][jsamp];
              __m256i prod_lo = _mm256_mullo_epi16(sampi1, sampj);
              __m256i prod_hi = _mm256_mulhi_epu16(sampi1, sampj);
              *cov_element = _mm256_add_epi32(*cov_element, _mm256_unpacklo_epi16(prod_lo, prod_hi));
              *(cov_element+1) = _mm256_add_epi32(*(cov_element+1), _mm256_unpackhi_epi16(prod_lo, prod_hi));

              sampj = samples[ievent+1][jsamp];
              prod_lo = _mm256_mullo_epi16(sampi2, sampj);
              prod_hi = _mm256_mulhi_epu16(sampi2, sampj);
              *cov_element = _mm256_add_epi32(*cov_element, _mm256_unpacklo_epi16(prod_lo, prod_hi));
              *(cov_element+1) = _mm256_add_epi32(*(cov_element+1), _mm256_unpackhi_epi16(prod_lo, prod_hi));

              sampj = samples[ievent+2][jsamp];
              prod_lo = _mm256_mullo_epi16(sampi3, sampj);
              prod_hi = _mm256_mulhi_epu16(sampi3, sampj);
              *cov_element = _mm256_add_epi32(*cov_element, _mm256_unpacklo_epi16(prod_lo, prod_hi));
              *(cov_element+1) = _mm256_add_epi32(*(cov_element+1), _mm256_unpackhi_epi16(prod_lo, prod_hi));

              sampj = samples[ievent+3][jsamp];
              prod_lo = _mm256_mullo_epi16(sampi4, sampj);
              prod_hi = _mm256_mulhi_epu16(sampi4, sampj);
              *cov_element = _mm256_add_epi32(*cov_element, _mm256_unpacklo_epi16(prod_lo, prod_hi));
              *(cov_element+1) = _mm256_add_epi32(*(cov_element+1), _mm256_unpackhi_epi16(prod_lo, prod_hi));

              cov_element += 2;
            }
          }
          cov_base += 2*(nsamp-isamp);
        }

      }
    }
  }
  for(unsigned ichan=0;ichan<20;ichan++) {
    unsigned iblock = ichan/16;
    unsigned irem = ichan%16;
    for(unsigned isamp=0;isamp<5;isamp++)
      std::cout << ' ' << cov[iblock*16*nsamp*(nsamp+1)/2 + isamp*16 + irem];
    std::cout << '\n';
  }
  for(unsigned ievent=0; ievent<GetParam(); ievent++) {
    free(hg[ievent]);
    free(samples[ievent]);
  }
}

INSTANTIATE_TEST_CASE_P(TestTraceCov,
                        NEvent,
                        ::testing::Values(1,2,4,8));


#endif // defined CALIN_HAS_NR3_AVX2_RNGCORE

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}