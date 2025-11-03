/*

   calin/math/fftw_codelet_container.hpp -- Stephen Fegan -- 2025-10-19

   Container for FFTW codelets

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

namespace calin::math::fftw_codelet_container {

  template<typename VCLReal> class alignas(VCLReal::vec_bytes) FFTWCodelet_Container
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
      ADD_CASE(128);
      ADD_CASE(256);
      ADD_CASE(512);
      ADD_CASE(1024);
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
      ADD_CASE(128);
      ADD_CASE(256);
      ADD_CASE(512);
      ADD_CASE(1024);
  #undef ADD_CASE
      default:
        throw std::runtime_error("No FFT codelet available for array size : " + std::to_string(size));
      }
    }

    void hc2r(unsigned size, const float_type* c, float_type* r) {
      switch(size) {
  #define ADD_CASE(n) case n : return hc2r_##n(c,r)
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
      ADD_CASE(128);
      ADD_CASE(256);
      ADD_CASE(512);
      ADD_CASE(1024);
  #undef ADD_CASE
      default:
        throw std::runtime_error("No FFT codelet available for array size : " + std::to_string(size));
      }
    }

  #define MAKE_R2HC(n) \
    inline void r2hc_##n(const float_type* r, float_type* c) { \
      dft_codelet_r2hc_##n(r, r+1, c, c+n); \
    }

  #define MAKE_HC2R(n) \
    inline void hc2r_##n(const float_type* c, float_type* r) { \
      dft_codelet_hc2r_##n(r, r+1, c, c+n); \
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
    MAKE_R2HC(128);
    MAKE_HC2R(128);
    MAKE_R2HC(256);
    MAKE_HC2R(256);
    MAKE_R2HC(512);
    MAKE_HC2R(512);
    MAKE_R2HC(1024);
    MAKE_HC2R(1024);

  private:
    using E                 = typename VCLReal::real_vt;
    using R                 = typename VCLReal::real_vt;
    using INT               = int;
    using stride            = int;

    inline int WSR(const stride i) { return 2*i; }
    inline int WSCR(stride i) { return i; }
    inline int WSCI(stride i) { return -i; }

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

  #include "../../src/math/genfft_codelets/dft_r2hc_8.c"
  #include "../../src/math/genfft_codelets/dft_hc2r_8.c"
  #include "../../src/math/genfft_codelets/dft_r2hc_12.c"
  #include "../../src/math/genfft_codelets/dft_hc2r_12.c"
  #include "../../src/math/genfft_codelets/dft_r2hc_15.c"
  #include "../../src/math/genfft_codelets/dft_hc2r_15.c"
  #include "../../src/math/genfft_codelets/dft_r2hc_16.c"
  #include "../../src/math/genfft_codelets/dft_hc2r_16.c"
  #include "../../src/math/genfft_codelets/dft_r2hc_18.c"
  #include "../../src/math/genfft_codelets/dft_hc2r_18.c"
  #include "../../src/math/genfft_codelets/dft_r2hc_20.c"
  #include "../../src/math/genfft_codelets/dft_hc2r_20.c"
  #include "../../src/math/genfft_codelets/dft_r2hc_24.c"
  #include "../../src/math/genfft_codelets/dft_hc2r_24.c"
  #include "../../src/math/genfft_codelets/dft_r2hc_28.c"
  #include "../../src/math/genfft_codelets/dft_hc2r_28.c"
  #include "../../src/math/genfft_codelets/dft_r2hc_30.c"
  #include "../../src/math/genfft_codelets/dft_hc2r_30.c"
  #include "../../src/math/genfft_codelets/dft_r2hc_32.c"
  #include "../../src/math/genfft_codelets/dft_hc2r_32.c"
  #include "../../src/math/genfft_codelets/dft_r2hc_36.c"
  #include "../../src/math/genfft_codelets/dft_hc2r_36.c"
  #include "../../src/math/genfft_codelets/dft_r2hc_40.c"
  #include "../../src/math/genfft_codelets/dft_hc2r_40.c"
  #include "../../src/math/genfft_codelets/dft_r2hc_48.c"
  #include "../../src/math/genfft_codelets/dft_hc2r_48.c"
  #include "../../src/math/genfft_codelets/dft_r2hc_56.c"
  #include "../../src/math/genfft_codelets/dft_hc2r_56.c"
  #include "../../src/math/genfft_codelets/dft_r2hc_60.c"
  #include "../../src/math/genfft_codelets/dft_hc2r_60.c"
  #include "../../src/math/genfft_codelets/dft_r2hc_64.c"
  #include "../../src/math/genfft_codelets/dft_hc2r_64.c"
  #include "../../src/math/genfft_codelets/dft_r2hc_128.c"
  #include "../../src/math/genfft_codelets/dft_hc2r_128.c"
  #include "../../src/math/genfft_codelets/dft_r2hc_256.c"
  #include "../../src/math/genfft_codelets/dft_hc2r_256.c"
  #include "../../src/math/genfft_codelets/dft_r2hc_512.c"
  #include "../../src/math/genfft_codelets/dft_hc2r_512.c"
  #include "../../src/math/genfft_codelets/dft_r2hc_1024.c"
  #include "../../src/math/genfft_codelets/dft_hc2r_1024.c"
  };

#undef DK
#undef MAKE_R2HC
#undef MAKE_HC2R
} // namespace calin::math::fftw_codelet_container

template<typename VCLReal> bool 
calin::math::fftw_util::FFTWCodelet<VCLReal>::has_codelet(unsigned size)
{
  calin::math::fftw_codelet_container::FFTWCodelet_Container<VCLReal> fftwcc;
  return fftwcc.has_codelet(size);
}

template<typename VCLReal> void 
calin::math::fftw_util::FFTWCodelet<VCLReal>::r2hc(unsigned size, const float_type* r, float_type* c)
{
  calin::math::fftw_codelet_container::FFTWCodelet_Container<VCLReal> fftwcc;
  return fftwcc.r2hc(size, r, c);
}

template<typename VCLReal> void 
calin::math::fftw_util::FFTWCodelet<VCLReal>::hc2r(unsigned size, const float_type* c, float_type* r)
{
  calin::math::fftw_codelet_container::FFTWCodelet_Container<VCLReal> fftwcc;
  return fftwcc.hc2r(size, c, r);
}
