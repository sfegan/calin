/*

   calin/util/vcl.hpp -- Stephen Fegan -- 2018-08-08

   Calin interface to Agner Fog's Vector Class Library (VCL)

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

#include <ostream>
#include <string>
#include <stdlib.h>

#define MAX_VECTOR_SIZE 512
#define VCL_NAMESPACE vcl

#include <VCL/vectorclass.h>
#include <VCL/vectormath_trig.h>
#include <VCL/vectormath_exp.h>
#include <VCL/vectormath_hyp.h>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>

namespace Eigen {

template<> struct NumTraits<::vcl::Vec4f>: NumTraits<float>
{
  typedef ::vcl::Vec4f Real;
  typedef ::vcl::Vec4f NonInteger;
  typedef ::vcl::Vec4f Nested;
  typedef float Literal;
  enum {
    IsComplex = 0,
    IsInteger = 0,
    IsSigned = 1,
    RequireInitialization = 0,
    ReadCost = 1,
    AddCost = 3,
    MulCost = 3
  };
};

template<> struct NumTraits<::vcl::Vec8f>: NumTraits<float>
{
  typedef ::vcl::Vec8f Real;
  typedef ::vcl::Vec8f NonInteger;
  typedef ::vcl::Vec8f Nested;
  typedef float Literal;
  enum {
    IsComplex = 0,
    IsInteger = 0,
    IsSigned = 1,
    RequireInitialization = 0,
    ReadCost = 1,
    AddCost = 3,
    MulCost = 3
  };
};

template<> struct NumTraits<::vcl::Vec16f>: NumTraits<float>
{
  typedef ::vcl::Vec16f Real;
  typedef ::vcl::Vec16f NonInteger;
  typedef ::vcl::Vec16f Nested;
  typedef float Literal;
  enum {
    IsComplex = 0,
    IsInteger = 0,
    IsSigned = 1,
    RequireInitialization = 0,
    ReadCost = 1,
    AddCost = 3,
    MulCost = 3
  };
};

template<> struct NumTraits<::vcl::Vec2d>: NumTraits<double>
{
  typedef ::vcl::Vec2d Real;
  typedef ::vcl::Vec2d NonInteger;
  typedef ::vcl::Vec2d Nested;
  typedef double Literal;
  enum {
    IsComplex = 0,
    IsInteger = 0,
    IsSigned = 1,
    RequireInitialization = 0,
    ReadCost = 1,
    AddCost = 3,
    MulCost = 3
  };
};

template<> struct NumTraits<::vcl::Vec4d>: NumTraits<double>
{
  typedef ::vcl::Vec4d Real;
  typedef ::vcl::Vec4d NonInteger;
  typedef ::vcl::Vec4d Nested;
  typedef double Literal;
  enum {
    IsComplex = 0,
    IsInteger = 0,
    IsSigned = 1,
    RequireInitialization = 0,
    ReadCost = 1,
    AddCost = 3,
    MulCost = 3
  };
};

template<> struct NumTraits<::vcl::Vec8d>: NumTraits<double>
{
  typedef ::vcl::Vec8d Real;
  typedef ::vcl::Vec8d NonInteger;
  typedef ::vcl::Vec8d Nested;
  typedef double Literal;
  enum {
    IsComplex = 0,
    IsInteger = 0,
    IsSigned = 1,
    RequireInitialization = 0,
    ReadCost = 1,
    AddCost = 3,
    MulCost = 3
  };
};

typedef Eigen::Matrix< ::vcl::Vec4f , 3 , 1> Vector3_4f;
typedef Eigen::Matrix< ::vcl::Vec4f , 3 , 3> Matrix3_4f;
typedef Eigen::Matrix< ::vcl::Vec8f , 3 , 1> Vector3_8f;
typedef Eigen::Matrix< ::vcl::Vec8f , 3 , 3> Matrix3_8f;
typedef Eigen::Matrix< ::vcl::Vec16f , 3 , 1> Vector3_16f;
typedef Eigen::Matrix< ::vcl::Vec16f , 3 , 3> Matrix3_16f;

typedef Eigen::Matrix< ::vcl::Vec2d , 3 , 1> Vector3_2d;
typedef Eigen::Matrix< ::vcl::Vec2d , 3 , 3> Matrix3_2d;
typedef Eigen::Matrix< ::vcl::Vec4d , 3 , 1> Vector3_4d;
typedef Eigen::Matrix< ::vcl::Vec4d , 3 , 3> Matrix3_4d;
typedef Eigen::Matrix< ::vcl::Vec8d , 3 , 1> Vector3_8d;
typedef Eigen::Matrix< ::vcl::Vec8d , 3 , 3> Matrix3_8d;

typedef Eigen::Matrix< ::vcl::Vec4f , 4 , 1> Vector4_4f;
typedef Eigen::Matrix< ::vcl::Vec4f , 4 , 4> Matrix4_4f;
typedef Eigen::Matrix< ::vcl::Vec8f , 4 , 1> Vector4_8f;
typedef Eigen::Matrix< ::vcl::Vec8f , 4 , 4> Matrix4_8f;
typedef Eigen::Matrix< ::vcl::Vec16f , 4 , 1> Vector4_16f;
typedef Eigen::Matrix< ::vcl::Vec16f , 4 , 4> Matrix4_16f;

typedef Eigen::Matrix< ::vcl::Vec2d , 4 , 1> Vector4_2d;
typedef Eigen::Matrix< ::vcl::Vec2d , 4 , 4> Matrix4_2d;
typedef Eigen::Matrix< ::vcl::Vec4d , 4 , 1> Vector4_4d;
typedef Eigen::Matrix< ::vcl::Vec4d , 4 , 4> Matrix4_4d;
typedef Eigen::Matrix< ::vcl::Vec8d , 4 , 1> Vector4_8d;
typedef Eigen::Matrix< ::vcl::Vec8d , 4 , 4> Matrix4_8d;

} // namespace Eigen

namespace vcl {

  static inline Vec4d to_double_low(Vec8i const & a) {
    return to_double(a.get_low());
  }

  static inline Vec4d to_double_high(Vec8i const & a) {
    return to_double(a.get_high());
  }

  static inline Vec8d to_double_low(Vec16i const & a) {
    return to_double(a.get_low());
  }

  static inline Vec8d to_double_high(Vec16i const & a) {
    return to_double(a.get_high());
  }

  static inline Vec8i truncate_to_int(Vec4d const & a, Vec4d const & b) {
    Vec4i t1 = truncate_to_int(a);
    Vec4i t2 = truncate_to_int(b);
    return Vec8i(t1,t2);
  }

  static inline Vec16i truncate_to_int(Vec8d const & a, Vec8d const & b) {
    Vec8i t1 = truncate_to_int(a);
    Vec8i t2 = truncate_to_int(b);
    return Vec16i(t1,t2);
  }

  // function truncate_to_int64_limited: round towards zero. (inefficient)
  // result as 64-bit integer vector, but with limited range. Deprecated!
  static inline Vec2q truncate_to_int64_limited(Vec2d const & a) {
  #if defined (__AVX512DQ__) && defined (__AVX512VL__)
      return truncate_to_int64(a);
  #else
      // Note: assume MXCSR control register is set to rounding
      Vec4i t1 = _mm_cvttpd_epi32(a);
      return extend_low(t1);
  #endif
  }

  // function round_to_int: round to nearest integer (even)
  // result as 64-bit integer vector, but with limited range. Deprecated!
  static inline Vec2q round_to_int64_limited(Vec2d const & a) {
  #if defined (__AVX512DQ__) && defined (__AVX512VL__)
      return round_to_int64(a);
  #else
      // Note: assume MXCSR control register is set to rounding
      Vec4i t1 = _mm_cvtpd_epi32(a);
      return extend_low(t1);
  #endif
  }
  // function to_double_limited: convert integer vector elements to double vector
  // limited to abs(x) < 2^31. Deprecated!
  static inline Vec2d to_double_limited(Vec2q const & x) {
  #if defined (__AVX512DQ__) && defined (__AVX512VL__)
      return to_double(x);
  #else
      Vec4i compressed = permute4i<0,2,-256,-256>(Vec4i(x));
      return _mm_cvtepi32_pd(compressed);
  #endif
  }

#if MAX_VECTOR_SIZE >= 256
#if INSTRSET >= 7
  // function truncate_to_int64_limited: round towards zero.
  // result as 64-bit integer vector, but with limited range. Deprecated!
  static inline Vec4q truncate_to_int64_limited(Vec4d const & a) {
  #if defined (__AVX512DQ__) && defined (__AVX512VL__)
      return truncate_to_int64(a);
  #elif VECTORI256_H > 1
      // Note: assume MXCSR control register is set to rounding
      Vec2q   b = _mm256_cvttpd_epi32(a);                    // round to 32-bit integers
      __m256i c = permute4q<0,-256,1,-256>(Vec4q(b,b));      // get bits 64-127 to position 128-191
      __m256i s = _mm256_srai_epi32(c, 31);                  // sign extension bits
      return      _mm256_unpacklo_epi32(c, s);               // interleave with sign extensions
  #else
      return Vec4q(truncate_to_int64_limited(a.get_low()), truncate_to_int64_limited(a.get_high()));
  #endif
  }
  // function round_to_int64_limited: round to nearest integer (even)
  // result as 64-bit integer vector, but with limited range. Deprecated!
  static inline Vec4q round_to_int64_limited(Vec4d const & a) {
  #if defined (__AVX512DQ__) && defined (__AVX512VL__)
      return round_to_int64(a);
  #elif VECTORI256_H > 1
      // Note: assume MXCSR control register is set to rounding
      Vec2q   b = _mm256_cvtpd_epi32(a);                     // round to 32-bit integers
      __m256i c = permute4q<0,-256,1,-256>(Vec4q(b,b));      // get bits 64-127 to position 128-191
      __m256i s = _mm256_srai_epi32(c, 31);                  // sign extension bits
      return      _mm256_unpacklo_epi32(c, s);               // interleave with sign extensions
  #else
      return Vec4q(round_to_int64_limited(a.get_low()), round_to_int64_limited(a.get_high()));
  #endif
  }
  // function to_double_limited: convert integer vector elements to double vector
  // limited to abs(x) < 2^31. Deprecated!
  static inline Vec4d to_double_limited(Vec4q const & x) {
  #if defined (__AVX512DQ__) && defined (__AVX512VL__)
      return to_double(x);
  #else
      Vec8i compressed = permute8i<0,2,4,6,-256,-256,-256,-256>(Vec8i(x));
      return _mm256_cvtepi32_pd(compressed.get_low());  // AVX
  #endif
  }
#else // if INSTRSET >= 7
  static inline Vec4q truncate_to_int64_limited(Vec4d const & a) {
    return Vec4q(truncate_to_int64_limited(a.get_low()),
                 truncate_to_int64_limited(a.get_high()));
  }
  static inline Vec4q round_to_int64_limited(Vec4d const & a) {
    return Vec4q(round_to_int64_limited(a.get_low()),
                 round_to_int64_limited(a.get_high()));
  }
  static inline Vec4d to_double_limited(Vec4q const & x) {
    return Vec4d(to_double_limited(x.get_low()),
                 to_double_limited(x.get_high()));
  }
#endif // if INSTRSET >= 7
#endif // if MAX_VECTOR_SIZE >= 256

#if MAX_VECTOR_SIZE >= 512
#if INSTRSET >= 9
  // function truncate_to_int64_limited: round towards zero.
  // result as 64-bit integer vector, but with limited range. Deprecated!
  static inline Vec8q truncate_to_int64_limited(Vec8d const & a) {
  #ifdef __AVX512DQ__
      return truncate_to_int64(a);
  #else
      // Note: assume MXCSR control register is set to rounding
      Vec4q   b = _mm512_cvttpd_epi32(a);                    // round to 32-bit integers
      __m512i c = permute8q<0,-256,1,-256,2,-256,3,-256>(Vec8q(b,b));      // get bits 64-127 to position 128-191, etc.
      __m512i s = _mm512_srai_epi32(c, 31);                  // sign extension bits
      return      _mm512_unpacklo_epi32(c, s);               // interleave with sign extensions
  #endif
  }
  // function round_to_int64_limited: round to nearest integer (even)
  // result as 64-bit integer vector, but with limited range. Deprecated!
  static inline Vec8q round_to_int64_limited(Vec8d const & a) {
  #ifdef __AVX512DQ__
      return round_to_int64(a);
  #else
      Vec4q   b = _mm512_cvt_roundpd_epi32(a, 0+8);     // round to 32-bit integers
      __m512i c = permute8q<0,-256,1,-256,2,-256,3,-256>(Vec8q(b,b));  // get bits 64-127 to position 128-191, etc.
      __m512i s = _mm512_srai_epi32(c, 31);                            // sign extension bits
      return      _mm512_unpacklo_epi32(c, s);                         // interleave with sign extensions
  #endif
  }
  // function to_double_limited: convert integer vector elements to double vector
  // limited to abs(x) < 2^31. Deprecated!
  static inline Vec8d to_double_limited(Vec8q const & x) {
  #if defined (__AVX512DQ__)
      return to_double(x);
  #else
      Vec16i compressed = permute16i<0,2,4,6,8,10,12,14,-256,-256,-256,-256,-256,-256,-256,-256>(Vec16i(x));
      return _mm512_cvtepi32_pd(compressed.get_low());
  #endif
  }
#else // if INSTRSET >= 9
  static inline Vec8q truncate_to_int64_limited(Vec8d const & a) {
    return Vec8q(truncate_to_int64_limited(a.get_low()),
                 truncate_to_int64_limited(a.get_high()));
  }
  static inline Vec8q round_to_int64_limited(Vec8d const & a) {
    return Vec8q(round_to_int64_limited(a.get_low()),
                 round_to_int64_limited(a.get_high()));
  }
  static inline Vec8d to_double_limited(Vec8q const & x) {
    return Vec8d(to_double_limited(x.get_low()),
                 to_double_limited(x.get_high()));
  }
#endif // if INSTRSET >= 9
#endif // if MAX_VECTOR_SIZE >= 512
}

namespace calin { namespace util { namespace vcl {

using namespace ::vcl;

template<typename VCLArchitecture> struct VCLFloatReal
{
  constexpr static unsigned vec_bits              = VCLArchitecture::vec_bits;
  constexpr static unsigned vec_bytes             = VCLArchitecture::vec_bytes;
  constexpr static unsigned num_real              = VCLArchitecture::num_float;
  typedef VCLArchitecture                         architecture;

  typedef int32_t                                 int_t;
  typedef uint32_t                                uint_t;
  typedef float                                   real_t;
  typedef Eigen::Vector3f                         vec3_t;
  typedef Eigen::Matrix3f                         mat3_t;
  typedef Eigen::VectorXf                         vecX_t;
  typedef Eigen::MatrixXf                         matX_t;

  typedef typename VCLArchitecture::int32_at      int_at;
  typedef typename VCLArchitecture::uint32_at     uint_at;
  typedef typename VCLArchitecture::float_at      real_at;

  typedef typename VCLArchitecture::int32_vt      int_vt;
  typedef typename VCLArchitecture::uint32_vt     uint_vt;
  typedef typename VCLArchitecture::int32_bvt     bool_int_vt;
  typedef typename VCLArchitecture::uint32_bvt    bool_uint_vt;

  typedef typename VCLArchitecture::float_vt      real_vt;
  typedef typename VCLArchitecture::float_bvt     bool_vt;

  typedef typename VCLArchitecture::Vector3f_vt   vec3_vt;
  typedef typename VCLArchitecture::Matrix3f_vt   mat3_vt;

  typedef typename VCLArchitecture::Vector4f_vt   vec4_vt;
  typedef typename VCLArchitecture::Matrix4f_vt   mat4_vt;

  static inline int_vt truncate_to_int_limited(real_vt x) {
    return vcl::truncate_to_int(x);
  }

  static inline int_vt round_to_int_limited(real_vt x) {
    return vcl::round_to_int(x);
  }

  static inline int_vt truncate_to_int(real_vt x) {
    return vcl::truncate_to_int(x);
  }

  static inline int_vt round_to_int(real_vt x) {
    return vcl::round_to_int(x);
  }

  static real_vt iota() { return VCLArchitecture::float_iota(); }

  static inline void* aligned_malloc(size_t nbytes) {
    return VCLArchitecture::aligned_malloc(nbytes);
  }
  static inline void aligned_free(void* p) {
    VCLArchitecture::aligned_free(p);
  }
};

template<typename VCLArchitecture> struct VCLDoubleReal
{
  constexpr static unsigned vec_bits              = VCLArchitecture::vec_bits;
  constexpr static unsigned vec_bytes             = VCLArchitecture::vec_bytes;
  constexpr static unsigned num_real              = VCLArchitecture::num_double;
  typedef VCLArchitecture                         architecture;

  typedef int64_t                                 int_t;
  typedef uint64_t                                uint_t;
  typedef double                                  real_t;
  typedef Eigen::Vector3d                         vec3_t;
  typedef Eigen::Matrix3d                         mat3_t;
  typedef Eigen::VectorXd                         vecX_t;
  typedef Eigen::MatrixXd                         matX_t;

  typedef typename VCLArchitecture::int64_at      int_at;
  typedef typename VCLArchitecture::uint64_at     uint_at;
  typedef typename VCLArchitecture::double_at     real_at;

  typedef typename VCLArchitecture::int64_vt      int_vt;
  typedef typename VCLArchitecture::uint64_vt     uint_vt;
  typedef typename VCLArchitecture::int64_bvt     bool_int_vt;
  typedef typename VCLArchitecture::uint64_bvt    bool_uint_vt;

  typedef typename VCLArchitecture::double_vt     real_vt;
  typedef typename VCLArchitecture::double_bvt    bool_vt;

  typedef typename VCLArchitecture::Vector3d_vt   vec3_vt;
  typedef typename VCLArchitecture::Matrix3d_vt   mat3_vt;

  typedef typename VCLArchitecture::Vector4d_vt   vec4_vt;
  typedef typename VCLArchitecture::Matrix4d_vt   mat4_vt;

  static inline int_vt truncate_to_int_limited(real_vt x) {
    return vcl::truncate_to_int64_limited(x);
  }

  static inline int_vt round_to_int_limited(real_vt x) {
    return vcl::round_to_int64_limited(x);
  }

  static inline int_vt truncate_to_int(real_vt x) {
    return vcl::truncate_to_int64(x);
  }

  static inline int_vt round_to_int(real_vt x) {
    return vcl::round_to_int64(x);
  }

  static real_vt iota() { return VCLArchitecture::double_iota(); }

  static inline void* aligned_malloc(size_t nbytes) {
    return VCLArchitecture::aligned_malloc(nbytes);
  }
  static inline void aligned_free(void* p) {
    VCLArchitecture::aligned_free(p);
  }
};

struct VCL128Architecture
{
  constexpr static char architecture_name[] = "VCL128Architecture";

  constexpr static unsigned vec_bits   = 128;
  constexpr static unsigned vec_bytes  = vec_bits/8;
  constexpr static unsigned num_int8   = vec_bytes/sizeof(int8_t);
  constexpr static unsigned num_uint8  = vec_bytes/sizeof(uint8_t);
  constexpr static unsigned num_int16  = vec_bytes/sizeof(int16_t);
  constexpr static unsigned num_uint16 = vec_bytes/sizeof(uint16_t);
  constexpr static unsigned num_int32  = vec_bytes/sizeof(int32_t);
  constexpr static unsigned num_uint32 = vec_bytes/sizeof(uint32_t);
  constexpr static unsigned num_int64  = vec_bytes/sizeof(int64_t);
  constexpr static unsigned num_uint64 = vec_bytes/sizeof(uint64_t);
  constexpr static unsigned num_float  = vec_bytes/sizeof(float);
  constexpr static unsigned num_double = vec_bytes/sizeof(double);

  typedef Vec128b bool_vt;
  typedef Vec16c  int8_vt;
  typedef Vec16uc uint8_vt;
  typedef Vec8s   int16_vt;
  typedef Vec8us  uint16_vt;
  typedef Vec4i   int32_vt;
  typedef Vec4ui  uint32_vt;
  typedef Vec2q   int64_vt;
  typedef Vec2uq  uint64_vt;
  typedef Vec4f   float_vt;
  typedef Vec2d   double_vt;

  typedef Vec16cb int8_bvt;
  typedef Vec16cb uint8_bvt;
  typedef Vec8sb  int16_bvt;
  typedef Vec8sb  uint16_bvt;
  typedef Vec4ib  int32_bvt;
  typedef Vec4ib  uint32_bvt;
  typedef Vec2qb  int64_bvt;
  typedef Vec2qb  uint64_bvt;
  typedef Vec4fb  float_bvt;
  typedef Vec2db  double_bvt;

  typedef int32_t  int32_at[num_int32] __attribute((aligned(16)));
  typedef uint32_t uint32_at[num_int32] __attribute((aligned(16)));
  typedef int64_t  int64_at[num_int64] __attribute((aligned(16)));
  typedef uint64_t uint64_at[num_int64] __attribute((aligned(16)));
  typedef float    float_at[num_float] __attribute((aligned(16)));
  typedef double   double_at[num_double] __attribute((aligned(16)));

  typedef Eigen::Vector3_4f Vector3f_vt;
  typedef Eigen::Matrix3_4f Matrix3f_vt;

  typedef Eigen::Vector3_2d  Vector3d_vt;
  typedef Eigen::Matrix3_2d  Matrix3d_vt;

  typedef Eigen::Vector4_4f Vector4f_vt;
  typedef Eigen::Matrix4_4f Matrix4f_vt;

  typedef Eigen::Vector4_2d  Vector4d_vt;
  typedef Eigen::Matrix4_2d  Matrix4d_vt;

  typedef VCLFloatReal<VCL128Architecture> float_real;
  typedef VCLDoubleReal<VCL128Architecture> double_real;

  static inline float_vt float_iota() {
    return float_vt(0.0f, 1.0f, 2.0f, 3.0f); }
  static inline double_vt double_iota() {
    return double_vt(0.0, 1.0); }
  static inline int32_vt int32_iota() {
    return int32_vt(0,1,2,3); }
  static inline int64_vt int64_iota() {
    return int64_vt(0,1); }
  static inline void* aligned_malloc(size_t nbytes) {
    void* p = nullptr;
    if(::posix_memalign(&p, vec_bytes, nbytes)==0) {
      return p;
    }
    throw std::bad_alloc();
  };
  static inline void aligned_free(void* p) {
    ::free(p);
  }
};

struct VCL256Architecture
{
  constexpr static char architecture_name[] = "VCL256Architecture";

  constexpr static unsigned vec_bits   = 256;
  constexpr static unsigned vec_bytes  = vec_bits/8;
  constexpr static unsigned num_int8   = vec_bytes/sizeof(int8_t);
  constexpr static unsigned num_uint8  = vec_bytes/sizeof(uint8_t);
  constexpr static unsigned num_int16  = vec_bytes/sizeof(int16_t);
  constexpr static unsigned num_uint16 = vec_bytes/sizeof(uint16_t);
  constexpr static unsigned num_int32  = vec_bytes/sizeof(int32_t);
  constexpr static unsigned num_uint32 = vec_bytes/sizeof(uint32_t);
  constexpr static unsigned num_int64  = vec_bytes/sizeof(int64_t);
  constexpr static unsigned num_uint64 = vec_bytes/sizeof(uint64_t);
  constexpr static unsigned num_float  = vec_bytes/sizeof(float);
  constexpr static unsigned num_double = vec_bytes/sizeof(double);

  typedef Vec256b bool_vt;
  typedef Vec32c  int8_vt;
  typedef Vec32uc uint8_vt;
  typedef Vec16s  int16_vt;
  typedef Vec16us uint16_vt;
  typedef Vec8i   int32_vt;
  typedef Vec8ui  uint32_vt;
  typedef Vec4q   int64_vt;
  typedef Vec4uq  uint64_vt;
  typedef Vec8f   float_vt;
  typedef Vec4d   double_vt;

  typedef Vec32cb int8_bvt;
  typedef Vec32cb uint8_bvt;
  typedef Vec16sb int16_bvt;
  typedef Vec16sb uint16_bvt;
  typedef Vec8ib  int32_bvt;
  typedef Vec8ib  uint32_bvt;
  typedef Vec4qb  int64_bvt;
  typedef Vec4qb  uint64_bvt;
  typedef Vec8fb  float_bvt;
  typedef Vec4db  double_bvt;

  typedef int32_t  int32_at[num_int32] __attribute((aligned(32)));
  typedef uint32_t uint32_at[num_int32] __attribute((aligned(32)));
  typedef int64_t  int64_at[num_int64] __attribute((aligned(32)));
  typedef uint64_t uint64_at[num_int64] __attribute((aligned(32)));
  typedef float    float_at[num_float] __attribute((aligned(32)));
  typedef double   double_at[num_double] __attribute((aligned(32)));

  typedef Eigen::Vector3_8f Vector3f_vt;
  typedef Eigen::Matrix3_8f Matrix3f_vt;

  typedef Eigen::Vector3_4d  Vector3d_vt;
  typedef Eigen::Matrix3_4d  Matrix3d_vt;

  typedef Eigen::Vector4_8f Vector4f_vt;
  typedef Eigen::Matrix4_8f Matrix4f_vt;

  typedef Eigen::Vector4_4d  Vector4d_vt;
  typedef Eigen::Matrix4_4d  Matrix4d_vt;

  typedef VCLFloatReal<VCL256Architecture> float_real;
  typedef VCLDoubleReal<VCL256Architecture> double_real;

  static inline float_vt float_iota() {
    return float_vt(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f); }
  static inline double_vt double_iota() {
    return double_vt(0.0, 1.0, 2.0, 3.0); }
  static inline int32_vt int32_iota() {
    return int32_vt(0,1,2,3,4,5,6,7); }
  static inline int64_vt int64_iota() {
    return int64_vt(0,1,2,3); }

  static inline void* aligned_malloc(size_t nbytes) {
    void* p = nullptr;
    if(::posix_memalign(&p, vec_bytes, nbytes)==0) {
      return p;
    }
    throw std::bad_alloc();
  };
  static inline void aligned_free(void* p) {
    ::free(p);
  }
};

struct VCL512Architecture
{
  constexpr static char architecture_name[] = "VCL512Architecture";

  constexpr static unsigned vec_bits   = 512;
  constexpr static unsigned vec_bytes  = vec_bits/8;
  constexpr static unsigned num_int8   = vec_bytes/sizeof(int8_t);
  constexpr static unsigned num_uint8  = vec_bytes/sizeof(uint8_t);
  constexpr static unsigned num_int16  = vec_bytes/sizeof(int16_t);
  constexpr static unsigned num_uint16 = vec_bytes/sizeof(uint16_t);
  constexpr static unsigned num_int32  = vec_bytes/sizeof(int32_t);
  constexpr static unsigned num_uint32 = vec_bytes/sizeof(uint32_t);
  constexpr static unsigned num_int64  = vec_bytes/sizeof(int64_t);
  constexpr static unsigned num_uint64 = vec_bytes/sizeof(uint64_t);
  constexpr static unsigned num_float  = vec_bytes/sizeof(float);
  constexpr static unsigned num_double = vec_bytes/sizeof(double);

  typedef Vec64c  int8_vt;
  typedef Vec64uc uint8_vt;
  typedef Vec32s  int16_vt;
  typedef Vec32us uint16_vt;
  typedef Vec16i  int32_vt;
  typedef Vec16ui uint32_vt;
  typedef Vec8q   int64_vt;
  typedef Vec8uq  uint64_vt;
  typedef Vec16f  float_vt;
  typedef Vec8d   double_vt;

  typedef Vec64cb int8_bvt;
  typedef Vec64cb uint8_bvt;
  typedef Vec32sb int16_bvt;
  typedef Vec32sb uint16_bvt;
  typedef Vec16ib int32_bvt;
  typedef Vec16ib uint32_bvt;
  typedef Vec8qb  int64_bvt;
  typedef Vec8qb  uint64_bvt;
  typedef Vec16fb float_bvt;
  typedef Vec8db  double_bvt;

  typedef int32_t  int32_at[num_int32] __attribute((aligned(64)));
  typedef uint32_t uint32_at[num_int32] __attribute((aligned(64)));
  typedef int64_t  int64_at[num_int64] __attribute((aligned(64)));
  typedef uint64_t uint64_at[num_int64] __attribute((aligned(64)));
  typedef float    float_at[num_float] __attribute((aligned(64)));
  typedef double   double_at[num_double] __attribute((aligned(64)));

  typedef Eigen::Vector3_16f Vector3f_vt;
  typedef Eigen::Matrix3_16f Matrix3f_vt;

  typedef Eigen::Vector3_8d  Vector3d_vt;
  typedef Eigen::Matrix3_8d  Matrix3d_vt;

  typedef Eigen::Vector4_16f Vector4f_vt;
  typedef Eigen::Matrix4_16f Matrix4f_vt;

  typedef Eigen::Vector4_8d  Vector4d_vt;
  typedef Eigen::Matrix4_8d  Matrix4d_vt;

  typedef VCLFloatReal<VCL512Architecture> float_real;
  typedef VCLDoubleReal<VCL512Architecture> double_real;

  static inline float_vt float_iota() {
    return float_vt(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f); }
  static inline double_vt double_iota() {
    return double_vt(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0); }
  static inline int32_vt int32_iota() {
    return int32_vt(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15); }
  static inline int64_vt int64_iota() {
    return int64_vt(0,1,2,3,4,5,6,7); }

  static inline void* aligned_malloc(size_t nbytes) {
    void* p = nullptr;
    if(::posix_memalign(&p, vec_bytes, nbytes)==0) {
      return p;
    }
    throw std::bad_alloc();
  };
  static inline void aligned_free(void* p) {
    ::free(p);
  }
};

typedef VCLFloatReal<VCL128Architecture> VCL128FloatReal;
typedef VCLFloatReal<VCL256Architecture> VCL256FloatReal;
typedef VCLFloatReal<VCL512Architecture> VCL512FloatReal;

typedef VCLDoubleReal<VCL128Architecture> VCL128DoubleReal;
typedef VCLDoubleReal<VCL256Architecture> VCL256DoubleReal;
typedef VCLDoubleReal<VCL512Architecture> VCL512DoubleReal;

#if INSTRSET >= 9
typedef VCL512Architecture VCLMaxArchitecture;
#elif INSTRSET >= 7
typedef VCL256Architecture VCLMaxArchitecture;
#else
typedef VCL128Architecture VCLMaxArchitecture;
#endif
typedef VCLFloatReal<VCLMaxArchitecture> VCLMaxFloatReal;
typedef VCLDoubleReal<VCLMaxArchitecture> VCLMaxDoubleReal;

template<typename VCLArchitecture> std::string templated_class_name(
  const std::string& class_name)
{
  return class_name + "<" + VCLArchitecture::architecture_name + ">";
}

template<typename T> struct vcl_type
{
  typedef void scalar_type;
  typedef void vcl_architecture;
};

#define DEFINE_VCL_TYPE(Vec,Scalar,Arch) \
  template<> struct vcl_type<Vec> \
  { \
    typedef Scalar scalar_type; \
    typedef Arch vcl_architecture; \
  }

DEFINE_VCL_TYPE(Vec16c,  int8_t,   VCL128Architecture);
DEFINE_VCL_TYPE(Vec16uc, uint8_t,  VCL128Architecture);
DEFINE_VCL_TYPE(Vec8s,   int16_t,  VCL128Architecture);
DEFINE_VCL_TYPE(Vec8us,  uint16_t, VCL128Architecture);
DEFINE_VCL_TYPE(Vec4i,   int32_t,  VCL128Architecture);
DEFINE_VCL_TYPE(Vec4ui,  uint32_t, VCL128Architecture);
DEFINE_VCL_TYPE(Vec2q,   int64_t,  VCL128Architecture);
DEFINE_VCL_TYPE(Vec2uq,  uint64_t, VCL128Architecture);
DEFINE_VCL_TYPE(Vec4f,   float,    VCL128Architecture);
DEFINE_VCL_TYPE(Vec2d,   double,   VCL128Architecture);

DEFINE_VCL_TYPE(Vec32c,  int8_t,   VCL256Architecture);
DEFINE_VCL_TYPE(Vec32uc, uint8_t,  VCL256Architecture);
DEFINE_VCL_TYPE(Vec16s,  int16_t,  VCL256Architecture);
DEFINE_VCL_TYPE(Vec16us, uint16_t, VCL256Architecture);
DEFINE_VCL_TYPE(Vec8i,   int32_t,  VCL256Architecture);
DEFINE_VCL_TYPE(Vec8ui,  uint32_t, VCL256Architecture);
DEFINE_VCL_TYPE(Vec4q,   int64_t,  VCL256Architecture);
DEFINE_VCL_TYPE(Vec4uq,  uint64_t, VCL256Architecture);
DEFINE_VCL_TYPE(Vec8f,   float,    VCL256Architecture);
DEFINE_VCL_TYPE(Vec4d,   double,   VCL256Architecture);

DEFINE_VCL_TYPE(Vec64c,  int8_t,   VCL512Architecture);
DEFINE_VCL_TYPE(Vec64uc, uint8_t,  VCL512Architecture);
DEFINE_VCL_TYPE(Vec32s,  int16_t,  VCL512Architecture);
DEFINE_VCL_TYPE(Vec32us, uint16_t, VCL512Architecture);
DEFINE_VCL_TYPE(Vec16i,  int32_t,  VCL512Architecture);
DEFINE_VCL_TYPE(Vec16ui, uint32_t, VCL512Architecture);
DEFINE_VCL_TYPE(Vec8q,   int64_t,  VCL512Architecture);
DEFINE_VCL_TYPE(Vec8uq,  uint64_t, VCL512Architecture);
DEFINE_VCL_TYPE(Vec16f,  float,    VCL512Architecture);
DEFINE_VCL_TYPE(Vec8d,   double,   VCL512Architecture);

#undef DEFINE_VCL_TYPE

template<typename Vec> void print_vec(std::ostream& s, const Vec& v)
{
  s << v[0];
  for(int i=1;i<v.size();i++)
    s << '_' << v[i];
}

template<typename VCLReal> inline void
insert_into_vec3_with_mask(typename VCLReal::vec3_vt& vv,
    const typename VCLReal::vec3_t& vs, const typename VCLReal::bool_vt& mask) {
  vv.x() = vcl::select(mask, vs.x(), vv.x());
  vv.y() = vcl::select(mask, vs.y(), vv.y());
  vv.z() = vcl::select(mask, vs.z(), vv.z());
}

template<typename VCLReal> inline void
insert_into_with_mask(typename VCLReal::real_vt& v,
    const typename VCLReal::real_t& s, const typename VCLReal::bool_vt& mask) {
  v = vcl::select(mask, s, v);
}

typedef Eigen::Matrix< ::vcl::Vec4f , 3 , 1> Vector3_4f;
typedef Eigen::Matrix< ::vcl::Vec4f , 3 , 3> Matrix3_4f;
typedef Eigen::Matrix< ::vcl::Vec8f , 3 , 1> Vector3_8f;
typedef Eigen::Matrix< ::vcl::Vec8f , 3 , 3> Matrix3_8f;
typedef Eigen::Matrix< ::vcl::Vec16f , 3 , 1> Vector3_16f;
typedef Eigen::Matrix< ::vcl::Vec16f , 3 , 3> Matrix3_16f;

typedef Eigen::Matrix< ::vcl::Vec2d , 3 , 1> Vector3_2d;
typedef Eigen::Matrix< ::vcl::Vec2d , 3 , 3> Matrix3_2d;
typedef Eigen::Matrix< ::vcl::Vec4d , 3 , 1> Vector3_4d;
typedef Eigen::Matrix< ::vcl::Vec4d , 3 , 3> Matrix3_4d;
typedef Eigen::Matrix< ::vcl::Vec8d , 3 , 1> Vector3_8d;
typedef Eigen::Matrix< ::vcl::Vec8d , 3 , 3> Matrix3_8d;


inline Vec2uq mul_low32_packed64(const Vec2uq& a, const Vec2uq& b) {
  return _mm_mul_epu32(a, b);
}

#if (defined (__AVX512DQ__) && defined (__AVX512VL__)) || INSTRSET < 5
  inline Vec2uq mul_64(const Vec2uq& a, const Vec2uq& b) {
    return a*b;
  }
#else
  inline Vec2uq mul_64(const Vec2uq& a, const Vec2uq& b) {
    __m128i prod = _mm_mul_epu32(_mm_shuffle_epi32(a, 0xB1), b);
    __m128i tmp = _mm_mul_epu32(a, _mm_shuffle_epi32(b, 0xB1));
    prod = _mm_add_epi64(prod, tmp);
    prod = _mm_slli_epi64(prod, 32);
    tmp = _mm_mul_epu32(a, b);
    prod = _mm_add_epi64(prod, tmp);
    return prod;
  }
#endif

#if INSTRSET >= 5   // SSE4.1
inline Vec4ui extend_16_to_32_low(const Vec8us x) {
  return _mm_cvtepu16_epi32(x);
}
inline Vec4ui extend_16_to_32_high(const Vec8us x) {
  return _mm_cvtepu16_epi32(_mm_srli_si128(x,8));
}
inline Vec4i extend_16_to_32_low(const Vec8s x) {
  return _mm_cvtepi16_epi32(x);
}
inline Vec4i extend_16_to_32_high(const Vec8s x) {
  return _mm_cvtepi16_epi32(_mm_srli_si128(x,8));
}
#else // INSTRSET < 5 no SSE4.1
inline Vec4ui extend_16_to_32_low(const Vec8us x) {
  return vcl::extend_low(x);
}
inline Vec4ui extend_16_to_32_high(const Vec8us x) {
  return vcl::extend_high(x);
}
inline Vec4i extend_16_to_32_low(const Vec8s x) {
  return vcl::extend_low(x);
}
inline Vec4i extend_16_to_32_high(const Vec8s x) {
  return vcl::extend_high(x);
}
#endif // INSTRSET < 5 no SSE4.1

#if MAX_VECTOR_SIZE >= 256
#if INSTRSET >= 8
  inline Vec4uq mul_low32_packed64(const Vec4uq& a, const Vec4uq& b) {
    return _mm256_mul_epu32(a, b);
  }

  inline Vec8ui extend_16_to_32_low(const Vec16us x) {
    return _mm256_cvtepu16_epi32(x.get_low());
  }
  inline Vec8ui extend_16_to_32_high(const Vec16us x) {
    return _mm256_cvtepu16_epi32(x.get_high());
  }
  inline Vec8i extend_16_to_32_low(const Vec16s x) {
    return _mm256_cvtepi16_epi32(x.get_low());
  }
  inline Vec8i extend_16_to_32_high(const Vec16s x) {
    return _mm256_cvtepi16_epi32(x.get_high());
  }

#if defined (__AVX512DQ__) && defined (__AVX512VL__)
  inline Vec4uq mul_64(const Vec4uq& a, const Vec4uq& b) {
    return a*b;
  }
#else
  inline Vec4uq mul_64(const Vec4uq& a, const Vec4uq& b) {
    __m256i prod = _mm256_mul_epu32(_mm256_shuffle_epi32(a, 0xB1), b);
    __m256i tmp = _mm256_mul_epu32(a, _mm256_shuffle_epi32(b, 0xB1));
    prod = _mm256_add_epi64(prod, tmp);
    prod = _mm256_slli_epi64(prod, 32);
    tmp = _mm256_mul_epu32(a, b);
    prod = _mm256_add_epi64(prod, tmp);
    return prod;
  }
#endif // defined (__AVX512DQ__) && defined (__AVX512VL__)
#else // INSTRSET < 8
  inline Vec4uq mul_low32_packed64(const Vec4uq& a, const Vec4uq& b) {
    return Vec4uq(mul_low32_packed64(a.get_low(), b.get_low()),
                  mul_low32_packed64(a.get_high(), b.get_high()));
  }

  inline Vec8ui extend_16_to_32_low(const Vec16us x) {
    return Vec8ui(extend_16_to_32_low(x.get_low()),
                  extend_16_to_32_high(x.get_low()));
  }
  inline Vec8ui extend_16_to_32_high(const Vec16us x) {
    return Vec8ui(extend_16_to_32_low(x.get_high()),
                  extend_16_to_32_high(x.get_high()));
  }
  inline Vec8i extend_16_to_32_low(const Vec16s x) {
    return Vec8i(extend_16_to_32_low(x.get_low()),
                 extend_16_to_32_high(x.get_low()));
  }
  inline Vec8i extend_16_to_32_high(const Vec16s x) {
    return Vec8i(extend_16_to_32_low(x.get_high()),
                 extend_16_to_32_high(x.get_high()));
  }

  inline Vec4uq mul_64(const Vec4uq& a, const Vec4uq& b) {
    return a*b;
  }
#endif // INSTRSET < 8
#endif // MAX_VECTOR_SIZE >= 256

#if MAX_VECTOR_SIZE >= 512
#if INSTRSET >= 9
  inline Vec8uq mul_low32_packed64(const Vec8uq& a, const Vec8uq& b) {
    return _mm512_mul_epu32(a, b);
  }

  inline Vec16ui extend_16_to_32_low(const Vec32us x) {
    return _mm512_cvtepu16_epi32(x.get_low());
  }
  inline Vec16ui extend_16_to_32_high(const Vec32us x) {
    return _mm512_cvtepu16_epi32(x.get_high());
  }
  inline Vec16i extend_16_to_32_low(const Vec32s x) {
    return _mm512_cvtepi16_epi32(x.get_low());
  }
  inline Vec16i extend_16_to_32_high(const Vec32s x) {
    return _mm512_cvtepi16_epi32(x.get_high());
  }
#else // INSTRSET < 9
  inline Vec8uq mul_low32_packed64(const Vec8uq& a, const Vec8uq& b) {
    return Vec8uq(mul_low32_packed64(a.get_low(), b.get_low()),
                  mul_low32_packed64(a.get_high(), b.get_high()));
  }

  inline Vec16ui extend_16_to_32_low(const Vec32us x) {
    return Vec16ui(extend_16_to_32_low(x.get_low()),
                  extend_16_to_32_high(x.get_low()));
  }
  inline Vec16ui extend_16_to_32_high(const Vec32us x) {
    return Vec16ui(extend_16_to_32_low(x.get_high()),
                  extend_16_to_32_high(x.get_high()));
  }
  inline Vec16i extend_16_to_32_low(const Vec32s x) {
    return Vec16i(extend_16_to_32_low(x.get_low()),
                 extend_16_to_32_high(x.get_low()));
  }
  inline Vec16i extend_16_to_32_high(const Vec32s x) {
    return Vec16i(extend_16_to_32_low(x.get_high()),
                 extend_16_to_32_high(x.get_high()));
  }
#endif // INSTRSET < 9

  inline Vec8uq mul_64(const Vec8uq& a, const Vec8uq& b) {
    return a*b;
  }
#endif // MAX_VECTOR_SIZE >= 512

void transpose(Vec8s* x);
void transpose(Vec8us* x);
void transpose(Vec4i* x);
void transpose(Vec4ui* x);
void transpose(Vec2q* x);
void transpose(Vec2uq* x);
void transpose(Vec4f* x);
void transpose(Vec2d* x);
#if MAX_VECTOR_SIZE >= 256
void transpose(Vec16s* x);
void transpose(Vec16us* x);
void transpose(Vec8i* x);
void transpose(Vec8ui* x);
void transpose(Vec4q* x);
void transpose(Vec4uq* x);
void transpose(Vec8f* x);
void transpose(Vec4d* x);
#endif
#if MAX_VECTOR_SIZE >= 512
void transpose(Vec32s* x);
void transpose(Vec32us* x);
void transpose(Vec16i* x);
void transpose(Vec16ui* x);
void transpose(Vec8q* x);
void transpose(Vec8uq* x);
void transpose(Vec16f* x);
void transpose(Vec8d* x);
#endif


inline Vec8s reverse(Vec8s x) { return ::vcl::permute8s<7,6,5,4,3,2,1,0>(x); }
inline Vec8us reverse(Vec8us x) { return ::vcl::permute8us<7,6,5,4,3,2,1,0>(x); }
inline Vec4i reverse(Vec4i x) { return ::vcl::permute4i<3,2,1,0>(x); }
inline Vec4ui reverse(Vec4ui x) { return ::vcl::permute4ui<3,2,1,0>(x); }
inline Vec2q reverse(Vec2q x) { return ::vcl::permute2q<1,0>(x); }
inline Vec2uq reverse(Vec2uq x) { return ::vcl::permute2uq<1,0>(x); }
inline Vec4f reverse(Vec4f x) { return ::vcl::permute4f<3,2,1,0>(x); }
inline Vec2d reverse(Vec2d x) { return ::vcl::permute2d<1,0>(x); }
#if MAX_VECTOR_SIZE >= 256
inline Vec16s reverse(Vec16s x) { return ::vcl::permute16s<15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0>(x); }
inline Vec16us reverse(Vec16us x) { return ::vcl::permute16us<15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0>(x); }
inline Vec8i reverse(Vec8i x) { return ::vcl::permute8i<7,6,5,4,3,2,1,0>(x); }
inline Vec8ui reverse(Vec8ui x) { return ::vcl::permute8ui<7,6,5,4,3,2,1,0>(x); }
inline Vec4q reverse(Vec4q x) { return ::vcl::permute4q<3,2,1,0>(x); }
inline Vec4uq reverse(Vec4uq x) { return ::vcl::permute4uq<3,2,1,0>(x); }
inline Vec8f reverse(Vec8f x) { return ::vcl::permute8f<7,6,5,4,3,2,1,0>(x); }
inline Vec4d reverse(Vec4d x) { return ::vcl::permute4d<3,2,1,0>(x); }
#endif
#if MAX_VECTOR_SIZE >= 512
inline Vec16i reverse(Vec16i x) { return ::vcl::permute16i<15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0>(x); }
inline Vec16ui reverse(Vec16ui x) { return Vec16ui(::vcl::permute16ui<15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0>(Vec16i(x))); }
inline Vec8q reverse(Vec8q x) { return ::vcl::permute8q<7,6,5,4,3,2,1,0>(x); }
inline Vec8uq reverse(Vec8uq x) { return ::vcl::permute8uq<7,6,5,4,3,2,1,0>(x); }
inline Vec16f reverse(Vec16f x) { return ::vcl::permute16f<15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0>(x); }
inline Vec8d reverse(Vec8d x) { return ::vcl::permute8d<7,6,5,4,3,2,1,0>(x); }
#endif


} } } // namespace calin::util::vcl

#define ADD_OSTREAM_OPERATOR(Vec) \
  inline std::ostream& operator<<(std::ostream& s, const Vec& v) { \
    calin::util::vcl::print_vec(s, v); \
    return s; \
  }

namespace vcl {

ADD_OSTREAM_OPERATOR(Vec16c);
ADD_OSTREAM_OPERATOR(Vec16uc);
ADD_OSTREAM_OPERATOR(Vec8s);
ADD_OSTREAM_OPERATOR(Vec8us);
ADD_OSTREAM_OPERATOR(Vec4i);
ADD_OSTREAM_OPERATOR(Vec4ui);
ADD_OSTREAM_OPERATOR(Vec2q);
ADD_OSTREAM_OPERATOR(Vec2uq);
ADD_OSTREAM_OPERATOR(Vec4f);
ADD_OSTREAM_OPERATOR(Vec2d);

ADD_OSTREAM_OPERATOR(Vec16cb);
ADD_OSTREAM_OPERATOR(Vec8sb);
ADD_OSTREAM_OPERATOR(Vec4ib);
ADD_OSTREAM_OPERATOR(Vec2qb);
ADD_OSTREAM_OPERATOR(Vec4fb);
ADD_OSTREAM_OPERATOR(Vec2db);

ADD_OSTREAM_OPERATOR(Vec32c);
ADD_OSTREAM_OPERATOR(Vec32uc);
ADD_OSTREAM_OPERATOR(Vec16s);
ADD_OSTREAM_OPERATOR(Vec16us);
ADD_OSTREAM_OPERATOR(Vec8i);
ADD_OSTREAM_OPERATOR(Vec8ui);
ADD_OSTREAM_OPERATOR(Vec4q);
ADD_OSTREAM_OPERATOR(Vec4uq);
ADD_OSTREAM_OPERATOR(Vec8f);
ADD_OSTREAM_OPERATOR(Vec4d);

ADD_OSTREAM_OPERATOR(Vec32cb);
ADD_OSTREAM_OPERATOR(Vec16sb);
ADD_OSTREAM_OPERATOR(Vec8ib);
ADD_OSTREAM_OPERATOR(Vec4qb);
ADD_OSTREAM_OPERATOR(Vec8fb);
ADD_OSTREAM_OPERATOR(Vec4db);

// ADD_OSTREAM_OPERATOR(Vec64c);
// ADD_OSTREAM_OPERATOR(Vec64uc);
ADD_OSTREAM_OPERATOR(Vec32s);
ADD_OSTREAM_OPERATOR(Vec32us);
ADD_OSTREAM_OPERATOR(Vec16i);
ADD_OSTREAM_OPERATOR(Vec16ui);
ADD_OSTREAM_OPERATOR(Vec8q);
ADD_OSTREAM_OPERATOR(Vec8uq);
ADD_OSTREAM_OPERATOR(Vec16f);
ADD_OSTREAM_OPERATOR(Vec8d);

ADD_OSTREAM_OPERATOR(Vec16ib);
ADD_OSTREAM_OPERATOR(Vec8qb);
ADD_OSTREAM_OPERATOR(Vec16fb);
ADD_OSTREAM_OPERATOR(Vec8db);
}

#undef ADD_OSTREAM_OPERATOR

namespace calin { namespace util { namespace vcl {

//template

} } } // namespace calin::util::vcl
