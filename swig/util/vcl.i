/*

   calin/util/utm.i -- Stephen Fegan -- 2022-07-31

   SWIG interface file for calin.util.utm

   Copyright 2022, Stephen Fegan <sfegan@llr.in2p3.fr>
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

%module (package="calin.util") vcl
%feature(autodoc,2);

%{
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>

#include <util/vcl.hpp>
#define SWIG_FILE_WITH_INIT
  %}

%init %{
  import_array();
%}

%include "calin_typemaps.i"
%import "calin_global_definitions.i"

namespace calin { namespace util { namespace vcl {

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
};

typedef VCLFloatReal<VCL128Architecture> VCL128FloatReal;
typedef VCLFloatReal<VCL256Architecture> VCL256FloatReal;
typedef VCLFloatReal<VCL512Architecture> VCL512FloatReal;

typedef VCLDoubleReal<VCL128Architecture> VCL128DoubleReal;
typedef VCLDoubleReal<VCL256Architecture> VCL256DoubleReal;
typedef VCLDoubleReal<VCL512Architecture> VCL512DoubleReal;

} } } // namespace calin::util::vcl

%template (VCLFloatRealVCL128Architecture) calin::util::vcl::VCLFloatReal<calin::util::vcl::VCL128Architecture>;
%template (VCLFloatRealVCL256Architecture) calin::util::vcl::VCLFloatReal<calin::util::vcl::VCL256Architecture>;
%template (VCLFloatRealVCL512Architecture) calin::util::vcl::VCLFloatReal<calin::util::vcl::VCL512Architecture>;

%template (VCLDoubleRealVCL128Architecture) calin::util::vcl::VCLDoubleReal<calin::util::vcl::VCL128Architecture>;
%template (VCLDoubleRealVCL256Architecture) calin::util::vcl::VCLDoubleReal<calin::util::vcl::VCL256Architecture>;
%template (VCLDoubleRealVCL512Architecture) calin::util::vcl::VCLDoubleReal<calin::util::vcl::VCL512Architecture>;
