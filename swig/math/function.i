//-*-mode:swig;-*-

/* 

   calin/math/function.i -- Stephen Fegan -- 2015-04-15

*/

%module (package="calin.math") function

%{
#include "math/function.hpp"
#define SWIG_FILE_WITH_INIT
  %}

%init %{
  import_array();
%}

%include "package_wide_definitions.i"

%include "math/function.hpp"

%template (VectorParameterAxis) std::vector<calin::math::function::ParameterAxis>;