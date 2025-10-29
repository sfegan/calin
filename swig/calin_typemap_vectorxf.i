/*

   calin/calin_typemaps_VectorXf.i -- Stephen Fegan -- 2017-01-19

   SWIG interface file for common calin typemaps : Eigen::VectorXf

   Copyright 2017, Stephen Fegan <sfegan@llr.in2p3.fr>
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

// =============================================================================
//
// Typemaps for using Eigen::VectorXf - these require data be copied once on
// input and again on output (for non-const references)
//
// =============================================================================

%fragment("Calin_Python_to_EigenFltVec",
          "header",
          fragment="NumPy_Array_Requirements",
          fragment="NumPy_Backward_Compatibility",
          fragment="NumPy_Macros",
          fragment="NumPy_Utilities")
{
  static bool calin_python_to_eigen_flt_vec(PyObject* input, Eigen::VectorXf* vec)
  {
    const int typecode = NPT_FLOAT;

    if(!_swig_numpy_is_array(input))
      {
        const char* desired_type = typecode_string(typecode);
        const char* actual_type  = pytype_string(input);
        PyErr_Format(PyExc_TypeError,
                     "Array of type '%s' required.  A '%s' was given",
                     desired_type,
                     actual_type);
        return false;
      }

    PyArrayObject* in_array = (PyArrayObject*) input;

    if(_swig_numpy_array_numdims(in_array) > 1)
    {
      PyErr_Format(PyExc_TypeError,
                   "Array must have 1 dimension. "
                   "Given array has %d dimensions",
                   _swig_numpy_array_numdims(in_array));
      return false;
    }

    if(_swig_numpy_array_numdims(in_array)==0 or _swig_numpy_array_size(in_array, 0)==0)
    {
      *vec = Eigen::VectorXf();
      return true;
    }

    npy_intp size[1] = { _swig_numpy_array_size(in_array, 0) };
    vec->resize(size[0]);

    PyArrayObject* out_array = (PyArrayObject*)
        PyArray_SimpleNewFromData(1, size, typecode, vec->data());
    if(out_array == nullptr)return false;

    if(PyArray_CopyInto(out_array, in_array) != 0)
      {
        Py_DECREF(out_array);
        return false;
      }

    Py_DECREF(out_array);
    return true;
  }

} // fragment("Calin_Python_to_EigenFltVec"

%fragment("Calin_EigenFltVec_to_Python",
          "header",
          fragment="NumPy_Array_Requirements",
          fragment="NumPy_Backward_Compatibility",
          fragment="NumPy_Macros",
          fragment="NumPy_Utilities")
{
  static bool calin_eigen_flt_vec_to_python(Eigen::VectorXf* vec,
                                        PyObject* output)
  {
    const int typecode = NPT_FLOAT;

    if(!_swig_numpy_is_array(output))
      {
        const char* desired_type = typecode_string(typecode);
        const char* actual_type  = pytype_string(output);
        PyErr_Format(PyExc_TypeError,
                     "Array of type '%s' required.  A '%s' was given",
                     desired_type,
                     actual_type);
        return false;
      }

    npy_intp size[1] = { vec->size() };
    PyArrayObject* in_array = (PyArrayObject*)
        PyArray_SimpleNewFromData(1, size, typecode, vec->data());
    if(in_array == nullptr)
      {
        return false;
      }

    PyArrayObject* out_array = (PyArrayObject*) output;

    PyArray_Dims dims = { size, 1 };
    if(PyArray_Resize(out_array, &dims, 0, NPY_ANYORDER) == nullptr)
      {
        // Do we need to call Py_DECREF on returned val??
        Py_DECREF(in_array);
        return false;
      }

    if(PyArray_CopyInto(out_array, in_array) != 0)
      {
        Py_DECREF(in_array);
        return false;
      }

    Py_DECREF(in_array);
    return true;
  }

} // fragment("Calin_EigenFltVec_to_Python"

// *************************** const Eigen::VectorXf& **************************

%typemap(in, fragment="Calin_Python_to_EigenFltVec")
     const Eigen::VectorXf& (Eigen::VectorXf temp)
{
  // typemap(in) const Eigen::VectorXf& -- calin_typemap_VectorXf.i
  $1 = &temp;
  if(!calin_python_to_eigen_flt_vec($input, $1))SWIG_fail;
}

%typemap(out, fragment="Calin_EigenFltVec_to_Python") const Eigen::VectorXf&
{
  // typemap(out) const Eigen::VectorXf& -- calin_typemap_VectorXf.i
  npy_intp size[1] { $1->size() };
  $result = PyArray_EMPTY(1, size, NPT_FLOAT, 0);
  if(!$result)SWIG_fail;
  if(!calin_eigen_flt_vec_to_python($1, $result))SWIG_fail;
}

%typemap(argout) const Eigen::VectorXf&
{
  // typemap(argout) const Eigen::VectorXf& -- calin_typemap_VectorXf.i
  // nothing to see here
}

%typemap(typecheck, precedence=5000) const Eigen::VectorXf&
{
  // typemap(typecheck) const Eigen::VectorXf& -- calin_typemap_VectorXf.i
  $1 = _swig_numpy_is_array($input) ? 1 : 0;
}

// ****************************** Eigen::VectorXf& *****************************

%typemap(in, fragment="Calin_Python_to_EigenFltVec")
     Eigen::VectorXf& (Eigen::VectorXf temp)
{
  // typemap(in) Eigen::VectorXf& -- calin_typemap_VectorXf.i
  $1 = &temp;
  if(!calin_python_to_eigen_flt_vec($input, $1))SWIG_fail;
}

%typemap(argout, fragment="Calin_EigenFltVec_to_Python") Eigen::VectorXf&
{
  // typemap(argout) Eigen::VectorXf& -- calin_typemap_VectorXf.i
  if(!calin_eigen_flt_vec_to_python($1, $input))SWIG_fail;
}

%typemap(typecheck, precedence=5000) Eigen::VectorXf&
{
  // typemap(typecheck) Eigen::VectorXf& -- calin_typemap_VectorXf.i
  $1 = _swig_numpy_is_array($input) ? 1 : 0;
}

// ************************** Eigen::VectorXf &OUTPUT **************************

%typemap(in, numinputs=0) Eigen::VectorXf &OUTPUT (Eigen::VectorXf temp)
{
  // typemap(in) Eigen::VectorXf &OUTPUT -- calin_typemap_VectorXf.i
  $1 = &temp;
}

%typemap(argout, fragment="Calin_EigenFltVec_to_Python") Eigen::VectorXf &OUTPUT
{
  // typemap(argout) Eigen::VectorXf &OUTPUT -- calin_typemap_VectorXf.i
  npy_intp size[1] { $1->size() };
  PyObject* temp_array = PyArray_EMPTY(1, size, NPT_FLOAT, 0);
  if(!temp_array)SWIG_fail;
  if(!calin_eigen_flt_vec_to_python($1, temp_array))
  {
    Py_DECREF(temp_array);
    SWIG_fail;
  }
  $result = SWIG_Python_AppendOutput($result, temp_array);
}

// ************************** Eigen::VectorXf &INOUT ***************************

%typemap(in, fragment="Calin_Python_to_EigenFltVec")
  Eigen::VectorXf &INOUT (Eigen::VectorXf temp)
{
  // typemap(in) const Eigen::VectorXf &INOUT -- calin_typemap_VectorXf.i
  $1 = &temp;
  if(!calin_python_to_eigen_flt_vec($input, $1))SWIG_fail;
}

%typemap(argout, fragment="Calin_EigenFltVec_to_Python") Eigen::VectorXf &INOUT
{
  // typemap(argout) Eigen::VectorXf &INOUT -- calin_typemap_VectorXf.i
  npy_intp size[1] { $1->size() };
  PyObject* temp_array = PyArray_EMPTY(1, size, NPT_FLOAT, 0);
  if(!temp_array)SWIG_fail;
  if(!calin_eigen_flt_vec_to_python($1, temp_array))
  {
    Py_DECREF(temp_array);
    SWIG_fail;
  }
  $result = SWIG_Python_AppendOutput($result, temp_array);
}

// ****************************** Eigen::VectorXf ******************************

%typemap(out, fragment="Calin_EigenFltVec_to_Python") Eigen::VectorXf
{
  // typemap(out) Eigen::VectorXf -- calin_typemap_VectorXf.i
  npy_intp size[1] { $1.size() };
  $result = PyArray_EMPTY(1, size, NPT_FLOAT, 0);
  if(!$result)SWIG_fail;
  if(!calin_eigen_flt_vec_to_python(&$1, $result))SWIG_fail;
}

%typemap(typecheck, precedence=5000) Eigen::VectorXf
{
  // typemap(typecheck) Eigen::VectorXf -- calin_typemap_VectorXf.i
  $1 = _swig_numpy_is_array($input) ? 1 : 0;
}
