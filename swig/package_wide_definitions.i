//-*-mode:swig;-*-

/* 

   calin/package_wide_definitions.i -- Stephen Fegan -- 2015-04-21

*/

%module (package="calin") package_wide_definitions

%{
#include <iostream>
#include "package_wide_definitions.hpp"
#define SWIG_FILE_WITH_INIT
  %}

%include "numpy.i"
%include "std_string.i"
%include "std_vector.i"

%init %{
  import_array();
%}

%template (VectorDouble) std::vector<double>;

%import "package_wide_definitions.hpp"

// =============================================================================
//
// Typemaps for using Eigen::VectorXd - these require data be copied once on
// input and again on output (for non-const references)
//
// =============================================================================

%fragment("Calin_Python_to_EigenVec",
          "header",
          fragment="NumPy_Array_Requirements",
          fragment="NumPy_Backward_Compatibility",
          fragment="NumPy_Macros",
          fragment="NumPy_Utilities")
{
  static bool calin_python_to_eigen_vec(PyObject* input, Eigen::VectorXd& vec)
  {
    const int typecode = NPY_DOUBLE;

    if(!is_array(input))
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

    if(PyArray_NDIM(in_array) > 1)
    {
      PyErr_Format(PyExc_TypeError,
                   "Array must have 1 dimension. "
                   "Given array has %d dimensions",
                   PyArray_NDIM(in_array));
      return false;
    }
    
    if(PyArray_NDIM(in_array)==0 or PyArray_DIM(in_array, 0)==0)
    {
      vec = Eigen::VectorXd();
      return true;
    }

    npy_intp size[1] = { PyArray_DIM(in_array, 0) };
    vec.resize(size[0]);

    PyArrayObject* out_array = (PyArrayObject*)
        PyArray_SimpleNewFromData(1, size, typecode, vec.data());
    if(out_array == nullptr)return false;
        
    if(PyArray_CopyInto(out_array, in_array) != 0)
      {
        Py_DECREF(out_array);
        return false;
      }

    Py_DECREF(out_array);
    return true;
  }

  static bool calin_eigen_vec_to_python(Eigen::VectorXd& vec,
                                        PyObject* output)
  {
    const int typecode = NPY_DOUBLE;

    if(!is_array(output))
      {
        const char* desired_type = typecode_string(typecode);
        const char* actual_type  = pytype_string(output);
        PyErr_Format(PyExc_TypeError,
                     "Array of type '%s' required.  A '%s' was given",
                     desired_type,
                     actual_type);
        return false;
      }

    npy_intp size[1] = { vec.size() };
    PyArrayObject* in_array = (PyArrayObject*)
        PyArray_SimpleNewFromData(1, size, typecode, vec.data());
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

}

%typemap(in,
         fragment="Calin_Python_to_EigenVec")
         const Eigen::VectorXd&
{
  $1 = new Eigen::VectorXd();
  if(!calin_python_to_eigen_vec($input, *$1))SWIG_fail;
}

%typemap(freearg) const Eigen::VectorXd&
{
  delete arg$argnum;
}

%typemap(in,
         fragment="Calin_Python_to_EigenVec")
         Eigen::VectorXd&
{
  $1 = new Eigen::VectorXd();
  if(!calin_python_to_eigen_vec($input, *$1))SWIG_fail;
}

%typemap(argout,
         fragment="Calin_Python_to_EigenVec")
         Eigen::VectorXd&
{
  //  if(!calin_eigen_vec_to_python(*$1, $result))SWIG_fail;
  if(!calin_eigen_vec_to_python(*$1, $input))SWIG_fail;
}

%typemap(freearg)
         Eigen::VectorXd&
{
  delete arg$argnum;
}

%typemap(out,
         fragment="Calin_Python_to_EigenVec")
         Eigen::VectorXd
{
  npy_intp size[1] { $1.size() };
  $result = PyArray_EMPTY(1, size, NPY_DOUBLE, 0);
  if(!calin_eigen_vec_to_python($1, $result))SWIG_fail;
}

// =============================================================================
//
// Typemaps for using Eigen::MatrixXd - these require data be copied once on
// input and again on output (for non-const references)
//
// =============================================================================

%fragment("Calin_Python_to_EigenMat",
          "header",
          fragment="NumPy_Array_Requirements",
          fragment="NumPy_Backward_Compatibility",
          fragment="NumPy_Macros",
          fragment="NumPy_Utilities")
{
  static bool calin_python_to_eigen_mat(PyObject* input, Eigen::MatrixXd& mat)
  {
    const int typecode = NPY_DOUBLE;

    if(!is_array(input))
      {
        const char* desired_type = typecode_string(typecode);
        const char* actual_type  = pytype_string(input);
        PyErr_Format(PyExc_TypeError,
                     "Array of type '%s' required. A '%s' was given",
                     desired_type,
                     actual_type);
        return false;
      }
    
    PyArrayObject* in_array = (PyArrayObject*) input;

    if(PyArray_NDIM(in_array) > 2)
    {
      PyErr_Format(PyExc_TypeError,
                   "Array must have at most 2 dimensions. "
                   "Given array has %d dimensions",
                   PyArray_NDIM(in_array));
      return false;
    }
    
    if(PyArray_NDIM(in_array)==0 or PyArray_DIM(in_array, 0)==0 or
       (PyArray_NDIM(in_array)==2 and PyArray_DIM(in_array, 1)==0))
    {
      mat = Eigen::MatrixXd();
      return true;
    }

    npy_intp size[2] = { PyArray_DIM(in_array, 0), 1 };
    if(PyArray_NDIM(in_array)==2)
      size[1] = array_size(in_array,1);

    mat.resize(size[0], size[1]);
    
    PyArrayObject* out_array = (PyArrayObject*)
        PyArray_New(&PyArray_Type, PyArray_NDIM(in_array), size, typecode,
                    NULL, mat.data(), 0, NPY_ARRAY_FARRAY, NULL);
    
    if(out_array == nullptr)return false;
        
    if(PyArray_CopyInto(out_array, in_array) != 0)
      {
        Py_DECREF(out_array);
        return false;
      }

    Py_DECREF(out_array);
    return true;
  }

  static bool calin_eigen_mat_to_python(Eigen::MatrixXd& mat,
                                        PyObject* output)
  {
    const int typecode = NPY_DOUBLE;

    if(!is_array(output))
      {
        const char* desired_type = typecode_string(typecode);
        const char* actual_type  = pytype_string(output);
        PyErr_Format(PyExc_TypeError,
                     "Array of type '%s' required.  A '%s' was given",
                     desired_type,
                     actual_type);
        return false;
      }

    npy_intp size[2] = { mat.rows(), mat.cols() };
    PyArrayObject* in_array = (PyArrayObject*)
        PyArray_New(&PyArray_Type, 2, size, typecode,
                    NULL, mat.data(), 0, NPY_ARRAY_FARRAY, NULL);
    if(in_array == nullptr)
      {
        return false;
      }
    
    PyArrayObject* out_array = (PyArrayObject*) output;

    PyArray_Dims dims = { size, 2 };
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

}

%typemap(in,
         fragment="Calin_Python_to_EigenMat")
         const Eigen::MatrixXd&
{
  $1 = new Eigen::MatrixXd();
  if(!calin_python_to_eigen_mat($input, *$1))SWIG_fail;
}

%typemap(freearg) const Eigen::MatrixXd&
{
  delete arg$argnum;
}

%typemap(in,
         fragment="Calin_Python_to_EigenMat")
         Eigen::MatrixXd&
{
  $1 = new Eigen::MatrixXd();
  if(!calin_python_to_eigen_mat($input, *$1))SWIG_fail;
}

%typemap(argout,
         fragment="Calin_Python_to_EigenMat")
         Eigen::MatrixXd&
{
  //  if(!calin_eigen_mat_to_python(*$1, $result))SWIG_fail;
  if(!calin_eigen_mat_to_python(*$1, $input))SWIG_fail;
}

%typemap(freearg)
         Eigen::MatrixXd&
{
  delete arg$argnum;
}

%typemap(out,
         fragment="Calin_Python_to_EigenMat")
         Eigen::MatrixXd
{
  npy_intp size[2] { $1.outerSize(), $1.innerSize() };
  $result = PyArray_EMPTY(2, size, NPY_DOUBLE, 0);
  if(!calin_eigen_mat_to_python($1, $result))SWIG_fail;
}

// =============================================================================
//
// Typemaps for using Eigen::Ref - these avoid copies if data types match
//
// =============================================================================

%typemap(in,
         fragment="NumPy_Fragments")
         const Eigen::Ref<const Eigen::VectorXd>&
         (PyArrayObject* array=NULL, int is_new_object=0,
          Eigen::Map<const Eigen::VectorXd>* eigenmap=0)
{
  npy_intp size[1] = { -1 };
  array = obj_to_array_contiguous_allow_conversion($input,
                                                   NPY_DOUBLE,
                                                   &is_new_object);
  if (!array || !require_dimensions(array, 1) ||
      !require_size(array, size, 1)) SWIG_fail;
  eigenmap =
      new Eigen::Map<const Eigen::VectorXd>((const double*)array_data(array),
                                            array_size(array,0));
  $1 = new $*1_ltype(*eigenmap);
}

%typemap(freearg) const Eigen::Ref<const Eigen::VectorXd>&
{
  delete arg$argnum;
  delete eigenmap$argnum;
  if (is_new_object$argnum && array$argnum)
    { Py_DECREF(array$argnum); }
}