/* 

   calin/math/nlopt_optimizer.hpp -- Stephen Fegan -- 2015-03-16

   Interface to MIT NLOpt optimizer suite

*/

#pragma once

#include <Eigen/Dense>

#include "function.hpp"
#include "optimizer.hpp"
#include "nlopt/nlopt.hpp"

namespace calin { namespace math { namespace optimizer {

class NLOptOptimizer: public Optimizer
{
 public:
  using ConstVecRef = function::ConstVecRef;
  using VecRef = function::VecRef;
  using MatRef = function::MatRef;

  using algorithm_type = nlopt::algorithm;

  NLOptOptimizer(algorithm_type algorithm,
                 MultiAxisFunction* fcn, bool adopt_fcn = false);
  ~NLOptOptimizer();

  static bool requires_gradient(algorithm_type a);
  static bool requires_hessian(algorithm_type a);
  static bool requires_box_constraints(algorithm_type a);
  static bool can_estimate_error(algorithm_type a);
  static bool can_use_gradient(algorithm_type a);
  static bool can_use_hessian(algorithm_type a);
  static bool can_impose_box_constraints(algorithm_type a);
  
  bool requires_gradient() override;
  bool requires_hessian() override;
  bool requires_box_constraints() override;
  bool can_estimate_error() override;
  bool can_use_gradient() override;
  bool can_use_hessian() override;
  bool can_impose_box_constraints() override;

  bool minimize(VecRef xopt, double& fopt) override;
  ErrorMatrixStatus error_matrix_estimate(MatRef err_mat) override;
  ErrorMatrixStatus calc_error_matrix(MatRef err_mat) override;

 protected:
  static double nlopt_callback(unsigned n, const double* x, double* grad,
                               void* self);
  double eval_func(unsigned n, const double* x, double* grad);
  
  algorithm_type algorithm_;

  unsigned iter_;
  Eigen::VectorXd xopt_;
  std::unique_ptr<ErrorMatrixEstimator> err_est_;
};

} // namesace optimizer

using optimizer::NLOptOptimizer;

} } // namespace calin::math