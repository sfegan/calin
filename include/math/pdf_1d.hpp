/* 

   calin/math/pdf_1d.hpp -- Stephen Fegan -- 2015-04-02

   Base classes for one-dimensional PDF functions. PDF functions are
   based on ParameterizableSingleAxisFunction, and are assumed to
   integrate out to 1.0. They also (optionally) provide analytic
   moments.

*/

#pragma once

#include <string>
#include <vector>
#include <limits>

#include <Eigen/Core>

#include "function.hpp"

namespace calin { namespace math { namespace pdf_1d {

using function::MatRef;
using function::VecRef;
using function::ConstVecRef;
using function::ConstMatRef;
using function::ParameterAxis;

class Parameterizable1DPDF: public ParameterizableSingleAxisFunction
{
public:
  virtual ~Parameterizable1DPDF();
  
  // Reiterate functions from ParameterizableSingleAxisFunction

  unsigned num_parameters() override = 0;
  std::vector<ParameterAxis> parameters() override = 0;
  Eigen::VectorXd parameter_values() override = 0;
  void set_parameter_values(ConstVecRef values) override = 0;

  DomainAxis domain_axis() override = 0;

  bool can_calculate_gradient() override = 0;
  bool can_calculate_hessian() override = 0;
  bool can_calculate_parameter_gradient() override = 0;
  bool can_calculate_parameter_hessian() override = 0;

  double value(double x) override = 0;
  double value_and_gradient(double x,  double& dfdx) override = 0;
  double value_gradient_and_hessian(double x, double& dfdx,
                                    double& d2fdx2) override = 0;
  double value_and_parameter_gradient(double x,  VecRef gradient) override = 0;
  double value_parameter_gradient_and_hessian(double x, VecRef gradient,
                                              MatRef hessian) override = 0;

  double error_up() override = 0;

  // Moments

  virtual bool can_calculate_mean_and_variance() = 0;
  virtual void get_mean_and_variance(double& mean, double& var) = 0;
};

// *****************************************************************************
//
// Miscellaneous PDFs
//
// *****************************************************************************

class GaussianPDF: public Parameterizable1DPDF
{
 public:
  GaussianPDF(double error_up = 0.5):
      Parameterizable1DPDF(), error_up_(error_up)
  { /* nothing to see here */ }
  
  virtual ~GaussianPDF();

  unsigned num_parameters() override;
  std::vector<ParameterAxis> parameters() override;
  Eigen::VectorXd parameter_values() override;
  void set_parameter_values(ConstVecRef values) override;
  DomainAxis domain_axis() override;

  bool can_calculate_gradient() override;
  bool can_calculate_hessian() override;
  bool can_calculate_parameter_gradient() override;
  bool can_calculate_parameter_hessian() override;

  double value(double x) override;
  double value_and_gradient(double x,  double& dfdx) override;
  double value_gradient_and_hessian(double x, double& dfdx,
                                    double& d2fdx2) override;
  double value_and_parameter_gradient(double x,  VecRef gradient) override;
  double value_parameter_gradient_and_hessian(double x, VecRef gradient,
                                              MatRef hessian) override;

  double error_up() override;

  bool can_calculate_mean_and_variance() override;
  void get_mean_and_variance(double& mean, double& var) override;

 protected:
  double error_up_ = 0.5;
  double x0_ = 0;
  double s_ = 1;
};

class LimitedGaussianPDF: public GaussianPDF
{
 public:
  constexpr static double inf = std::numeric_limits<double>::infinity();

  LimitedGaussianPDF(double xlo, double xhi, double error_up = 0.5):
      GaussianPDF(error_up), xlo_(xlo), xhi_(xhi),
      norm_gradient_(2), norm_hessian_(2,2) { set_cache(); }
  
  virtual ~LimitedGaussianPDF();

  DomainAxis domain_axis() override;

  void set_parameter_values(ConstVecRef values) override;

  double value(double x) override;
  double value_and_gradient(double x,  double& dfdx) override;
  double value_gradient_and_hessian(double x, double& dfdx,
                                    double& d2fdx2) override;
  double value_and_parameter_gradient(double x,  VecRef gradient) override;
  double value_parameter_gradient_and_hessian(double x, VecRef gradient,
                                              MatRef hessian) override;

  bool can_calculate_mean_and_variance() override;
  void get_mean_and_variance(double& mean, double& var) override;
protected:
  void set_cache();
  double xlo_;
  double xhi_;
  double norm_ = 0;
  Eigen::VectorXd norm_gradient_;
  Eigen::MatrixXd norm_hessian_;
};

class LimitedExponentialPDF: public Parameterizable1DPDF
{
 public:
  constexpr static double inf = std::numeric_limits<double>::infinity();

  LimitedExponentialPDF(double xlo=0.0, double xhi=inf, double error_up = 0.5):
      Parameterizable1DPDF(), error_up_(error_up), xlo_(xlo), xhi_(xhi) {
    set_cache(); }

  virtual ~LimitedExponentialPDF();

  unsigned num_parameters() override;
  std::vector<ParameterAxis> parameters() override;
  Eigen::VectorXd parameter_values() override;
  void set_parameter_values(ConstVecRef values) override;
  DomainAxis domain_axis() override;

  bool can_calculate_gradient() override;
  bool can_calculate_hessian() override;
  bool can_calculate_parameter_gradient() override;
  bool can_calculate_parameter_hessian() override;

  double value(double x) override;
  double value_and_gradient(double x,  double& dfdx) override;
  double value_gradient_and_hessian(double x, double& dfdx,
                                    double& d2fdx2) override;
  double value_and_parameter_gradient(double x,  VecRef gradient) override;
  double value_parameter_gradient_and_hessian(double x, VecRef gradient,
                                              MatRef hessian) override;

  double error_up() override;

  bool can_calculate_mean_and_variance() override;
  void get_mean_and_variance(double& mean, double& var) override;  
 protected:
  void set_cache();
  double error_up_      = 0.5;
  double a_             = 1.0;
  double xlo_;
  double xhi_;
  double norm_          = 1.0;
  double norm_gradient_ = 0.0;
  double norm_hessian_  = 0.0;
};

} // namespace pdf_1d

using pdf_1d::Parameterizable1DPDF;

using pdf_1d::GaussianPDF;
using pdf_1d::LimitedGaussianPDF;
using pdf_1d::LimitedExponentialPDF;

} } // namespace calin::math