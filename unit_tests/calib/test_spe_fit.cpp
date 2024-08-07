/*

   calin/unit_tests/calib/test_spe_fit.cpp -- Stephen Fegan -- 2015-03-15

   Unit tests for spe_fit classes

   Copyright 2015, Stephen Fegan <sfegan@llr.in2p3.fr>
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

#include <fstream>

#include <gsl/gsl_multimin.h>
#include <gtest/gtest.h>
#include "Eigen/Dense"
// #include "nlopt/nlopt.hpp"
#include "calib/spe_fit.hpp"
#include "karkar_data.hpp"
// #include "math/optimizer.hpp"
// #include "math/nlopt_optimizer.hpp"
// #include "math/cminpack_optimizer.hpp"

using namespace calin::math;
using namespace calin::math::histogram;
using namespace calin::calib;
using namespace calin::calib::spe_fit;
using namespace calin::unit_tests;

TEST(TestPoissonGaussianMES, SetAndRecallParameters) {
  PoissonGaussianMES pg_mes1;
  PoissonGaussianMES_HighAccuracy pg_mes2;

  Eigen::VectorXd params(5);
  params <<  1.0, 0.1, 0.2, 1.0, 0.45;
  pg_mes1.set_parameter_values(params);
  pg_mes2.set_parameter_values(params);

  Eigen::VectorXd params1 { pg_mes1.parameter_values() };
  Eigen::VectorXd params2 { pg_mes2.parameter_values() };

  EXPECT_EQ(params, params1);
  EXPECT_EQ(params, params2);
}

TEST(TestPoissonGaussianMES, PDFEqualityWithLegacyCode_Ped) {
  PoissonGaussianMES pg_mes1;
  PoissonGaussianMES_HighAccuracy pg_mes2;

  Eigen::VectorXd params(5);
  params <<  1.0, 0.1, 0.2, 1.0, 0.45;
  pg_mes1.set_parameter_values(params);
  pg_mes2.set_parameter_values(params);

  Eigen::VectorXd gradient(5);

  for(double x=-1.0;x<1.001;x+=0.01)
  {
    EXPECT_NEAR(pg_mes1.pdf_ped(x),pg_mes2.pdf_ped(x),pg_mes2.pdf_ped(x)*1e-8);
    EXPECT_NEAR(pg_mes1.pdf_ped(x),pg_mes1.pdf_gradient_ped(x,gradient),
                pg_mes1.pdf_ped(x)*1e-8);
    EXPECT_NEAR(pg_mes2.pdf_ped(x),pg_mes2.pdf_gradient_ped(x,gradient),
                pg_mes2.pdf_ped(x)*1e-8);
  }
}

TEST(TestPoissonGaussianMES, PDFEqualityWithLegacyCode_MES) {
  PoissonGaussianMES pg_mes1(20);
  PoissonGaussianMES_HighAccuracy pg_mes2;

  Eigen::VectorXd params(5);
  params <<  1.0, 0.1, 0.2, 1.0, 0.45;
  pg_mes1.set_parameter_values(params);
  pg_mes2.set_parameter_values(params);

  Eigen::VectorXd gradient(5);

  for(double x=-1.0;x<10.001;x+=0.01)
  {
    EXPECT_NEAR(pg_mes1.pdf_mes(x),pg_mes2.pdf_mes(x),pg_mes2.pdf_mes(x)*1e-8);
    EXPECT_NEAR(pg_mes1.pdf_mes(x),pg_mes1.pdf_gradient_mes(x,gradient),
                pg_mes1.pdf_mes(x)*1e-8);
    EXPECT_NEAR(pg_mes2.pdf_mes(x),pg_mes2.pdf_gradient_mes(x,gradient),
                pg_mes2.pdf_mes(x)*1e-8);
  }
}

namespace {

using namespace calin;

void
mes_gradient_test(MultiElectronSpectrum* mes,
                  double(MultiElectronSpectrum::*val_get_f)(double),
                  double(MultiElectronSpectrum::*grad_get_f)(double,VecRef) ,
        double(MultiElectronSpectrum::*hess_get_f)(double,VecRef,MatRef),
                  const std::vector<double> vp, const std::vector<double> vdp,
                  double xlo, double xhi, double dx, double good_max = 0.5)
{
  Eigen::VectorXd p(mes->num_parameters());
  Eigen::VectorXd dp(mes->num_parameters());
  for(unsigned ipar = 0; ipar<mes->num_parameters(); ipar++)
  {
    p[ipar] = vp[ipar];
    dp[ipar] = vdp[ipar];
  }
  for(double xval = xlo; xval<xhi; xval+=dx)
  {
    bool check_ok;
    Eigen::VectorXd good(mes->num_parameters());
    function::ValGetter<MultiElectronSpectrum> val_get =
        [xval,val_get_f](MultiElectronSpectrum* mes) {
      return (mes->*val_get_f)(xval); };
    function::GradGetter<MultiElectronSpectrum>  grad_get =
        [xval,grad_get_f](MultiElectronSpectrum* mes, VecRef grad) {
      return (mes->*grad_get_f)(xval,grad); };
    function::HessGetter<MultiElectronSpectrum> hess_get =
      [xval,hess_get_f](MultiElectronSpectrum* mes, VecRef grad, MatRef hess) {
      return (mes->*hess_get_f)(xval,grad,hess); };

    check_ok = function::gradient_check_par(*mes, p, dp, good,
                                            val_get, grad_get, hess_get,
                                            good_max);
    EXPECT_TRUE(check_ok);

    for(unsigned ipar=0;ipar<mes->num_parameters();ipar++)
      if(good(ipar)>good_max)
      {
        std::cout << "At x = " << xval << '\n';
        break;
      }
    for(unsigned ipar=0;ipar<mes->num_parameters();ipar++)
      EXPECT_LE(good(ipar),good_max);
  }
}

} // anonymous namespace

TEST(TestPoissonGaussianMES, GradientCheck_PED)
{
  double dp1 = 1e-7;
  PoissonGaussianMES mes(40);
  mes_gradient_test(&mes,
                    &MultiElectronSpectrum::pdf_ped,
                    &MultiElectronSpectrum::pdf_gradient_ped,
                    &MultiElectronSpectrum::pdf_gradient_hessian_ped,
                    { 1.123, 0.100000, 0.2, 1.321, 0.45 },
                    { dp1, dp1, dp1, dp1, dp1}, -1.0, 1.0, 0.1);
}

TEST(TestPoissonGaussianMES, GradientCheck_MES)
{
  double dp1 = 1e-7;
  PoissonGaussianMES mes(40);
  mes_gradient_test(&mes,
                    &MultiElectronSpectrum::pdf_mes,
                    &MultiElectronSpectrum::pdf_gradient_mes,
                    &MultiElectronSpectrum::pdf_gradient_hessian_mes,
                    { 1.123, 0.100000, 0.2, 1.321, 0.45 },
                    { dp1, dp1, dp1, dp1, dp1}, -1.0, 10.0, 0.1);
}

TEST(TestPoissonGaussianMES_HighAccuracy, GradientCheck_PED)
{
  double dp1 = 1e-7;
  PoissonGaussianMES_HighAccuracy mes(1e-20);
  mes_gradient_test(&mes,
                    &MultiElectronSpectrum::pdf_ped,
                    &MultiElectronSpectrum::pdf_gradient_ped,
                    &MultiElectronSpectrum::pdf_gradient_hessian_ped,
                    { 1.123, 0.100000, 0.2, 1.321, 0.45 },
                    { dp1, dp1, dp1, dp1, dp1}, -1.0, 1.0, 0.1);
}

TEST(TestPoissonGaussianMES_HighAccuracy, GradientCheck_MES)
{
  double dp1 = 1e-7;
  PoissonGaussianMES_HighAccuracy mes(1e-20);
  mes_gradient_test(&mes,
                    &MultiElectronSpectrum::pdf_mes,
                    &MultiElectronSpectrum::pdf_gradient_mes,
                    &MultiElectronSpectrum::pdf_gradient_hessian_mes,
                    { 1.123, 0.100000, 0.2, 1.321, 0.45 },
                    { dp1, dp1, dp1, dp1, dp1}, -1.0, 10.0, 0.1,
                    /* relax required accuracy here for Travis CI : */ 1.0);
}

namespace {

using namespace calin;

void
mes_hessian_test(MultiElectronSpectrum* mes,
                 double(MultiElectronSpectrum::*val_get_f)(double),
                 double(MultiElectronSpectrum::*grad_get_f)(double,VecRef) ,
        double(MultiElectronSpectrum::*hess_get_f)(double,VecRef,MatRef),
                 const std::vector<double> vp, const std::vector<double> vdp,
                 double xlo, double xhi, double dx, double good_max = 0.5)
{
  Eigen::VectorXd p(5);
  p << vp[0], vp[1], vp[2], vp[3], vp[4];
  Eigen::VectorXd dp(5);
  dp << vdp[0], vdp[1], vdp[2], vdp[3], vdp[4];
  for(double xval = xlo; xval<xhi; xval+=dx)
  {
    bool check_ok;
    Eigen::MatrixXd good(5,5);
    function::ValGetter<MultiElectronSpectrum> val_get =
        [xval,val_get_f](MultiElectronSpectrum* mes) {
      return (mes->*val_get_f)(xval); };
    function::GradGetter<MultiElectronSpectrum>  grad_get =
        [xval,grad_get_f](MultiElectronSpectrum* mes, VecRef grad) {
      return (mes->*grad_get_f)(xval,grad); };
    function::HessGetter<MultiElectronSpectrum> hess_get =
      [xval,hess_get_f](MultiElectronSpectrum* mes, VecRef grad, MatRef hess) {
      return (mes->*hess_get_f)(xval,grad,hess); };

    check_ok = function::hessian_check_par(*mes, p, dp, good,
                                           val_get, grad_get, hess_get);

    EXPECT_TRUE(check_ok);
    for(unsigned ipar=0;ipar<5;ipar++)
      for(unsigned jpar=0;jpar<5;jpar++)EXPECT_LE(good(ipar,jpar), good_max);
  }
}

} // anonymous namespace

TEST(TestPoissonGaussianMES, HessianCheck_PED)
{
  double dp1 = 1e-7;
  PoissonGaussianMES mes(40);
  mes_hessian_test(&mes,
                   &MultiElectronSpectrum::pdf_ped,
                   &MultiElectronSpectrum::pdf_gradient_ped,
                   &MultiElectronSpectrum::pdf_gradient_hessian_ped,
                   { 1.123, 0.100000, 0.2, 1.321, 0.45 },
                   { dp1, dp1, dp1, dp1, dp1}, -1.0, 1.0, 0.1);
}

TEST(TestPoissonGaussianMES, HessianCheck_MES)
{
  double dp1 = 1e-7;
  PoissonGaussianMES mes(40);
  mes_hessian_test(&mes,
                   &MultiElectronSpectrum::pdf_mes,
                   &MultiElectronSpectrum::pdf_gradient_mes,
                   &MultiElectronSpectrum::pdf_gradient_hessian_mes,
                   { 1.123, 0.100000, 0.2, 1.321, 0.45 },
                   { dp1, dp1, dp1, dp1, dp1}, -1.0, 10.0, 0.1, 1.0);
}

TEST(TestSPELikelihood, KarkarPG_GradientCheck) {
  auto mes_data = karkar_data();
  SimpleHist mes_hist(1.0);
  for(auto idata : mes_data)mes_hist.insert(idata);
  PoissonGaussianMES mes_model(20);
  SPELikelihood like(mes_model, mes_hist);
  Eigen::VectorXd x(5);
  x << 0.55349034289601895, 3094.2718624743093,
      19.614139336940855, 89.181964780087668, 0.32388058781378032;
  Eigen::VectorXd dx(5);
  dx << 1e-7, 1e-7, 1e-7, 1e-7, 1e-7;
  Eigen::VectorXd good(5);

  bool check_ok = function::gradient_check(like, x, dx, good);
  EXPECT_TRUE(check_ok);
  for(unsigned ipar=0;ipar<5;ipar++)EXPECT_LE(good(ipar),0.5);
}

TEST(TestSPELikelihood, KarkarPG_HessianCheck) {
  auto mes_data = karkar_data();
  SimpleHist mes_hist(1.0);
  for(auto idata : mes_data)mes_hist.insert(idata);
  PoissonGaussianMES mes_model(20);
  SPELikelihood like(mes_model, mes_hist);
  Eigen::VectorXd x(5);
  x << 0.55349034289601895, 3094.2718624743093,
      19.614139336940855, 89.181964780087668, 0.32388058781378032;
  Eigen::VectorXd dx(5);
  dx << 1e-7, 1e-7, 1e-7, 1e-7, 1e-7;
  Eigen::MatrixXd good(5,5);

  bool check_ok = function::hessian_check(like, x, dx, good);
  EXPECT_TRUE(check_ok);
  for(unsigned ipar=0;ipar<5;ipar++)
    for(unsigned jpar=0;jpar<5;jpar++)EXPECT_LE(good(ipar,jpar),0.5);
}

TEST(TestSPELikelihood, KarkarPG_GradientCheck_WithPed) {
  auto mes_data = karkar_data();
  SimpleHist mes_hist(1.0);
  for(auto idata : mes_data)mes_hist.insert(idata);
  PoissonGaussianMES mes_model(20);
  SPELikelihood like(mes_model, mes_hist, mes_hist);
  Eigen::VectorXd x(5);
  x << 0.55349034289601895, 3094.2718624743093,
      19.614139336940855, 89.181964780087668, 0.32388058781378032;
  Eigen::VectorXd dx(5);
  dx << 1e-7, 1e-7, 1e-7, 1e-7, 1e-7;
  Eigen::VectorXd good(5);

  bool check_ok = function::gradient_check(like, x, dx, good);
  EXPECT_TRUE(check_ok);
  for(unsigned ipar=0;ipar<5;ipar++)EXPECT_LE(good(ipar),0.5);
}

#if 0

double my_f(const gsl_vector * x, void * params)
{
  SPELikelihood* like = static_cast<SPELikelihood*>(params);
  double value = like->value(x->data);
#if 0
  std::cout << value << ' '
            << x->data[0] << ' '
            << x->data[1] << ' '
            << x->data[2] << ' '
            << x->data[3] << ' '
            << x->data[4] << '\n';
#endif
  return value;
}

void my_df(const gsl_vector * x, void * params, gsl_vector* g)
{
  SPELikelihood* like = static_cast<SPELikelihood*>(params);
  like->value_and_gradient(x->data, g->data);
#if 0
  std::cout << g->data[0] << ' '
            << g->data[1] << ' '
            << g->data[2] << ' '
            << g->data[3] << ' '
            << g->data[4] << ' '
            << x->data[0] << ' '
            << x->data[1] << ' '
            << x->data[2] << ' '
            << x->data[3] << ' '
            << x->data[4] << '\n';
#endif
}

void my_fdf(const gsl_vector * x, void * params, double* value, gsl_vector* g)
{
  SPELikelihood* like = static_cast<SPELikelihood*>(params);
  *value = like->value_and_gradient(x->data, g->data);
#if 0
  std::cout << *value << ' '
            << g->data[0] << ' '
            << g->data[1] << ' '
            << g->data[2] << ' '
            << g->data[3] << ' '
            << g->data[4] << ' '
            << x->data[0] << ' '
            << x->data[1] << ' '
            << x->data[2] << ' '
            << x->data[3] << ' '
            << x->data[4] << '\n';
#endif
}

}

TEST(TestSPELikelihood, Minimize_GSL_Simplex)
{
  auto mes_data = karkar_data();
  SimpleHist mes_hist(1.0);
  for(auto idata : mes_data)mes_hist.insert(idata);
  PoissonGaussianMES mes_model(20);
  SPELikelihood like(mes_model, mes_hist);

  const gsl_multimin_fminimizer_type *T =
    gsl_multimin_fminimizer_nmsimplex2;
  gsl_multimin_fminimizer *s = NULL;
  gsl_vector *ss,*x;
  gsl_multimin_function minex_func;

  size_t iter = 0;
  int status;
  double size;

  /* Starting point */
  x = gsl_vector_alloc (5);
  gsl_vector_set (x, 0, 1.0);
  gsl_vector_set (x, 1, 3100.0);
  gsl_vector_set (x, 2, 20.0);
  gsl_vector_set (x, 3, 100.0);
  gsl_vector_set (x, 4, 0.45);

  /* Step size */
  ss = gsl_vector_alloc (5);
  gsl_vector_set (ss, 0, 0.1);
  gsl_vector_set (ss, 1, 10.0);
  gsl_vector_set (ss, 2, 1.0);
  gsl_vector_set (ss, 3, 10.0);
  gsl_vector_set (ss, 4, 0.05);

  /* Initialize method and iterate */
  minex_func.n = 5;
  minex_func.f = my_f;
  minex_func.params = &like;

  s = gsl_multimin_fminimizer_alloc (T, 5);
  gsl_multimin_fminimizer_set (s, &minex_func, x, ss);

  do
    {
      iter++;
      status = gsl_multimin_fminimizer_iterate(s);

      if (status)
        break;

      size = gsl_multimin_fminimizer_size (s);
      status = gsl_multimin_test_size (size, 1e-3);

#if 0
      if (status == GSL_SUCCESS)
        {
          printf ("converged to minimum at\n");
        }

      printf ("%5lu %10.8f %10.4f %10.7f %10.7f %10.8f f() = %7.3f size = %.3f\n",
              iter,
>              gsl_vector_get (s->x, 0),
              gsl_vector_get (s->x, 1),
              gsl_vector_get (s->x, 2),
              gsl_vector_get (s->x, 3),
              gsl_vector_get (s->x, 4),
              s->fval, size);
#endif
    }
  while (status == GSL_CONTINUE && iter < 10000);

  EXPECT_EQ(status, GSL_SUCCESS);
  EXPECT_NEAR(s->x->data[0], 0.55349337, 0.0001);
  EXPECT_NEAR(s->x->data[1], 3094.2715, 0.01);
  EXPECT_NEAR(s->x->data[2], 19.6141970, 0.001);
  EXPECT_NEAR(s->x->data[3], 89.1810077, 0.01);
  EXPECT_NEAR(s->x->data[4], 0.32388838, 0.0001);

  gsl_vector_free(x);
  gsl_vector_free(ss);
  gsl_multimin_fminimizer_free (s);
}

TEST(TestSPELikelihood, Minimize_GSL_BFGS2)
{
  auto mes_data = karkar_data();
  SimpleHist mes_hist(1.0);
  for(auto idata : mes_data)mes_hist.insert(idata);
  PoissonGaussianMES mes_model(20);
  //PoissonGaussianMES_HighAccuracy mes_model;
  SPELikelihood like(mes_model, mes_hist);

  const gsl_multimin_fdfminimizer_type *T = nullptr;
  gsl_multimin_fdfminimizer *s = nullptr;

  gsl_vector *ss, *x;
  gsl_multimin_function_fdf minex_func;

  size_t iter = 0;
  int status;
  double size;

  /* Starting point */
  x = gsl_vector_alloc (5);
  gsl_vector_set (x, 0, 1.0);
  gsl_vector_set (x, 1, 3100.0);
  gsl_vector_set (x, 2, 20.0);
  gsl_vector_set (x, 3, 100.0);
  gsl_vector_set (x, 4, 0.45);

  /* Initialize method and iterate */
  minex_func.n = 5;
  minex_func.f = my_f;
  minex_func.df = my_df;
  minex_func.fdf = my_fdf;
  minex_func.params = &like;

  T = gsl_multimin_fdfminimizer_vector_bfgs2;
  //T = gsl_multimin_fdfminimizer_conjugate_fr;
  //T = gsl_multimin_fdfminimizer_steepest_descent;
  s = gsl_multimin_fdfminimizer_alloc (T, 5);

  gsl_multimin_fdfminimizer_set (s, &minex_func, x, .01, .01);

  do
  {
    iter++;
    status = gsl_multimin_fdfminimizer_iterate (s);

    if (status)
      break;

    status = gsl_multimin_test_gradient (s->gradient, 0.1);

#if 0
    if (status == GSL_SUCCESS)
      printf ("Minimum found at:\n");

    printf ("%5lu %10.8f %10.4f %10.7f %10.7f %10.8f f() = %7.3f\n",
            iter,
            gsl_vector_get (s->x, 0),
            gsl_vector_get (s->x, 1),
            gsl_vector_get (s->x, 2),
            gsl_vector_get (s->x, 3),
            gsl_vector_get (s->x, 4),
            s->f);
#endif
  }
  while (status == GSL_CONTINUE && iter < 10000);

  EXPECT_EQ(status, GSL_SUCCESS);
  EXPECT_NEAR(s->x->data[0], 0.55349337, 0.0001);
  EXPECT_NEAR(s->x->data[1], 3094.2715, 0.01);
  EXPECT_NEAR(s->x->data[2], 19.6141970, 0.001);
  EXPECT_NEAR(s->x->data[3], 89.1810077, 0.01);
  EXPECT_NEAR(s->x->data[4], 0.32388838, 0.0001);

  gsl_multimin_fdfminimizer_free (s);
  gsl_vector_free (x);
}
#endif

#if 0
TEST(TestSPELikelihood, Optimize_NLOpt)
{
  auto mes_data = karkar_data();
  SimpleHist mes_hist(1.0);
  for(auto idata : mes_data)mes_hist.insert(idata);
  PoissonGaussianMES mes_model(20);
  SPELikelihood like(mes_model, mes_hist);

  //optimizer::NLOptOptimizer opt("LN_SBPLX", &like);
  optimizer::NLOptOptimizer opt("LD_LBFGS", &like);
  opt.set_scale({0.1,0.1,1.0,1.0,0.05});
  //opt.set_verbosity_level(optimizer::OptimizerVerbosityLevel::MAX);
  opt.set_verbosity_level(optimizer::OptimizerVerbosityLevel::SILENT);
  opt.set_abs_tolerance(0.0001);
  opt.set_initial_values(std::vector<double>{ 1.0, 3100.0, 20.0, 100.0, 0.45 });
  Eigen::VectorXd x_opt(5);
  double f_val;
  opt.minimize(x_opt, f_val);

  //EXPECT_EQ(status, GSL_SUCCESS);
  EXPECT_NEAR(x_opt[0], 0.55349337, 0.0001);
  EXPECT_NEAR(x_opt[1], 3094.2715, 0.01);
  EXPECT_NEAR(x_opt[2], 19.6141970, 0.001);
  EXPECT_NEAR(x_opt[3], 89.1810077, 0.01);
  EXPECT_NEAR(x_opt[4], 0.32388838, 0.0001);

  // std::cout << std::fixed << std::setprecision(3);
  // std::cout << x_opt[0] << ' ' << x_opt[1] << ' ' << x_opt[2] << ' '
  //           << x_opt[3] << ' ' << x_opt[4] << '\n';

  Eigen::MatrixXd hessian_mat(5,5);
  Eigen::VectorXd gradient(5);
  like.value_gradient_and_hessian(x_opt, gradient, hessian_mat);
  Eigen::MatrixXd err_mat = hessian_mat.inverse();
  // std::cout << std::scientific << std::setprecision(8) << err_mat << '\n';
  //
  // std::cout << std::sqrt(err_mat(0,0)) << ' '
  //           << std::sqrt(err_mat(1,1)) << ' '
  //           << std::sqrt(err_mat(2,2)) << ' '
  //           << std::sqrt(err_mat(3,3)) << ' '
  //           << std::sqrt(err_mat(4,4)) << '\n';
}

TEST(TestSPELikelihood, Optimize_CMinpack)
{
  auto mes_data = karkar_data();
  SimpleHist mes_hist(1.0);
  for(auto idata : mes_data)mes_hist.insert(idata);
  PoissonGaussianMES mes_model(20);
  SPELikelihood like(mes_model, mes_hist);

  //optimizer::NLOptOptimizer opt("LN_SBPLX", &like);
  optimizer::CMinpackOptimizer opt(&like);
  opt.set_scale({0.1,0.1,1.0,1.0,0.05});
  //opt.set_verbosity_level(optimizer::OptimizerVerbosityLevel::MAX);
  opt.set_verbosity_level(optimizer::OptimizerVerbosityLevel::SILENT);
  opt.set_abs_tolerance(0.0001);
  opt.set_initial_values(std::vector<double>{ 0.5, 3100.0, 20.0, 90.0, 0.35 });
  Eigen::VectorXd x_opt(5);
  double f_val;
  opt.minimize(x_opt, f_val);

  //EXPECT_EQ(status, GSL_SUCCESS);
  EXPECT_NEAR(x_opt[0], 0.55349337, 0.0001);
  EXPECT_NEAR(x_opt[1], 3094.2715, 0.01);
  EXPECT_NEAR(x_opt[2], 19.6141970, 0.001);
  EXPECT_NEAR(x_opt[3], 89.1810077, 0.01);
  EXPECT_NEAR(x_opt[4], 0.32388838, 0.0001);

  // std::cout << f_val << ' ' << std::fixed << std::setprecision(3);
  // std::cout << x_opt[0] << ' ' << x_opt[1] << ' ' << x_opt[2] << ' '
  //           << x_opt[3] << ' ' << x_opt[4] << '\n';

  Eigen::MatrixXd hessian_mat;
  Eigen::VectorXd gradient;
  like.value_gradient_and_hessian(x_opt, gradient, hessian_mat);
  Eigen::MatrixXd err_mat = hessian_mat.inverse();
  // std::cout << std::fixed << std::setprecision(9) << err_mat << '\n';
}
#endif

TEST(TestGeneralPoissonMES_Gauss, SetAndRecallParameters) {
  pdf_1d::GaussianPDF ped;
  pdf_1d::LimitedGaussianPDF ses(0,std::numeric_limits<double>::infinity());
  GeneralPoissonMES mes(0, 1, 1025, &ses, &ped);
  EXPECT_EQ(mes.num_parameters(), 5U);
  Eigen::VectorXd p(5);
  p << 1.0, 100.0, 20.0, 100.0, 35.0;
  mes.set_parameter_values(p);
  EXPECT_EQ(mes.parameter_values(), p);
#if 0
  std::ofstream file("spec.dat");
  std::vector<double> mes_spec = mes.multi_electron_spectrum();
  std::vector<double> ped_spec = mes.pedestal_spectrum();
  std::vector<double> one_es_spec = mes.n_electron_spectrum(1);
  std::vector<double> two_es_spec = mes.n_electron_spectrum(2);
  for(unsigned i=0;i<mes_spec.size();i++)
    file << mes_spec[i] << ' ' << ped_spec[i] << ' '
         << one_es_spec[i] << ' ' << two_es_spec[i] << '\n';
#endif
}

#if 0
TEST(TestGeneralPoissonMES_Gauss, Optimize_NLOpt_Simplex)
{
  auto mes_data = karkar_data();
  SimpleHist mes_hist(1.0);
  for(auto idata : mes_data)mes_hist.insert(idata);
  pdf_1d::GaussianPDF ped;
  pdf_1d::LimitedGaussianPDF ses(0,std::numeric_limits<double>::infinity());
  GeneralPoissonMES mes_model(mes_hist.xval0(), mes_hist.dxval(),
                              mes_hist.size(), &ses, &ped);

  SPELikelihood like(mes_model, mes_hist);

  optimizer::NLOptOptimizer opt("LN_SBPLX", &like);
  //optimizer::NLOptOptimizer opt("LD_LBFGS", &like);
  opt.set_scale({0.1,0.1,1.0,1.0,0.05});
  // opt.set_verbosity_level(optimizer::OptimizerVerbosityLevel::SUMMARY_AND_PROGRESS);
  opt.set_verbosity_level(optimizer::OptimizerVerbosityLevel::SILENT);
  opt.set_abs_tolerance(0.0001);
  opt.set_initial_values(std::vector<double>{ 1.0, 3100.0, 20.0, 100.0, 45.0 });
  Eigen::VectorXd x_opt(5);
  double f_val;
  opt.minimize(x_opt, f_val);
}
#endif

TEST(TestGeneralPoissonMES_ExpGauss, GradientCheck_MES)
{
  double inf = std::numeric_limits<double>::infinity();
  pdf_1d::GaussianPDF ped;
  pdf_1d::LimitedExponentialPDF exp_pdf(0,inf);
  exp_pdf.limit_scale(0.1, inf);
  pdf_1d::LimitedGaussianPDF gauss_pdf(0,inf);
  pdf_1d::TwoComponent1DPDF ses(&exp_pdf, "exp", &gauss_pdf, "gauss");
  GeneralPoissonMES mes_model(-1.1,0.01,1024,&ses,&ped);
  double dp1 = 1e-7;
  mes_gradient_test(&mes_model,
                    &MultiElectronSpectrum::pdf_mes,
                    &MultiElectronSpectrum::pdf_gradient_mes,
                    &MultiElectronSpectrum::pdf_gradient_hessian_mes,
                    { 1.123, 0.100000, 0.2, 0.3, 0.2, 1.321, 0.35 },
                    { dp1, dp1, dp1, dp1, dp1, dp1, dp1}, -1.0, 9.0, 1.0);

  // std::ofstream file("spec.dat");
  // Eigen::VectorXd mes_spec = mes_model.multi_electron_spectrum();
  // Eigen::VectorXd ped_spec = mes_model.pedestal_spectrum();
  // Eigen::VectorXd one_es_spec = mes_model.n_electron_spectrum(1);
  // //one_es_spec = mes_model.n_electron_spectrum(1);
  // Eigen::VectorXd two_es_spec = mes_model.n_electron_spectrum(2);
  // Eigen::VectorXd three_es_spec = mes_model.n_electron_spectrum(3);
  // for(unsigned i=0;i<mes_spec.size();i++)
  //   file << mes_spec[i] << ' ' << ped_spec[i] << ' '
  //        << one_es_spec[i] << ' ' << two_es_spec[i] << ' '
  //        << three_es_spec[i] << ' ' << '\n';
}

TEST(TestGeneralPoissonMES_ExpGauss, GradientCheck_PED)
{
  double inf = std::numeric_limits<double>::infinity();
  pdf_1d::GaussianPDF ped;
  pdf_1d::LimitedExponentialPDF exp_pdf(0,inf);
  exp_pdf.limit_scale(0.1, inf);
  pdf_1d::LimitedGaussianPDF gauss_pdf(0,inf);
  pdf_1d::TwoComponent1DPDF ses(&exp_pdf, "exp", &gauss_pdf, "gauss");
  GeneralPoissonMES mes_model(-1.1,0.01,1024,&ses,&ped);
  double dp1 = 1e-7;
  mes_gradient_test(&mes_model,
                    &MultiElectronSpectrum::pdf_ped,
                    &MultiElectronSpectrum::pdf_gradient_ped,
                    &MultiElectronSpectrum::pdf_gradient_hessian_ped,
                    { 1.123, 0.100000, 0.2, 0.3, 0.2, 1.321, 0.35 },
                    { dp1, dp1, dp1, dp1, dp1, dp1, dp1}, -1.0, 9.0, 0.5);
}

TEST(TestGeneralPoissonMES_ExpGauss, Repeatability)
{
  double inf = std::numeric_limits<double>::infinity();
  std::vector<double> all_val;
  std::vector<Eigen::VectorXd> all_grad;
  for(unsigned iloop=0;iloop<10;iloop++)
  {
    auto mes_data = karkar_data();
    SimpleHist mes_hist(1.0);
    for(auto idata : mes_data)mes_hist.insert(idata);
    pdf_1d::GaussianPDF ped;
    pdf_1d::LimitedExponentialPDF exp_pdf(0,inf,mes_hist.dxval());
    exp_pdf.limit_scale(0.1, inf);
    pdf_1d::LimitedGaussianPDF gauss_pdf(0,inf);
    pdf_1d::TwoComponent1DPDF ses(&exp_pdf, "exp", &gauss_pdf, "gauss");
    GeneralPoissonMES mes_model(mes_hist.xval_left(0),
                                mes_hist.dxval(),
                                mes_hist.size(), &ses, &ped);
    SPELikelihood like(mes_model, mes_hist);
    Eigen::VectorXd p(7);
    p << 0.56, 3094.7, 19.6, 0.1, 5.0, 88.9, 29.3;

    Eigen::VectorXd grad(7);
    double val = like.value_and_gradient(p,grad);

    all_val.push_back(val);
    all_grad.push_back(grad);
  }

  EXPECT_EQ(std::count(all_val.begin(), all_val.end(), all_val.front()),
            (int)all_val.size());

  EXPECT_EQ(std::count(all_grad.begin(), all_grad.end(), all_grad.front()),
            (int)all_grad.size());
}

#if 0
TEST(TestGeneralPoissonMES_ExpGauss, Optimize_NLOpt_Simplex)
{
  double inf = std::numeric_limits<double>::infinity();
  auto mes_data = karkar_data();
  SimpleHist mes_hist(1.0);
  for(auto idata : mes_data)mes_hist.insert(idata);
  pdf_1d::GaussianPDF ped;
  pdf_1d::LimitedExponentialPDF exp_pdf(0,inf,mes_hist.dxval());
  exp_pdf.limit_scale(0.1, inf);
  pdf_1d::LimitedGaussianPDF gauss_pdf(0,inf);
  pdf_1d::TwoComponent1DPDF ses(&exp_pdf, "exp", &gauss_pdf, "gauss");
  GeneralPoissonMES mes_model(mes_hist.xval_left(0),
                              mes_hist.dxval(),
                              mes_hist.size(), &ses, &ped);
  SPELikelihood like(mes_model, mes_hist);

  //optimizer::NLOptOptimizer opt("LN_SBPLX", &like);
  optimizer::NLOptOptimizer opt("LD_LBFGS", &like);
  opt.set_scale({0.01,0.1,1.0,0.01,0.1,1.0,1.0});
  // opt.set_verbosity_level(optimizer::OptimizerVerbosityLevel::MAX);
  opt.set_verbosity_level(optimizer::OptimizerVerbosityLevel::SILENT);
  opt.set_abs_tolerance(0.001);
  opt.set_initial_values(std::vector<double>{ 1.0, 3100.0, 20.0, 0.05, 50.0, 100.0, 45.0 });
  opt.set_initial_values(std::vector<double>{ 0.56, 3094.7, 19.6, 0.1, 5.0, 88.9, 29.3 });
  opt.set_limits_lo({ 0.001, 3000.0,   1.0, 0.0,    1.0,  10.0,  10.0});
  opt.set_limits_hi({ 100.0, 3200.0, 100.0, 1.0, 1000.0, 300.0, 100.0});
  Eigen::VectorXd x_opt(7);
  double f_val;

  try {
    opt.minimize(x_opt, f_val);
  }
  catch(const nlopt::forced_stop& x)
  {
    std::cout << "Caught: nlopt::forced_stop: " << x.what() << '\n';
    throw;
  }
  catch(const nlopt::roundoff_limited& x)
  {
    std::cout << "Caught: nlopt::roundoff_limited: " << x.what() << '\n';
    throw;
  }
  catch(const std::runtime_error& x)
  {
    std::cout << "Caught: runtime_error: " << x.what() << '\n';
    throw;
  }

#if 0
  Eigen::VectorXd p(7);
  p << 1.0, 3100.0, 10.0, 0.4, 50.0, 100.0, 25.0;
  //mes_model.set_parameter_values(p);
  std::ofstream file("spec.dat");
  Eigen::VectorXd mes_spec = mes_model.multi_electron_spectrum();
  Eigen::VectorXd ped_spec = mes_model.pedestal_spectrum();
  Eigen::VectorXd one_es_spec = mes_model.n_electron_spectrum(1);
  Eigen::VectorXd two_es_spec = mes_model.n_electron_spectrum(2);
  Eigen::VectorXd three_es_spec = mes_model.n_electron_spectrum(3);
  Eigen::VectorXd zero_es_cpt = mes_model.mes_n_electron_cpt(0);
  Eigen::VectorXd one_es_cpt = mes_model.mes_n_electron_cpt(1);
  Eigen::VectorXd two_es_cpt = mes_model.mes_n_electron_cpt(2);
  Eigen::VectorXd three_es_cpt = mes_model.mes_n_electron_cpt(3);

  for(unsigned i=0;i<mes_spec.size();i++)
    file << mes_spec[i] << ' '
         << ped_spec[i] << ' ' << one_es_spec[i] << ' '
         << two_es_spec[i] << ' ' << three_es_spec[i] << ' '
         << zero_es_cpt[i] << ' ' << one_es_cpt[i] << ' '
         << two_es_cpt[i] << ' ' << three_es_cpt[i] << ' ' << '\n';
#endif
}
#endif

TEST(TestGeneralPoissonMES_GaussWithShift, SetAndRecallParameters) {
  pdf_1d::GaussianPDF ped;
  pdf_1d::LimitedGaussianPDF ses(0,std::numeric_limits<double>::infinity());
  auto opt = GeneralPoissonMES::default_config();
  opt.set_include_on_off_ped_shift(true);
  GeneralPoissonMES mes(0, 1, 1025, &ses, &ped, opt);
  EXPECT_EQ(mes.num_parameters(), 6U);
  Eigen::VectorXd p(6);
  p << 1.0, -10.0, 100.0, 20.0, 100.0, 35.0;
  mes.set_parameter_values(p);
  EXPECT_EQ(mes.parameter_values(), p);
  Eigen::VectorXd ped_spec = mes.pedestal_spectrum();
  Eigen::VectorXd off_spec = mes.off_pedestal_spectrum();
  double ped_sum_w = 0;
  double ped_sum_wx = 0;
  for(unsigned i=0; i<ped_spec.size(); i++)
    ped_sum_w += ped_spec[i], ped_sum_wx += ped_spec[i] * (double(i)+0.5);
  double off_sum_w = 0;
  double off_sum_wx = 0;
  for(unsigned i=0; i<ped_spec.size(); i++)
    off_sum_w += off_spec[i], off_sum_wx += off_spec[i] * (double(i)+0.5);
  std::cout << ped_sum_wx/ped_sum_w << ' ' << off_sum_wx/off_sum_w << '\n';
  Eigen::VectorXd grad(6);
  mes.pdf_gradient_mes(10.0, grad);
  // std::cout << grad.transpose() << '\n';
  mes.pdf_gradient_ped(10.0, grad);
  // std::cout << grad.transpose() << '\n';
  assert(ped_sum_w > 0);
  assert(ped_sum_wx > 0);
  assert(off_sum_w > 0);
  assert(off_sum_wx > 0);
}

TEST(TestGeneralPoissonMES_ExpGaussWithShift, GradientCheck_MES)
{
  double inf = std::numeric_limits<double>::infinity();
  pdf_1d::GaussianPDF ped;
  pdf_1d::LimitedExponentialPDF exp_pdf(0,inf);
  exp_pdf.limit_scale(0.1, inf);
  pdf_1d::LimitedGaussianPDF gauss_pdf(0,inf);
  pdf_1d::TwoComponent1DPDF ses(&exp_pdf, "exp", &gauss_pdf, "gauss");
  auto opt = GeneralPoissonMES::default_config();
  opt.set_include_on_off_ped_shift(true);
  GeneralPoissonMES mes_model(-1.1,0.01,1024,&ses,&ped,opt);
  double dp1 = 1e-7;
  mes_gradient_test(&mes_model,
                    &MultiElectronSpectrum::pdf_mes,
                    &MultiElectronSpectrum::pdf_gradient_mes,
                    &MultiElectronSpectrum::pdf_gradient_hessian_mes,
                    { 1.123, 10.0, 0.100000, 0.2, 0.3, 0.2, 1.321, 0.35 },
                    { dp1, dp1, dp1, dp1, dp1, dp1, dp1, dp1}, -1.0, 9.0, 1.0);
}

TEST(TestGeneralPoissonMES_ExpGaussWithShift, GradientCheck_PED)
{
  double inf = std::numeric_limits<double>::infinity();
  pdf_1d::GaussianPDF ped;
  pdf_1d::LimitedExponentialPDF exp_pdf(0,inf);
  exp_pdf.limit_scale(0.1, inf);
  pdf_1d::LimitedGaussianPDF gauss_pdf(0,inf);
  pdf_1d::TwoComponent1DPDF ses(&exp_pdf, "exp", &gauss_pdf, "gauss");
  auto opt = GeneralPoissonMES::default_config();
  opt.set_include_on_off_ped_shift(true);
  GeneralPoissonMES mes_model(-1.1,0.01,1024,&ses,&ped,opt);
  double dp1 = 1e-7;
  mes_gradient_test(&mes_model,
                    &MultiElectronSpectrum::pdf_ped,
                    &MultiElectronSpectrum::pdf_gradient_ped,
                    &MultiElectronSpectrum::pdf_gradient_hessian_ped,
                    { 1.123, 10.0, 0.100000, 0.2, 0.3, 0.2, 1.321, 0.35 },
                    { dp1, dp1, dp1, dp1, dp1, dp1, dp1, dp1}, -1.0, 9.0, 0.5);
}

TEST(TestFastGeneralPoissonMES, SetAndRecallParameters) {
  pdf_1d::GaussianPDF ped;
  pdf_1d::LimitedGaussianPDF ses(0,std::numeric_limits<double>::infinity());
  GeneralPoissonMES mes_base(0, 1, 1025, &ses, &ped);
  Eigen::VectorXd p_base(5);
  p_base << 1.0, 100.0, 20.0, 100.0, 35.0;
  mes_base.set_parameter_values(p_base);
  FastSingleValueGeneralPoissonMES mes(&mes_base);
  EXPECT_EQ(mes.num_parameters(), 1U);
  Eigen::VectorXd p(1);
  p << 1.0;
  mes.set_parameter_values(p);
  EXPECT_EQ(mes.parameter_values(), p);
}

TEST(TestFastGeneralPoissonMES, ValueEqualityWithBase_MES) {
  pdf_1d::GaussianPDF ped;
  pdf_1d::LimitedGaussianPDF ses(0,std::numeric_limits<double>::infinity());
  GeneralPoissonMES mes_base(0, 1, 1000, &ses, &ped);
  Eigen::VectorXd p_base(5);
  p_base << 1.0, 100.0, 20.0, 100.0, 35.0;
  mes_base.set_parameter_values(p_base);
  FastSingleValueGeneralPoissonMES mes(&mes_base);
  Eigen::VectorXd p(1);
  p << 1.0;
  mes.set_parameter_values(p);
  for(double x=0.5; x<1000.0; x+=1.0)
  {
    EXPECT_NEAR(mes_base.pdf_mes(x), mes.pdf_mes(x), mes_base.pdf_mes(x)*1e-8)
      << "x=" << x;
  }
}

TEST(TestFastGeneralPoissonMES, ValueEqualityWithBase_PED) {
  pdf_1d::GaussianPDF ped;
  pdf_1d::LimitedGaussianPDF ses(0,std::numeric_limits<double>::infinity());
  GeneralPoissonMES mes_base(0, 1, 1000, &ses, &ped);
  Eigen::VectorXd p_base(5);
  p_base << 1.0, 100.0, 20.0, 100.0, 35.0;
  mes_base.set_parameter_values(p_base);
  FastSingleValueGeneralPoissonMES mes(&mes_base);
  Eigen::VectorXd p(1);
  p << 1.0;
  mes.set_parameter_values(p);
  for(double x=0.5; x<1000.0; x+=1.0)
  {
    EXPECT_NEAR(mes_base.pdf_ped(x), mes.pdf_ped(x), mes_base.pdf_ped(x)*1e-8)
      << "x=" << x;
  }
}

TEST(TestFastGeneralPoissonMES, GradientEqualityWithBase_MES) {
  pdf_1d::GaussianPDF ped;
  pdf_1d::LimitedGaussianPDF ses(0,std::numeric_limits<double>::infinity());
  GeneralPoissonMES mes_base(0, 1, 1000, &ses, &ped);
  Eigen::VectorXd p_base(5);
  p_base << 1.0, 100.0, 20.0, 100.0, 35.0;
  mes_base.set_parameter_values(p_base);
  FastSingleValueGeneralPoissonMES mes(&mes_base);
  Eigen::VectorXd p(1);
  p << 1.0;
  mes.set_parameter_values(p);
  Eigen::VectorXd grad_base;
  Eigen::VectorXd grad;
  for(double x=0.5; x<1000.0; x+=1.0)
  {
    EXPECT_NEAR(mes_base.pdf_gradient_mes(x, grad_base),
      mes.pdf_gradient_mes(x, grad), mes_base.pdf_mes(x)*1e-8)
      << "x=" << x;
    EXPECT_NEAR(grad_base(0), grad(0), std::abs(grad(0))*1e-8)
      << "x=" << x;
  }
}

TEST(TestFastGeneralPoissonMES, GradientEqualityWithBase_PED) {
  pdf_1d::GaussianPDF ped;
  pdf_1d::LimitedGaussianPDF ses(0,std::numeric_limits<double>::infinity());
  GeneralPoissonMES mes_base(0, 1, 1000, &ses, &ped);
  Eigen::VectorXd p_base(5);
  p_base << 1.0, 100.0, 20.0, 100.0, 35.0;
  mes_base.set_parameter_values(p_base);
  FastSingleValueGeneralPoissonMES mes(&mes_base);
  Eigen::VectorXd p(1);
  p << 1.0;
  mes.set_parameter_values(p);
  Eigen::VectorXd grad_base;
  Eigen::VectorXd grad;
  for(double x=0.5; x<1000.0; x+=1.0)
  {
    EXPECT_NEAR(mes_base.pdf_gradient_ped(x, grad_base),
      mes.pdf_gradient_ped(x, grad), mes_base.pdf_ped(x)*1e-8)
      << "x=" << x;
    EXPECT_NEAR(grad_base(0), grad(0), std::abs(grad(0))*1e-8)
      << "x=" << x;
  }
}

TEST(TestFastGeneralPoissonMES, GradientCheck_MES)
{
  pdf_1d::GaussianPDF ped;
  pdf_1d::LimitedGaussianPDF ses(0,std::numeric_limits<double>::infinity());
  GeneralPoissonMES mes_base(0, 1, 1000, &ses, &ped);
  Eigen::VectorXd p_base(5);
  p_base << 1.0, 100.0, 20.0, 100.0, 35.0;
  mes_base.set_parameter_values(p_base);
  FastSingleValueGeneralPoissonMES mes(&mes_base);
  EXPECT_EQ(mes.num_parameters(), 1U);
  Eigen::VectorXd p(1);
  p << 1.0;
  mes.set_parameter_values(p);
  double dp1 = 1e-7;
  mes_gradient_test(&mes,
                    &MultiElectronSpectrum::pdf_mes,
                    &MultiElectronSpectrum::pdf_gradient_mes,
                    &MultiElectronSpectrum::pdf_gradient_hessian_mes,
                    { 1.123 },
                    { dp1 }, 0.5, 1000.0, 1.0, 0.75);
  // std::ofstream file("nes.dat");
  // file << mes.nes_pmf_matrix();
}

TEST(TestFastGeneralPoissonMES, Ovsersampled_GradientCheck_MES)
{
  pdf_1d::GaussianPDF ped;
  pdf_1d::LimitedGaussianPDF ses(0,std::numeric_limits<double>::infinity());

  auto cfg = GeneralPoissonMES::default_config();
  cfg.set_include_on_off_ped_shift(false);
  cfg.set_num_pe_convolutions(10);
  cfg.set_oversampling_factor(16);

  GeneralPoissonMES mes_base(0, 1, 1000, &ses, &ped, cfg);
  Eigen::VectorXd p_base(5);
  p_base << 1.0, 100.0, 20.0, 100.0, 35.0;
  mes_base.set_parameter_values(p_base);
  FastSingleValueGeneralPoissonMES mes(&mes_base);
  EXPECT_EQ(mes.num_parameters(), 1U);
  Eigen::VectorXd p(1);
  p << 1.0;
  mes.set_parameter_values(p);
  double dp1 = 1e-7;
  mes_gradient_test(&mes,
                    &MultiElectronSpectrum::pdf_mes,
                    &MultiElectronSpectrum::pdf_gradient_mes,
                    &MultiElectronSpectrum::pdf_gradient_hessian_mes,
                    { 1.123 },
                    { dp1 }, 0.5, 1000.0, 1.0, 0.75);
  // std::ofstream file("nes.dat");
  // file << mes.nes_pmf_matrix();
}

TEST(TestFastGeneralPoissonMES, RepeatabilityOfGradient)
{
  double val0;
  Eigen::VectorXd grad0;
  for(unsigned iloop=0;iloop<10;iloop++)
  {
    pdf_1d::GaussianPDF ped;
    pdf_1d::LimitedGaussianPDF ses(0,std::numeric_limits<double>::infinity());
    GeneralPoissonMES mes_base(0, 1, 1000, &ses, &ped);
    Eigen::VectorXd p_base(5);
    p_base << 1.0, 100.0, 20.0, 100.0, 35.0;
    mes_base.set_parameter_values(p_base);
    FastSingleValueGeneralPoissonMES mes(&mes_base);
    EXPECT_EQ(mes.num_parameters(), 1U);
    Eigen::VectorXd p(1);
    p << 1.0;
    mes.set_parameter_values(p);
    Eigen::VectorXd grad;
    double val = mes.pdf_gradient_mes(562.5, grad);
    if(iloop==0) {
      val0 = val;
      grad0 = grad;
    } else {
      EXPECT_EQ(val, val0);
      EXPECT_EQ(grad, grad0);
    }
  }
}

TEST(TestFastGeneralPoissonMES, GradientCheck_PED)
{
  pdf_1d::GaussianPDF ped;
  pdf_1d::LimitedGaussianPDF ses(0,std::numeric_limits<double>::infinity());
  GeneralPoissonMES mes_base(0, 1, 1000, &ses, &ped);
  Eigen::VectorXd p_base(5);
  p_base << 1.0, 100.0, 20.0, 100.0, 35.0;
  mes_base.set_parameter_values(p_base);
  FastSingleValueGeneralPoissonMES mes(&mes_base);
  EXPECT_EQ(mes.num_parameters(), 1U);
  Eigen::VectorXd p(1);
  p << 1.0;
  mes.set_parameter_values(p);
  double dp1 = 1e-7;
  mes_gradient_test(&mes,
                    &MultiElectronSpectrum::pdf_ped,
                    &MultiElectronSpectrum::pdf_gradient_ped,
                    &MultiElectronSpectrum::pdf_gradient_hessian_ped,
                    { 1.123 },
                    { dp1 }, 0.5, 1000.0, 1.0, 0.75);
}

TEST(TestFastGeneralPoissonMES, RepeatabilityOfLikelihood)
{
  double inf = std::numeric_limits<double>::infinity();
  std::vector<double> all_val;
  std::vector<Eigen::VectorXd> all_grad;
  for(unsigned iloop=0;iloop<10;iloop++)
  {
    auto mes_data = karkar_data();
    SimpleHist mes_hist(1.0);
    for(auto idata : mes_data)mes_hist.insert(idata);
    pdf_1d::GaussianPDF ped;
    pdf_1d::LimitedExponentialPDF exp_pdf(0,inf,mes_hist.dxval());
    exp_pdf.limit_scale(0.1, inf);
    pdf_1d::LimitedGaussianPDF gauss_pdf(0,inf);
    pdf_1d::TwoComponent1DPDF ses(&exp_pdf, "exp", &gauss_pdf, "gauss");
    GeneralPoissonMES mes_base(mes_hist.xval_left(0),
                                mes_hist.dxval(),
                                mes_hist.size(), &ses, &ped);
    Eigen::VectorXd p_base(7);
    p_base << 0.56, 3094.7, 19.6, 0.1, 5.0, 88.9, 29.3;
    mes_base.set_parameter_values(p_base);
    FastSingleValueGeneralPoissonMES mes(&mes_base);
    SPELikelihood like(mes, mes_hist);
    Eigen::VectorXd p(1);
    p << 0.56;
    Eigen::VectorXd grad(1);
    double val = like.value_and_gradient(p,grad);

    all_val.push_back(val);
    all_grad.push_back(grad);
  }

  EXPECT_EQ(std::count(all_val.begin(), all_val.end(), all_val.front()),
            (int)all_val.size());

  EXPECT_EQ(std::count(all_grad.begin(), all_grad.end(), all_grad.front()),
            (int)all_grad.size());
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
