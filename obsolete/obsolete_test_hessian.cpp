/*

   calin/unit_tests/math/test_hessian.cpp -- Stephen Fegan -- 2015-04-24

   Unit tests for hessian classes

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

#include <iostream>
#include <iomanip>
#include <gtest/gtest.h>
#include <vector>

#include "Eigen/Dense"
#include "calib/spe_fit.hpp"
#include "../calib/karkar_data.hpp"
#include "math/nlopt_optimizer.hpp"
#include "math/hessian.hpp"

using namespace calin::math;
using namespace calin::math::histogram;
using namespace calin::calib;
using namespace calin::calib::spe_fit;
using namespace calin::unit_tests;

TEST(TestHessian, Minimize_NLOpt_LD_LBFGS) {
  auto mes_data = karkar_data();
  SimpleHist mes_hist(1.0);
  for(auto idata : mes_data)mes_hist.insert(idata);
  PoissonGaussianMES mes_model(20);
  SPELikelihood like(mes_model, mes_hist);

  //optimizer::NLOptOptimizer opt("LN_SBPLX", &like);
  optimizer::NLOptOptimizer opt("LD_LBFGS", &like);
  opt.set_scale({0.1,0.1,1.0,1.0,0.05});
  opt.set_verbosity_level(optimizer::OptimizerVerbosityLevel::MAX);
  opt.set_abs_tolerance(0.0001);
  opt.set_initial_values(std::vector<double>{ 1.0, 3100.0, 20.0, 100.0, 0.45 });
  Eigen::VectorXd x(5);
  double f_val;
  opt.minimize(x, f_val);

  //EXPECT_EQ(status, GSL_SUCCESS);
  EXPECT_NEAR(x[0], 0.55349337, 0.0001);
  EXPECT_NEAR(x[1], 3094.2715, 0.01);
  EXPECT_NEAR(x[2], 19.6141970, 0.001);
  EXPECT_NEAR(x[3], 89.1810077, 0.01);
  EXPECT_NEAR(x[4], 0.32388838, 0.0001);

  std::cout << std::fixed << std::setprecision(3);
  std::cout << x[0] << ' ' << x[1] << ' ' << x[2] << ' '
            << x[3] << ' ' << x[4] << '\n';

  Eigen::MatrixXd hessian_mat(5,5);
  Eigen::VectorXd gradient(5);
  like.value_gradient_and_hessian(x, gradient, hessian_mat);
  std::cout << std::scientific << std::setprecision(8) << hessian_mat << "\n\n";

  Eigen::MatrixXd hessian_mat_num(5,5);
  hessian::calculate_hessian_1st_order_eps(like, x, hessian_mat_num, 100.0);
  std::cout << std::scientific << std::setprecision(8) << hessian_mat_num << "\n\n";

  Eigen::MatrixXd err_mat = hessian_mat.inverse();
  std::cout << std::scientific << std::setprecision(8) << err_mat << "\n\n";

  Eigen::MatrixXd err_mat_num(5,5);
  Eigen::MatrixXd eigenvectors(5,5);
  Eigen::VectorXd eigenvalues(5);

  hessian::hessian_to_error_matrix(like, hessian_mat_num, err_mat_num,
                                   eigenvalues, eigenvectors);
  std::cout << std::scientific << std::setprecision(8) << err_mat_num << "\n\n";
  std::cout << eigenvalues.transpose() << "\n\n";

  Eigen::MatrixXd err_mat_est(5,5);
  opt.error_matrix_estimate(err_mat_est);
  std::cout << std::scientific << std::setprecision(8) << err_mat_est << "\n\n";
  std::cout << "Eigen: " << err_mat_est.diagonal().array().sqrt().transpose() << "\n\n";
  Eigen::VectorXd dx =
    hessian::step_size_err_up(like, x, err_mat_est.diagonal().array().sqrt());
  std::cout << "DX: " << dx.transpose() << "\n\n";
  Eigen::MatrixXd hessian_mat_num2(5,5);
  hessian::calculate_hessian_1st_order_dx(like, x, dx, hessian_mat_num2);

  Eigen::MatrixXd err_mat_num2(5,5);
  Eigen::MatrixXd eigenvectors2(5,5);
  Eigen::VectorXd eigenvalues2(5);

  hessian::hessian_to_error_matrix(like, hessian_mat_num2, err_mat_num2,
                                   eigenvalues2, eigenvectors2);
  std::cout << std::scientific << std::setprecision(8) << err_mat_num2
            << "\n\n";
  std::cout << eigenvalues2.transpose() << "\n\n";

  Eigen::MatrixXd hessian_mat_2nd(5,5);
  hessian::calculate_hessian_2nd_order_dx(like, x, dx, hessian_mat_2nd);

  Eigen::MatrixXd err_mat_2nd(5,5);
  Eigen::MatrixXd eigenvectors3(5,5);
  Eigen::VectorXd eigenvalues3(5);

  hessian::hessian_to_error_matrix(like, hessian_mat_2nd, err_mat_2nd,
                                   eigenvalues3, eigenvectors3);
  std::cout << std::scientific << std::setprecision(8) << err_mat_2nd
            << "\n\n";
  std::cout << eigenvalues3.transpose() << "\n\n";



  std::cout << std::sqrt(err_mat(0,0)) << ' '
            << std::sqrt(err_mat(1,1)) << ' '
            << std::sqrt(err_mat(2,2)) << ' '
            << std::sqrt(err_mat(3,3)) << ' '
            << std::sqrt(err_mat(4,4)) << '\n';

  std::cout << std::sqrt(err_mat_num(0,0)) << ' '
            << std::sqrt(err_mat_num(1,1)) << ' '
            << std::sqrt(err_mat_num(2,2)) << ' '
            << std::sqrt(err_mat_num(3,3)) << ' '
            << std::sqrt(err_mat_num(4,4)) << '\n';

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
