/*

   calin/math/nspace.cpp -- Stephen Fegan -- 2020-11-27

   (Re)Implemantion of the U Utah and UCLA N-Space algorithm

   Copyright 2020, Stephen Fegan <sfegan@llr.in2p3.fr>
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

#include <exception>
#include <algorithm>

#include <util/log.hpp>
#include <math/nspace.hpp>
#include <calin_global_definitions.hpp>

using namespace calin::math::nspace;
using namespace calin::util::log;

namespace {

Eigen::VectorXd xlo_from_axes(const std::vector<calin::math::nspace::Axis>& axes)
{
  Eigen::VectorXd xlo(axes.size());
  std::transform(axes.begin(), axes.end(), xlo.data(), [](Axis a){return a.xlo;});
  return xlo;
}

Eigen::VectorXd xhi_from_axes(const std::vector<calin::math::nspace::Axis>& axes)
{
  Eigen::VectorXd xhi(axes.size());
  std::transform(axes.begin(), axes.end(), xhi.data(), [](Axis a){return a.xhi;});
  return xhi;
}

Eigen::VectorXi n_from_axes(const std::vector<calin::math::nspace::Axis>& axes)
{
  Eigen::VectorXi n(axes.size());
  std::transform(axes.begin(), axes.end(), n.data(), [](Axis a){return a.n;});
  return n;
}

} // anonymous namespace

TreeSparseNSpace::
TreeSparseNSpace(const Eigen::VectorXd& xlo, const Eigen::VectorXd& xhi,
    const Eigen::VectorXi& n):
  xlo_(xlo.size()), xhi_(xlo.size()), dx_(xlo.size()), dx_inv_(xlo.size()),
  n_(xlo.size())
{
  if(std::min({xlo.size(), xhi.size(), n.size()}) !=
      std::max({xlo.size(), xhi.size(), n.size()})) {
    throw std::runtime_error("TreeSparseNSpace: xhi, xlo and n must all have same size");
  }

  N_ = 1;
  for(unsigned i=0; i<xlo.size(); ++i) {
    const double dx = (xhi[i] - xlo[i])/n[i];
    xlo_[i] = xlo[i];
    xhi_[i] = xhi[i];
    n_[i] = n[i];
    N_ *= n[i];
    dx_[i] = dx;
    dx_inv_[i] = 1.0/dx;
  }
}

TreeSparseNSpace::TreeSparseNSpace(const std::vector<Axis>& axes):
  TreeSparseNSpace(xlo_from_axes(axes), xhi_from_axes(axes), n_from_axes(axes))
{
  // nothing to see here
}

TreeSparseNSpace::TreeSparseNSpace(const calin::ix::math::nspace::NSpaceData& proto):
  TreeSparseNSpace(
    Eigen::Map<const Eigen::VectorXd>(proto.axes_lower_bounds().data(), proto.axes_lower_bounds().size()),
    Eigen::Map<const Eigen::VectorXd>(proto.axes_upper_bounds().data(), proto.axes_upper_bounds().size()),
    Eigen::Map<const Eigen::VectorXi>(proto.axes_num_bins().data(), proto.axes_num_bins().size()))
{
  this->accumulate_from_proto(proto);
}

void TreeSparseNSpace::injest(const TreeSparseNSpace& o)
{
  if(xlo_!=o.xlo_ or xhi_!=o.xhi_ or n_!=o.n_) {
    throw std::runtime_error("TreeSparseNSpace: cannot injest space with incompatibe axis defifinition");
  }
  for(auto i : o.bins_) {
    bins_[i.first] += i.second;
  }
}

void TreeSparseNSpace::accumulate_many(const Eigen::MatrixXd& x, double w)
{
  Eigen::VectorXd xx(xlo_.size());
  if(x.cols() != xlo_.size()){
    throw std::runtime_error("BlockSparseNSpace: dimensional mismatch in number of matrix columns");
  }
  for(unsigned irow=0; irow<x.rows(); ++irow) {
    xx = x.row(irow);
    bins_[index(xx)] += w;
  }
}

void TreeSparseNSpace::accumulate_many(const Eigen::MatrixXd& x, const Eigen::VectorXd& w)
{
  Eigen::VectorXd xx(xlo_.size());
  if(x.cols() != xlo_.size()) {
    throw std::runtime_error("BlockSparseNSpace: dimensional mismatch in number of matrix columns");
  }
  if(x.rows() != w.rows()) {
    throw std::runtime_error("BlockSparseNSpace: dimensional mismatch in number of matrix & vector rows");
  }
  for(unsigned irow=0; irow<x.rows(); ++irow) {
    xx = x.row(irow);
    bins_[index(xx)] += w[irow];
  }
}

std::vector<calin::math::nspace::Axis> TreeSparseNSpace::axes() const
{
  std::vector<calin::math::nspace::Axis> a;
  for(unsigned i=0; i<xlo_.size(); i++) {
    a.push_back({xlo_[i], xhi_[i], static_cast<unsigned int>(n_[i])});
  }
  return a;
}

calin::math::nspace::Axis TreeSparseNSpace::axis(unsigned iaxis) const
{
  if(iaxis >= n_.size()) {
    throw std::runtime_error("TreeSparseNSpace: iaxis out of range");
  }
  calin::math::nspace::Axis a { xlo_[iaxis], xhi_[iaxis], static_cast<unsigned int>(n_[iaxis]) };
  return a;
}

Eigen::VectorXd TreeSparseNSpace::axis_bin_centers(unsigned iaxis) const
{
  if(iaxis >= n_.size()) {
    throw std::runtime_error("TreeSparseNSpace: iaxis out of range");
  }
  Eigen::VectorXd x(n_[iaxis]);
  for(int i=0; i<n_[iaxis]; i++) {
    x[i] = xlo_[iaxis] + dx_[iaxis] * (0.5 + i);
  }
  return x;
}

Eigen::VectorXd TreeSparseNSpace::axis_bin_edges(unsigned iaxis) const
{
  if(iaxis >= n_.size()) {
    throw std::runtime_error("TreeSparseNSpace: iaxis out of range");
  }
  Eigen::VectorXd x(n_[iaxis]+1);
  for(int i=0; i<=n_[iaxis]; i++) {
    x[i] = xlo_[iaxis] + dx_[iaxis] * i;
  }
  return x;
}

calin::math::nspace::TreeSparseNSpace* TreeSparseNSpace::
project_along_axis(unsigned iaxis, unsigned axis_cell_lo, unsigned axis_cell_hi) const
{
  if(iaxis >= n_.size()) {
    throw std::runtime_error("TreeSparseNSpace: iaxis out of range");
  }

  std::vector<calin::math::nspace::Axis> a;
  for(unsigned i=0; i<xlo_.size(); i++) {
    if(i != iaxis) {
      a.push_back({xlo_[i], xhi_[i], static_cast<unsigned int>(n_[i])});
    }
  }

  TreeSparseNSpace* newspace = new TreeSparseNSpace(a);

  int64_t div_lo = 1;
  for(unsigned jaxis=iaxis+1; jaxis<n_.size(); jaxis++) {
    div_lo *= n_[jaxis];
  }

  for(auto i : bins_) {
    if(i.first < 0) {
      newspace->bins_[i.first] += i.second;
    } else {
      auto qr_lo = std::div(i.first, div_lo);
      auto qr_hi = std::div(qr_lo.quot, int64_t(n_[iaxis]));
      if(qr_hi.rem >= axis_cell_lo and qr_hi.rem <= axis_cell_hi) {
        int64_t inew = qr_lo.rem + qr_hi.quot * div_lo;
        newspace->bins_[inew] += i.second;
      } else {
        newspace->bins_[-1] += i.second;
      }
    }
  }
  return newspace;
}

calin::math::nspace::TreeSparseNSpace* TreeSparseNSpace::
project_along_axis(unsigned iaxis) const
{
  return this->project_along_axis(iaxis, 0, n_[std::min(iaxis,unsigned(n_.size()-1))]);
}

calin::math::nspace::TreeSparseNSpace* TreeSparseNSpace::
sum_x_to_the_n_along_axis(unsigned iaxis, unsigned n, unsigned axis_cell_lo, unsigned axis_cell_hi) const
{
  if(iaxis >= n_.size()) {
    throw std::runtime_error("TreeSparseNSpace: iaxis out of range");
  }

  std::vector<calin::math::nspace::Axis> a;
  for(unsigned i=0; i<xlo_.size(); i++) {
    if(i != iaxis) {
      a.push_back({xlo_[i], xhi_[i], static_cast<unsigned int>(n_[i])});
    }
  }

  TreeSparseNSpace* newspace = new TreeSparseNSpace(a);

  int64_t div_lo = 1;
  for(unsigned jaxis=iaxis+1; jaxis<n_.size(); jaxis++) {
    div_lo *= n_[jaxis];
  }

  for(auto i : bins_) {
    if(i.first < 0) {
      newspace->bins_[i.first] += i.second;
    } else {
      auto qr_lo = std::div(i.first, div_lo);
      auto qr_hi = std::div(qr_lo.quot, int64_t(n_[iaxis]));
      double x = xlo_[iaxis] + dx_[iaxis] * (qr_hi.rem + 0.5);
      double xn = 1.0;
      switch(n) {
      case 0:
        break;
      case 1:
        xn = x;
        break;
      case 2:
        xn = x*x;
        break;
      case 3:
        xn = x*x*x;
        break;
      default:
        for(unsigned i=0;i<n;i++)xn *= x;
        break;
      }
      if(qr_hi.rem >= axis_cell_lo and qr_hi.rem <= axis_cell_hi) {
        int64_t inew = qr_lo.rem + qr_hi.quot * div_lo;
        newspace->bins_[inew] += i.second * xn;
      } else {
        newspace->bins_[-1] += i.second * xn;
      }
    }
  }
  return newspace;
}

calin::math::nspace::TreeSparseNSpace* TreeSparseNSpace::
sum_x_to_the_n_along_axis(unsigned iaxis, unsigned n) const
{
  return this->sum_x_to_the_n_along_axis(iaxis, n, 0, n_[std::min(iaxis,unsigned(n_.size()-1))]);
}

Eigen::VectorXd TreeSparseNSpace::as_vector() const
{
  if(n_.size() != 1) {
    throw std::runtime_error("TreeSparseNSpace: only single-axis spaces can be converted to vectors");
  }
  Eigen::VectorXd v(n_[0]);
  v.setZero();
  for(auto i : bins_) {
    if(i.first >= 0) {
      v(i.first) = i.second;
    }
  }
  return v;
}

Eigen::MatrixXd TreeSparseNSpace::as_matrix() const
{
  if(n_.size() != 2) {
    throw std::runtime_error("TreeSparseNSpace: only two-axis spaces can be converted to matrices");
  }
  Eigen::MatrixXd m(n_[0], n_[1]);
  m.setZero();
  for(auto i : bins_) {
    if(i.first >= 0) {
      auto qr = std::div(i.first, int64_t(n_[1]));
      m(qr.quot, qr.rem) = i.second;
    }
  }
  return m;
}

double TreeSparseNSpace::total_weight() const {
  double w0 = 0;
  for(auto i : bins_) {
    if(i.first >= 0) {
      w0 += i.second;
    }
  }
  return w0;
}

Eigen::VectorXd TreeSparseNSpace::mean_and_total_weight(double& w0) const {
  Eigen::VectorXd w1(xlo_.size());
  Eigen::VectorXd x(xlo_.size());
  w0 = 0;
  w1.setZero();
  for(auto i : bins_) {
    if(i.first >= 0) {
      x_center(x, i.first);
      w0 += i.second;
      w1 += x * i.second;
    }
  }
  return w1/w0;
}

Eigen::VectorXd TreeSparseNSpace::mean() const {
  double w0;
  return mean_and_total_weight(w0);
}

Eigen::MatrixXd TreeSparseNSpace::covar_mean_and_total_weight(Eigen::VectorXd& w1, double& w0) const {
  Eigen::MatrixXd w2(xlo_.size(), xlo_.size());
  w1.resize(xlo_.size());
  Eigen::VectorXd x(xlo_.size());
  w0 = 0;
  w1.setZero();
  w2.setZero();
  for(auto i : bins_) {
    if(i.first >= 0) {
      x_center(x, i.first);
      w0 += i.second;
      w1 += x * i.second;
      w2.noalias() += x * x.transpose() * i.second; // outer product
    }
  }
  w1 /= w0;
  return w2/w0 - w1*w1.transpose(); // outer product
}

Eigen::MatrixXd TreeSparseNSpace::covar() const {
  double w0;
  Eigen::VectorXd w1(xlo_.size());
  return covar_mean_and_total_weight(w1, w0);
}

void TreeSparseNSpace::save_to_proto(ix::math::nspace::NSpaceData* proto) const
{
  proto->Clear();
  proto->set_num_axes(xlo_.size());
  for(unsigned iaxis=0;iaxis<xlo_.size();++iaxis) {
    proto->add_axes_lower_bounds(xlo_[iaxis]);
    proto->add_axes_upper_bounds(xhi_[iaxis]);
    proto->add_axes_num_bins(n_[iaxis]);
  }
  proto->mutable_bin_indices()->Reserve(bins_.size());
  proto->mutable_bin_weights()->Reserve(bins_.size());
  for(const auto& bin : bins_) {
    if(bin.first == -1) {
      proto->set_overflow_weight(bin.first);
    } else if (bin.second != 0.0) {
      proto->add_bin_indices(bin.first);
      proto->add_bin_weights(bin.second);
    }
  }
}

calin::ix::math::nspace::NSpaceData* TreeSparseNSpace::as_proto() const
{
  auto* proto = new calin::ix::math::nspace::NSpaceData;
  save_to_proto(proto); 
  return proto;
}

void TreeSparseNSpace::accumulate_from_proto(const calin::ix::math::nspace::NSpaceData& proto)
{
  if(not std::equal(xlo_.begin(), xlo_.end(), proto.axes_lower_bounds().begin(), proto.axes_lower_bounds().end())) {
    throw std::runtime_error("TreeSparseNSpace: lower bounds do not match");
  }
  if(not std::equal(xhi_.begin(), xhi_.end(), proto.axes_upper_bounds().begin(), proto.axes_upper_bounds().end())) {
    throw std::runtime_error("TreeSparseNSpace: upper bounds do not match");
  }
  if(not std::equal(n_.begin(), n_.end(), proto.axes_num_bins().begin(), proto.axes_num_bins().end())) {
    throw std::runtime_error("TreeSparseNSpace: number of bins do not match");
  }
  if(proto.bin_indices_size() != proto.bin_weights_size()) {
    throw std::runtime_error("TreeSparseNSpace: bin indices and weights do not match");
  }
  for(int ibin=0; ibin<proto.bin_indices_size(); ++ibin) {
    bins_[proto.bin_indices(ibin)] +=  proto.bin_weights(ibin);
  }
  if(proto.overflow_weight() != 0.0) {
    bins_[-1] += proto.overflow_weight();
  }
}

TreeSparseNSpace* TreeSparseNSpace::create_from_proto(const calin::ix::math::nspace::NSpaceData& proto)
{
  return new TreeSparseNSpace(proto);
}

// =============================================================================
// =============================================================================

// BlockSparseNSpace

// =============================================================================
// =============================================================================

BlockSparseNSpace::
BlockSparseNSpace(const Eigen::VectorXd& xlo, const Eigen::VectorXd& xhi,
    const Eigen::VectorXi& n, unsigned log2_block_size):
  xlo_(xlo.size()), xhi_(xlo.size()), dx_(xlo.size()), dx_inv_(xlo.size()),
  n_(xlo.size()), narray_(xlo.size())

{
  if(std::min({xlo.size(), xhi.size(), n.size()}) !=
      std::max({xlo.size(), xhi.size(), n.size()})) {
    throw std::runtime_error("BlockSparseNSpace: xhi, xlo and n must all have same size");
  }

  N_ = 1;
  for(unsigned i=0; i<xlo.size(); ++i) {
    const double dx = (xhi[i] - xlo[i])/n[i];
    xlo_[i] = xlo[i];
    xhi_[i] = xhi[i];
    n_[i] = n[i];
    N_ *= n[i];
    dx_[i] = dx;
    dx_inv_[i] = 1.0/dx;
  }

  block_shift_ = validated_log2_block_size(log2_block_size, xlo.size());
  block_mask_ = (1<<block_shift_) - 1;
  block_size_ = 1<<(block_shift_ * xlo.size());

  alloc_size_ = std::max(block_size_, 1048576U);

  Narray_ = 1;
  for(unsigned i=0; i<xlo.size(); ++i) {
    narray_[i] = (n[i]+block_mask_)>>block_shift_;
    Narray_ *= narray_[i];
  }

  array_.resize(Narray_, nullptr);
}

BlockSparseNSpace::BlockSparseNSpace(const std::vector<Axis>& axes, unsigned log2_block_size):
  BlockSparseNSpace(xlo_from_axes(axes), xhi_from_axes(axes), n_from_axes(axes), log2_block_size)
{
  // nothing to see here
}

BlockSparseNSpace::BlockSparseNSpace(const calin::ix::math::nspace::NSpaceData& proto, unsigned log2_block_size ):
  BlockSparseNSpace(
    Eigen::Map<const Eigen::VectorXd>(proto.axes_lower_bounds().data(), proto.axes_lower_bounds().size()),
    Eigen::Map<const Eigen::VectorXd>(proto.axes_upper_bounds().data(), proto.axes_upper_bounds().size()),
    Eigen::Map<const Eigen::VectorXi>(proto.axes_num_bins().data(), proto.axes_num_bins().size()),
    log2_block_size)
{
  this->accumulate_from_proto(proto);
}

BlockSparseNSpace::~BlockSparseNSpace()
{
  for(auto* array : alloc_all_list_) {
    delete[] array;
  }
}

unsigned BlockSparseNSpace::
validated_log2_block_size(unsigned log2_block_size, unsigned naxis)
{
  if(log2_block_size==0) {
    unsigned block_size = 1;
    for(log2_block_size=1; block_size<1024; log2_block_size++) {
      block_size <<= naxis;
    }
  }
  return std::max(1U, log2_block_size);
}

bool BlockSparseNSpace::index(
  const Eigen::VectorXd& x, int64_t& array_index, int64_t& block_index) const
{
  if(x.size() != xlo_.size()) {
    throw std::runtime_error("BlockSparseNSpace: dimensional mismatch");
  }

  array_index = 0;
  block_index = 0;
  for(int i=0; i<xlo_.size(); i++) {
    int ii = std::floor((x(i)-xlo_(i))*dx_inv_(i));
    if(ii<0 or ii>=n_(i)) {
      array_index = block_index = -1;
      return false;
    }

    block_index = (block_index<<block_shift_) | (ii&block_mask_);
    array_index = array_index*narray_(i) + (ii>>block_shift_);
  }
  return true;
}

bool BlockSparseNSpace::x_center(Eigen::VectorXd& x, int64_t array_index, int64_t block_index) const
{
  if(block_index<0 or block_index>=block_size_ or array_index<0 or array_index>=int64_t(array_.size())) {
    return false;
  }

  if(x.size() != xlo_.size()) {
    x.resize(xlo_.size());
  }

  for(unsigned i=xlo_.size(); i>0;) {
    --i;
    auto qr = std::div(array_index, int64_t(narray_[i]));
    unsigned ix = (qr.rem<<block_shift_) | (block_index&block_mask_);
    if(ix >= unsigned(n_[i])) {
      return false;
    }
    x[i] = xlo_[i] + dx_[i] * (ix + 0.5);
    array_index = qr.quot;
    block_index >>= block_shift_;
  }

  return true;
}

bool BlockSparseNSpace::index_of_bin(const Eigen::VectorXi& ix, int64_t& array_index, int64_t& block_index) const
{
  if(ix.size() != xlo_.size()) {
    throw std::runtime_error("BlockSparseNSpace: dimensional mismatch");
  }

  array_index = 0;
  block_index = 0;
  for(int i=0; i<xlo_.size(); i++) {
    int ii = ix(i);
    if(ii<0 or ii>=n_(i)) {
      array_index = block_index = -1;
      return false;
    }

    block_index = (block_index<<block_shift_) | (ii&block_mask_);
    array_index = array_index*narray_(i) + (ii>>block_shift_);
  }
  return true;
}

bool BlockSparseNSpace::bin_coords(Eigen::VectorXi& ix, int64_t array_index, int64_t block_index) const
{
  if(block_index<0 or block_index>=block_size_ or array_index<0 or array_index>=int64_t(array_.size())) {
    return false;
  }

  if(ix.size() != xlo_.size()) {
    ix.resize(xlo_.size());
  }

  for(unsigned i=xlo_.size(); i>0;) {
    --i;
    auto qr = std::div(array_index, int64_t(narray_[i]));
    int iix = (qr.rem<<block_shift_) | (block_index&block_mask_);
    if(iix >= n_[i]) {
      return false;
    }
    ix[i] = iix;
    array_index = qr.quot;
    block_index >>= block_shift_;
  }

  return true;
}

double* BlockSparseNSpace::block_ptr(int64_t array_index)
{
  double* block = array_[array_index];
  if(block == nullptr) {
    if(alloc_next_ == nullptr) {
      if(alloc_free_list_.empty()) {
        alloc_next_ = new double[alloc_size_];
        alloc_all_list_.push_front(alloc_next_);
      } else {
        alloc_next_ = alloc_free_list_.front();
        alloc_free_list_.pop_front();
      }
      alloc_end_ = alloc_next_ + alloc_size_;
    }

    block = array_[array_index] = alloc_next_;
    alloc_next_ += block_size_;
    std::fill(block, block+block_size_, 0);

    if(alloc_next_ == alloc_end_) {
      alloc_next_ = alloc_end_ = nullptr;
    }
  }
  return block;
}

double& BlockSparseNSpace::cell_ref(int64_t array_index, int64_t block_index)
{
  return block_ptr(array_index)[block_index];
}

double BlockSparseNSpace::cell_val(int64_t array_index, int64_t block_index) const
{
  const double* block = array_[array_index];
  if(block == nullptr) { return 0; }
  return block[block_index];
}

void BlockSparseNSpace::accumulate(const Eigen::VectorXd& x, double w)
{
  int64_t array_index;
  int64_t block_index;
  if(index(x, array_index, block_index)) {
    // LOG(INFO) << x << ' ' << array_index << ' ' << block_index;
    cell_ref(array_index,block_index) += w;
  } else {
    overflow_ += w;
  }
}

void BlockSparseNSpace::accumulate_many(const Eigen::MatrixXd& x, double w)
{
  Eigen::VectorXd xx(xlo_.size());
  int64_t array_index;
  int64_t block_index;
  if(x.cols() != xlo_.size()){
    throw std::runtime_error("BlockSparseNSpace: dimensional mismatch in number of matrix columns");
  }
  for(unsigned irow=0; irow<x.rows(); ++irow) {
    xx = x.row(irow);
    if(index(xx, array_index, block_index)) {
      // LOG(INFO) << x << ' ' << array_index << ' ' << block_index;
      cell_ref(array_index,block_index) += w;
    } else {
      overflow_ += w;
    }
  }
}

void BlockSparseNSpace::accumulate_many(const Eigen::MatrixXd& x, const Eigen::VectorXd& w)
{
  Eigen::VectorXd xx(xlo_.size());
  int64_t array_index;
  int64_t block_index;
  if(x.cols() != xlo_.size()) {
    throw std::runtime_error("BlockSparseNSpace: dimensional mismatch in number of matrix columns");
  }
  if(x.rows() != w.rows()) {
    throw std::runtime_error("BlockSparseNSpace: dimensional mismatch in number of matrix & vector rows");
  }
  for(unsigned irow=0; irow<x.rows(); ++irow) {
    xx = x.row(irow);
    if(index(xx, array_index, block_index)) {
      // LOG(INFO) << x << ' ' << array_index << ' ' << block_index;
      cell_ref(array_index,block_index) += w[irow];
    } else {
      overflow_ += w[irow];
    }
  }
}

void BlockSparseNSpace::clear()
{
  overflow_ = 0;
  std::fill(array_.begin(), array_.end(), nullptr);
  alloc_next_ = alloc_end_ = nullptr;
  alloc_free_list_ = alloc_all_list_;
}

void BlockSparseNSpace::prune_below_threshold(double threshold)
{
  for(auto* block : array_) {
    if(block) {
      for(unsigned i=0;i<block_size_; ++i) {
        if(block[i] <= threshold) {
          overflow_ += block[i];
          block[i] = 0;
        }
      }
    }
  }
}

void BlockSparseNSpace::injest(const BlockSparseNSpace& o)
{
  if(xlo_!=o.xlo_ or xhi_!=o.xhi_ or n_!=o.n_) {
    throw std::runtime_error("BlockSparseNSpace: cannot injest space with incompatible axis definition");
  }
  if(o.block_shift_ == block_shift_) {
    for(unsigned array_index=0; array_index<array_.size(); ++array_index) {
      const double* oblock = o.array_[array_index];
      if(oblock) {
        double* block = block_ptr(array_index);
        for(unsigned block_index=0; block_index<block_size_; ++block_index) {
          block[block_index] += oblock[block_index];
        }
      }
    }
  } else {
    throw std::runtime_error("BlockSparseNSpace: different block sizes not (yet) supported");
  }
  overflow_ += o.overflow_;
}

void BlockSparseNSpace::injest_from_subspace(const Eigen::VectorXd& x_super, const BlockSparseNSpace& o)
{
  if(xlo_.size() != x_super.size() + o.xlo_.size()) {
    throw std::runtime_error("BlockSparseNSpace: incotrrect number of axes in coordinates and subspace");
  }
  if(xlo_.tail(xlo_.size()-x_super.size())!=o.xlo_ or
      xhi_.tail(xhi_.size()-x_super.size())!=o.xhi_ or
      n_.tail(n_.size()-x_super.size())!=o.n_) {
    throw std::runtime_error("BlockSparseNSpace: cannot injest sub-space with incompatible axis definition");
  }

  Eigen::VectorXi xi_super(n_.size());
  for(int i=0; i<x_super.size(); i++) {
    int ii = std::floor((x_super(i)-xlo_(i))*dx_inv_(i));
    if(ii<0 or ii>=n_(i)) {
      overflow_ += o.total_weight();
      return;
    }
    xi_super(i) = ii;
  }

  Eigen::VectorXi xi_sub(o.n_.size());
  for(unsigned oarray_index=0; oarray_index<o.array_.size(); ++oarray_index) {
    const double* oblock = o.array_[oarray_index];
    if(oblock) {
      for(unsigned oblock_index=0; oblock_index<o.block_size_; ++oblock_index) {
        o.bin_coords(xi_sub, oarray_index, oblock_index);
        xi_super.tail(xi_sub.size()) = xi_sub;
        int64_t array_index;
        int64_t block_index;
        if(not index_of_bin(xi_super, array_index, block_index)) {
          throw std::logic_error("Index does not appear to be in superspace");
        }
        double* block = block_ptr(array_index);
        block[block_index] += oblock[oblock_index];
      }
    }
  }

  overflow_ += o.overflow_;
}

std::vector<Axis> BlockSparseNSpace::axes() const
{
  std::vector<calin::math::nspace::Axis> a;
  for(unsigned i=0; i<xlo_.size(); i++) {
    a.push_back({xlo_[i], xhi_[i], static_cast<unsigned int>(n_[i])});
  }
  return a;
}

Axis BlockSparseNSpace::axis(unsigned iaxis) const
{
  if(iaxis >= n_.size()) {
    throw std::runtime_error("BlockSparseNSpace: iaxis out of range");
  }
  calin::math::nspace::Axis a { xlo_[iaxis], xhi_[iaxis], static_cast<unsigned int>(n_[iaxis]) };
  return a;
}

std::vector<Axis> BlockSparseNSpace::make_superspace_axes(
  const Eigen::VectorXd& xlo, const Eigen::VectorXd& xhi, const Eigen::VectorXi& n) const
{
  if(std::min({xlo.size(), xhi.size(), n.size()}) !=
      std::max({xlo.size(), xhi.size(), n.size()})) {
    throw std::runtime_error("BlockSparseNSpace: xhi, xlo and n must all have same size");
  }
  std::vector<calin::math::nspace::Axis> a;
  for(unsigned i=0; i<xlo.size(); i++) {
    a.push_back({xlo[i], xhi[i], static_cast<unsigned int>(n[i])});
  }
  for(unsigned i=0; i<xlo_.size(); i++) {
    a.push_back({xlo_[i], xhi_[i], static_cast<unsigned int>(n_[i])});
  }
  return a;
}

std::vector<Axis> BlockSparseNSpace::make_superspace_axes(const std::vector<Axis>& axes) const
{
  std::vector<calin::math::nspace::Axis> a = axes;
  for(unsigned i=0; i<xlo_.size(); i++) {
    a.push_back({xlo_[i], xhi_[i], static_cast<unsigned int>(n_[i])});
  }
  return a;
}

Eigen::VectorXd BlockSparseNSpace::axis_bin_centers(unsigned iaxis) const
{
  if(iaxis >= n_.size()) {
    throw std::runtime_error("BlockSparseNSpace: iaxis out of range");
  }
  Eigen::VectorXd x(n_[iaxis]);
  for(int i=0; i<n_[iaxis]; i++) {
    x[i] = xlo_[iaxis] + dx_[iaxis] * (0.5 + i);
  }
  return x;
}

Eigen::VectorXd BlockSparseNSpace::axis_bin_edges(unsigned iaxis) const
{
  if(iaxis >= n_.size()) {
    throw std::runtime_error("BlockSparseNSpace: iaxis out of range");
  }
  Eigen::VectorXd x(n_[iaxis]+1);
  for(int i=0; i<=n_[iaxis]; i++) {
    x[i] = xlo_[iaxis] + dx_[iaxis] * i;
  }
  return x;
}

double BlockSparseNSpace::overflow_weight() const
{
  return overflow_;
}

double BlockSparseNSpace::weight(const Eigen::VectorXd& x) const
{
  int64_t array_index;
  int64_t block_index;
  if(index(x, array_index, block_index)) {
    return cell_val(array_index,block_index);
  } else {
    return overflow_;
  }
}

BlockSparseNSpace* BlockSparseNSpace::project_along_axis(unsigned iaxis,
  unsigned axis_cell_lo, unsigned axis_cell_hi, unsigned log2_block_size) const
{
  if(iaxis >= n_.size()) {
    throw std::runtime_error("BlockSparseNSpace: iaxis out of range");
  }

  std::vector<calin::math::nspace::Axis> a;
  for(unsigned i=0; i<xlo_.size(); i++) {
    if(i != iaxis) {
      a.push_back({xlo_[i], xhi_[i], static_cast<unsigned int>(n_[i])});
    }
  }

  BlockSparseNSpace* new_space = new BlockSparseNSpace(a, log2_block_size);
  new_space->overflow_ = overflow_;

  Eigen::VectorXi ix(n_.size());
  Eigen::VectorXi new_ix(n_.size() - 1);

  for(unsigned array_index=0; array_index<array_.size(); ++array_index) {
    double* block = array_[array_index];
    if(block) {
      for(unsigned block_index=0; block_index<block_size_; ++block_index) {
        if(bin_coords(ix, array_index, block_index)) {
          if(ix[iaxis] >= int(axis_cell_lo) and ix[iaxis] <= int(axis_cell_hi)) {
            if(iaxis != 0) {
              new_ix.head(iaxis) = ix.head(iaxis);
            }
            if(iaxis != n_.size()-1) {
              new_ix.tail(n_.size()-1-iaxis) = ix.tail(n_.size()-1-iaxis);
            }
            int64_t new_array_index;
            int64_t new_block_index;
            new_space->index_of_bin(new_ix, new_array_index, new_block_index);
            new_space->block_ptr(new_array_index)[new_block_index] += block[block_index];
          } else {
            new_space->overflow_ += block[block_index];
          }
        }
      }
    }
  }
  return new_space;
}

BlockSparseNSpace* BlockSparseNSpace::project_along_axis(unsigned iaxis,
  unsigned log2_block_size) const
{
  return this->project_along_axis(iaxis, 0, n_[std::min(iaxis,unsigned(n_.size()-1))], log2_block_size);
}

BlockSparseNSpace* BlockSparseNSpace::sum_x_to_the_n_along_axis(unsigned iaxis, unsigned n,
  unsigned axis_cell_lo, unsigned axis_cell_hi, unsigned log2_block_size) const
{
  if(iaxis >= n_.size()) {
    throw std::runtime_error("BlockSparseNSpace: iaxis out of range");
  }

  std::vector<calin::math::nspace::Axis> a;
  for(unsigned i=0; i<xlo_.size(); i++) {
    if(i != iaxis) {
      a.push_back({xlo_[i], xhi_[i], static_cast<unsigned int>(n_[i])});
    }
  }

  BlockSparseNSpace* new_space = new BlockSparseNSpace(a, log2_block_size);
  new_space->overflow_ = overflow_;

  Eigen::VectorXi ix(n_.size());
  Eigen::VectorXi new_ix(n_.size() - 1);

  for(unsigned array_index=0; array_index<array_.size(); ++array_index) {
    double* block = array_[array_index];
    if(block) {
      for(unsigned block_index=0; block_index<block_size_; ++block_index) {
        if(bin_coords(ix, array_index, block_index)) {
          double x = xlo_[iaxis] + dx_[iaxis]*(ix[iaxis] + 0.5);
          double xn = 1.0;
          switch(n) {
          case 0:
            break;
          case 1:
            xn = x;
            break;
          case 2:
            xn = x*x;
            break;
          case 3:
            xn = x*x*x;
            break;
          default:
            for(unsigned i=0;i<n;i++)xn *= x;
            break;
          }
          if(ix[iaxis] >= int(axis_cell_lo) and ix[iaxis] <= int(axis_cell_hi)) {
            if(iaxis != 0) {
              new_ix.head(iaxis) = ix.head(iaxis);
            }
            if(iaxis != n_.size()-1) {
              new_ix.tail(n_.size()-1-iaxis) = ix.tail(n_.size()-1-iaxis);
            }
            int64_t new_array_index;
            int64_t new_block_index;
            new_space->index_of_bin(new_ix, new_array_index, new_block_index);
            new_space->block_ptr(new_array_index)[new_block_index] += xn*block[block_index];
          } else {
            new_space->overflow_ += xn*block[block_index];
          }
        }
      }
    }
  }
  return new_space;
}

BlockSparseNSpace* BlockSparseNSpace::sum_x_to_the_n_along_axis(
  unsigned iaxis, unsigned n, unsigned log2_block_size) const
{
  return this->sum_x_to_the_n_along_axis(iaxis, n, 0, n_[std::min(iaxis,unsigned(n_.size()-1))], log2_block_size);
}

Eigen::MatrixXd BlockSparseNSpace::select_as_vector(const Eigen::VectorXi& bin_coords) const
{
  Eigen::VectorXi xi = bin_coords;
  if(xi.size() != n_.size()) {
    throw std::runtime_error("BlockSparseNSpace: dimensional mismatch in bin coordinates");
  }
  int iaxis = -1;
  for(unsigned i=0;i<n_.size();++i) {
    if(xi[i] < 0) {
      if(iaxis == -1) {
        iaxis = i;
      } else {
        throw std::runtime_error("BlockSparseNSpace: only one dimension can be extracted in vector");
      }
    } else if(xi[i] >= n_[i]) {
      throw std::runtime_error("BlockSparseNSpace: selected index out of range");
    }
  }
  if(iaxis == -1) {
    throw std::runtime_error("BlockSparseNSpace: one dimension must be chosen for extraction");
  }

  Eigen::VectorXd v(n_[iaxis]);
  v.setZero();

  for(int ix=0; ix<n_[iaxis]; ++ix) {
    xi[iaxis] = ix;
    int64_t array_index;
    int64_t block_index;
    if(index_of_bin(xi, array_index, block_index) and array_[array_index]!=nullptr) {
      v(ix) = array_[array_index][block_index];
    }
  }

  return v;
}

Eigen::MatrixXd BlockSparseNSpace::select_as_matrix(const Eigen::VectorXi& bin_coords) const
{
  Eigen::VectorXi xi = bin_coords;
  if(xi.size() != n_.size()) {
    throw std::runtime_error("BlockSparseNSpace: dimensional mismatch in bin coordinates");
  }
  int iaxis = -1;
  int jaxis = -1;
  for(unsigned i=0;i<n_.size();++i) {
    if(xi[i] < 0) {
      if(iaxis == -1) {
        iaxis = i;
      } else if(jaxis == -1) {
        jaxis = i;
      } else {
        throw std::runtime_error("BlockSparseNSpace: only two dimensions can be extracted in matrix");
      }
    } else if(xi[i] >= n_[i]) {
      throw std::runtime_error("BlockSparseNSpace: selected index out of range");
    }
  }
  if(jaxis == -1) {
    throw std::runtime_error("BlockSparseNSpace: two dimensions must be chosen for extraction");
  }

  Eigen::MatrixXd m(n_[iaxis],n_[jaxis]);
  m.setZero();

  for(int ix=0; ix<n_[iaxis]; ++ix) {
    xi[iaxis] = ix;
    for(unsigned jx=0; jx<unsigned(n_[jaxis]); ++jx) {
      xi[jaxis] = jx;
      int64_t array_index;
      int64_t block_index;
      if(index_of_bin(xi, array_index, block_index) and array_[array_index]!=nullptr) {
        m(ix, jx) = array_[array_index][block_index];
      }
    }
  }

  return m;
}


Eigen::VectorXd BlockSparseNSpace::as_vector() const
{
  if(n_.size() != 1) {
    throw std::runtime_error("BlockSparseNSpace: only single-axis spaces can be converted to vectors");
  }
  Eigen::VectorXi ix(n_.size() /* =1 */);
  Eigen::VectorXd v(n_[0]);
  v.setZero();

  for(unsigned array_index=0; array_index<array_.size(); ++array_index)
  {
    auto* block = array_[array_index];
    if(block) {
      for(unsigned block_index=0; block_index<block_size_; ++block_index) {
        if(bin_coords(ix, array_index, block_index)) {
          v(ix(0)) = block[block_index];
        }
      }
    }
  }
  return v;
}

Eigen::MatrixXd BlockSparseNSpace::as_matrix() const
{
  if(n_.size() != 2) {
    throw std::runtime_error("BlockSparseNSpace: only two-axis spaces can be converted to matrices");
  }
  Eigen::VectorXi ix(n_.size() /* =2 */);
  Eigen::MatrixXd m(n_[0],n_[1]);
  m.setZero();

  for(unsigned array_index=0; array_index<array_.size(); ++array_index)
  {
    auto* block = array_[array_index];
    if(block) {
      for(unsigned block_index=0; block_index<block_size_; ++block_index) {
        if(bin_coords(ix, array_index, block_index)) {
          m(ix(0),ix(1)) = block[block_index];
        }
      }
    }
  }
  return m;
}

uint64_t BlockSparseNSpace::num_occupied_cells() const
{
  uint64_t count = 0;
  for(auto* block : array_) {
    if(block) {
      for(unsigned i=0;i<block_size_;++i) {
        if(block[i]) {
          ++count;
        };
      }
    }
  }
  return count;
}

double BlockSparseNSpace::total_weight() const
{
  double w0 = 0;
  for(auto* block : array_) {
    if(block) {
      for(unsigned i=0;i<block_size_;++i) {
        w0 += block[i];
      }
    }
  }
  return w0;
}

Eigen::VectorXd BlockSparseNSpace::mean_and_total_weight(double& w0) const
{
  Eigen::VectorXd w1(xlo_.size());
  Eigen::VectorXd x(xlo_.size());
  w0 = 0;
  w1.setZero();
  for(unsigned array_index=0; array_index<array_.size(); ++array_index)
  {
    auto* block = array_[array_index];
    if(block) {
      for(unsigned block_index=0;block_index<block_size_;++block_index) {
        if(x_center(x, array_index, block_index)) {
          w0 += block[block_index];
          w1 += x * block[block_index];
        }
      }
    }
  }
  return w1/w0;
}

Eigen::VectorXd BlockSparseNSpace::mean() const
{
  double w0;
  return mean_and_total_weight(w0);
}

Eigen::MatrixXd BlockSparseNSpace::covar_mean_and_total_weight(Eigen::VectorXd& w1, double& w0) const
{
  Eigen::MatrixXd w2(xlo_.size(), xlo_.size());
  w1.resize(xlo_.size());
  Eigen::VectorXd x(xlo_.size());
  w0 = 0;
  w1.setZero();
  w2.setZero();
  for(unsigned array_index=0; array_index<array_.size(); ++array_index)
  {
    auto* block = array_[array_index];
    if(block) {
      for(unsigned block_index=0;block_index<block_size_;++block_index) {
        if(x_center(x, array_index, block_index)) {
          w0 += block[block_index];
          w1 += x * block[block_index];
          w2.noalias() += x * x.transpose() * block[block_index]; // outer product
        }
      }
    }
  }
  w1 /= w0;
  return w2/w0 - w1*w1.transpose(); // outer product
}

Eigen::MatrixXd BlockSparseNSpace::covar() const {
  double w0;
  Eigen::VectorXd w1(xlo_.size());
  return covar_mean_and_total_weight(w1, w0);
}

int64_t BlockSparseNSpace::map_index(const Eigen::VectorXi& ix) const
{
  if(ix.size() != xlo_.size()) {
    throw std::runtime_error("BlockSparseNSpace: map coordinates dimensional mismatch");
  }
  int64_t indx = 0;
  for(int i=0; i<xlo_.size(); i++) {
    int ii = ix[i];
    if(ii<0 or ii>=n_(i)) {
      throw std::runtime_error("BlockSparseNSpace: map coordinates out of range");
    }
    indx = indx*n_(i) + ii;
  }
  return indx;
}

void BlockSparseNSpace::map_bin_coords(Eigen::VectorXi& ix_out, int64_t indx) const
{
  if(indx < 0 or indx >= N_) {
    throw std::runtime_error("BlockSparseNSpace: map index out of range");
  }
  int n = xlo_.size();
  ix_out.resize(n);
  for(unsigned i=n; i>0;) {
    --i;
    auto qr = std::div(indx, int64_t(n_[i]));
    ix_out[i] = qr.rem;
    indx = qr.quot;
  }
}

void BlockSparseNSpace::save_to_proto(ix::math::nspace::NSpaceData* proto) const
{
  proto->Clear();
  proto->set_num_axes(xlo_.size());
  for(unsigned iaxis=0;iaxis<xlo_.size();++iaxis) {
    proto->add_axes_lower_bounds(xlo_[iaxis]);
    proto->add_axes_upper_bounds(xhi_[iaxis]);
    proto->add_axes_num_bins(n_[iaxis]);
  }
  Eigen::VectorXi ix(xlo_.size());
  unsigned nblock = array_.size() - std::count(array_.begin(), array_.end(), nullptr);
  proto->mutable_bin_indices()->Reserve(nblock * block_size_);
  proto->mutable_bin_weights()->Reserve(nblock * block_size_);
  for(unsigned array_index=0; array_index<array_.size(); ++array_index)
  {
    auto* block = array_[array_index];
    if(block) {
      for(unsigned block_index=0;block_index<block_size_;++block_index) {
        if(block[block_index] and bin_coords(ix, array_index, block_index)) {
          proto->add_bin_indices(map_index(ix));
          proto->add_bin_weights(block[block_index]);
        }
      }
    }
  }
  if(overflow_) {
    proto->set_overflow_weight(overflow_);
  }
}

calin::ix::math::nspace::NSpaceData* BlockSparseNSpace::as_proto() const
{
  auto* proto = new calin::ix::math::nspace::NSpaceData;
  save_to_proto(proto); 
  return proto;
}

void BlockSparseNSpace::accumulate_from_proto(const calin::ix::math::nspace::NSpaceData& proto)
{
  if(not std::equal(xlo_.begin(), xlo_.end(), proto.axes_lower_bounds().begin(), proto.axes_lower_bounds().end())) {
    throw std::runtime_error("BlockSparseNSpace: lower bounds do not match");
  }
  if(not std::equal(xhi_.begin(), xhi_.end(), proto.axes_upper_bounds().begin(), proto.axes_upper_bounds().end())) {
    throw std::runtime_error("BlockSparseNSpace: upper bounds do not match");
  }
  if(not std::equal(n_.begin(), n_.end(), proto.axes_num_bins().begin(), proto.axes_num_bins().end())) {
    throw std::runtime_error("BlockSparseNSpace: number of bins do not match");
  }
  if(proto.bin_indices_size() != proto.bin_weights_size()) {
    throw std::runtime_error("BlockSparseNSpace: bin indices and weights do not match");
  }
  Eigen::VectorXi ix(xlo_.size());
  int64_t array_index;
  int64_t block_index;
  for(int ibin=0; ibin<proto.bin_indices_size(); ++ibin) {
    map_bin_coords(ix, proto.bin_indices(ibin));
    index_of_bin(ix, array_index, block_index);
    cell_ref(array_index,block_index) += proto.bin_weights(ibin);
  }
  if(proto.overflow_weight() != 0.0) {
    overflow_ += proto.overflow_weight();
  }
}

BlockSparseNSpace* BlockSparseNSpace::create_from_proto(const calin::ix::math::nspace::NSpaceData& proto, 
  unsigned log2_block_size)
{
  return new BlockSparseNSpace(proto, log2_block_size);
}

#if 0
void BlockSparseNSpace::subspace_covar_mean_and_total_weight(const Eigen::VectorXi& subspace_axes,
  BlockSparseNSpace** w2_space, BlockSparseNSpace** w1_space, BlockSparseNSpace** w0_space)
{
  if(subspace_axes.size() == 0) {
    std::runtime_error("BlockSparseNSpace: subspace should have at least one axis");
  }
  for(unsigned i=0; i<subspace_axes.size(); ++i) {
    if(subspace_axes[i] >= n_.size() {
      throw std::runtime_error("BlockSparseNSpace: subspace axis out of range");
    }
  }

}
#endif

#if 0
#include <blosc.h>
uint64_t BlockSparseNSpace::test_blosc_size(int clevel, int doshuffle,
  const std::string& compname, int nthreads) const
{
  uint64_t blosc_size = 0;
  double* cdata = new double[block_size_ + 16];
  blosc_init();
  blosc_set_nthreads(std::max(nthreads,1));
  blosc_set_compressor(compname.c_str());
  for(unsigned array_index=0; array_index<array_.size(); ++array_index)
  {
    auto* block = array_[array_index];
    if(block) {
      int csize = blosc_compress(clevel, doshuffle, sizeof(double),
        block_size_*sizeof(double), block,
        cdata, (block_size_+16)*sizeof(double));
      if (csize == 0) {
        throw std::runtime_error("Buffer is uncompressible.");
      } else if (csize < 0) {
        throw std::runtime_error("Blosc error reported.");
      }
      blosc_size += csize;
    }
  }
  blosc_destroy();
  delete[] cdata;
  return blosc_size;
}
#endif
