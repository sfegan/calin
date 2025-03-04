/*

   calin/diagnostics/simple_charge_stats.cpp -- Stephen Fegan -- 2020-03-22

   Channel info diagnostics

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

#include <algorithm>

#include <util/log.hpp>
#include <math/special.hpp>
#include <util/algorithm.hpp>
#include <diagnostics/simple_charge_stats.hpp>
#include <math/covariance_calc.hpp>
#include <iact_data/algorithms.hpp>
#include <io/json.hpp>
#include <util/string.hpp>

using namespace calin::util::log;
using namespace calin::diagnostics::simple_charge_stats;
using namespace calin::ix::diagnostics::simple_charge_stats;
using namespace calin::iact_data::waveform_treatment_event_visitor;

using calin::math::special::SQR;
using calin::math::covariance_calc::cov_i64_gen;
using calin::math::covariance_calc::cov_double_gen;

SimpleChargeStatsParallelEventVisitor::
SimpleChargeStatsParallelEventVisitor(
    OptimalWindowSumWaveformTreatmentParallelEventVisitor* high_gain_visitor,
    OptimalWindowSumWaveformTreatmentParallelEventVisitor* low_gain_visitor,
    const SimpleChargeStatsConfig& config):
  ParallelEventVisitor(), config_(config),
  high_gain_visitor_(high_gain_visitor), low_gain_visitor_(low_gain_visitor)
{
  // nothing to see here
}

SimpleChargeStatsParallelEventVisitor::~SimpleChargeStatsParallelEventVisitor()
{
  for(auto* h : chan_hists_)delete h;
  delete camera_hists_;
  delete data_order_camera_;
}

SimpleChargeStatsParallelEventVisitor* SimpleChargeStatsParallelEventVisitor::new_sub_visitor(
  std::map<calin::iact_data::event_visitor::ParallelEventVisitor*,
    calin::iact_data::event_visitor::ParallelEventVisitor*> antecedent_visitors)
{
  auto* hgv = dynamic_cast<OptimalWindowSumWaveformTreatmentParallelEventVisitor*>(
    antecedent_visitors[high_gain_visitor_]);
  auto* lgv = dynamic_cast<OptimalWindowSumWaveformTreatmentParallelEventVisitor*>(
    antecedent_visitors[low_gain_visitor_]); // good in case of nullptr also
  auto* child = new SimpleChargeStatsParallelEventVisitor(hgv, lgv, config_);
  child->parent_ = this;
  return child;
}

bool SimpleChargeStatsParallelEventVisitor::visit_telescope_run(
  const calin::ix::iact_data::telescope_run_configuration::TelescopeRunConfiguration* run_config,
  calin::iact_data::event_visitor::EventLifetimeManager* event_lifetime_manager,
  calin::ix::provenance::chronicle::ProcessingRecord* processing_record)
{
  if(processing_record) {
    processing_record->set_type("SimpleChargeStatsParallelEventVisitor");
    processing_record->set_description("Per-channel waveform statistics and analysis");
    auto* config_json = processing_record->add_config();
    config_json->set_type(config_.GetTypeName());
    config_json->set_json(calin::io::json::encode_protobuf_to_json_string(config_));
    config_json = processing_record->add_config();
    std::vector<std::pair<std::string,std::string> > keyval;
    keyval.emplace_back("highGainWaveformSumInstance",
      calin::io::json::json_string_value(calin::util::string::instance_identifier(high_gain_visitor_)));
    keyval.emplace_back("lowGainWaveformSumInstance",
      calin::io::json::json_string_value(calin::util::string::instance_identifier(low_gain_visitor_)));
    config_json->set_json(calin::io::json::json_for_dictionary(keyval));
  }

  partials_.Clear();
  results_.Clear();

  has_dual_gain_ = (run_config->camera_layout().adc_gains() !=
    calin::ix::iact_data::instrument_layout::CameraLayout::SINGLE_GAIN);
  for(int ichan=0;ichan<run_config->configured_channel_id_size();++ichan) {
    partials_.add_channel();
  }

  for(auto* h : chan_hists_) {
    delete h;
  }
  chan_hists_.resize(run_config->configured_channel_id_size());
  for(auto*& h : chan_hists_) {
    // For now on do not use per-channel low gain histograms
    h = new ChannelHists(has_dual_gain_, config_.ped_time_hist_resolution(), 
      config_.channel_ped_time_hist_range(),
      config_.dual_gain_sample_resolution(), config_.dual_gain_sum_resolution());
  }

  delete camera_hists_;
  camera_hists_ = new CameraHists(has_dual_gain_, config_.ped_time_hist_resolution(), 
    config_.camera_ped_time_hist_range());

  delete data_order_camera_;
  data_order_camera_ = calin::iact_data::instrument_layout::reorder_camera_channels(
    run_config->camera_layout(),
    reinterpret_cast<const unsigned*>(run_config->configured_channel_id().data()),
    run_config->configured_channel_id_size());
  channel_island_id_.resize(run_config->configured_channel_id_size());
  channel_island_count_.resize(run_config->configured_channel_id_size());

  // if(high_gain_visitor_) {
  //   high_gain_results_->Clear();
  //   high_gain_results_->set_integration_n(high_gain_visitor_->window_n());
  //   high_gain_results_->set_bkg_integration_0(high_gain_visitor_->bkg_window_0());
  //   int sig_win_0 = -2;
  //   for(int sw0 : high_gain_visitor_->sig_window_0()) {
  //     if(sig_win_0==-2)sig_win_0 = sw0;
  //     else if(sig_win_0 != sw0)sig_win_0 = -1;
  //     high_gain_results_->add_chan_sig_integration_0(sw0);
  //   };
  //   high_gain_results_->sig_integration_0(std::max(sig_win_0,-1));
  // }
  // if(low_gain_visitor_) {
  //   low_gain_results_->Clear();
  //   low_gain_results_->set_integration_n(low_gain_visitor_->window_n());
  //   low_gain_results_->set_bkg_integration_0(low_gain_visitor_->bkg_window_0());
  //   int sig_win_0 = -2;
  //   for(int sw0 : low_gain_visitor_->sig_window_0()) {
  //     if(sig_win_0==-2)sig_win_0 = sw0;
  //     else if(sig_win_0 != sw0)sig_win_0 = -1;
  //     low_gain_results_->add_chan_sig_integration_0(sw0);
  //   };
  //   low_gain_results_->sig_integration_0(std::max(sig_win_0,-1));
  // }
  return true;
}

void SimpleChargeStatsParallelEventVisitor::integrate_one_gain_partials(
  calin::ix::diagnostics::simple_charge_stats::OneGainSimpleChargeStats* results_g,
  const calin::ix::diagnostics::simple_charge_stats::PartialOneGainChannelSimpleChargeStats& partials_gc)
{
  results_g->add_all_trigger_event_count(partials_gc.all_trig_num_events());
  if(partials_gc.all_trig_num_events() > 0) {
    results_g->add_all_trigger_ped_win_mean(
      double(partials_gc.all_trig_ped_win_sum())/double(partials_gc.all_trig_num_events()));
    results_g->add_all_trigger_ped_win_var(cov_i64_gen(
      partials_gc.all_trig_ped_win_sumsq(), partials_gc.all_trig_num_events(),
      partials_gc.all_trig_ped_win_sum(), partials_gc.all_trig_num_events(),
      partials_gc.all_trig_ped_win_sum(), partials_gc.all_trig_num_events()));
  } else {
    results_g->add_all_trigger_ped_win_mean(0.0);
    results_g->add_all_trigger_ped_win_var(0.0);
  }
  results_g->add_all_trigger_num_wf_clipped(partials_gc.all_trig_num_wf_clipped());

  if(partials_gc.has_all_trig_pedwin_vs_time_1_sum()) {
    auto* mean_hist = new calin::ix::math::histogram::Histogram1DData;
    auto* var_hist = new calin::ix::math::histogram::Histogram1DData;
    mean_hist->CopyFrom(partials_gc.all_trig_pedwin_vs_time_q_sum());
    var_hist->CopyFrom(partials_gc.all_trig_pedwin_vs_time_q2_sum());
    for(int ibin=0;ibin<partials_gc.all_trig_pedwin_vs_time_1_sum().bins_size();ibin++) {
      double count = partials_gc.all_trig_pedwin_vs_time_1_sum().bins(ibin);
      if(count>0) {
        mean_hist->set_bins(ibin, mean_hist->bins(ibin)/count);
        var_hist->set_bins(ibin, var_hist->bins(ibin)/count - SQR(mean_hist->bins(ibin)));
      }
    }
    calin::math::histogram::sparsify(partials_gc.all_trig_pedwin_vs_time_1_sum(),
      results_g->add_all_trigger_ped_win_count_vs_time());
    calin::math::histogram::sparsify(*mean_hist,
      results_g->add_all_trigger_ped_win_mean_vs_time());
    calin::math::histogram::sparsify(*var_hist,
      results_g->add_all_trigger_ped_win_var_vs_time());
    delete var_hist;
    delete mean_hist;
  }

  if(partials_gc.has_ped_trig_vs_time_1_sum()) {
    auto* mean_hist = new calin::ix::math::histogram::Histogram1DData;
    auto* var_hist = new calin::ix::math::histogram::Histogram1DData;
    mean_hist->CopyFrom(partials_gc.ped_trig_vs_time_q_sum());
    var_hist->CopyFrom(partials_gc.ped_trig_vs_time_q2_sum());
    for(int ibin=0;ibin<partials_gc.ped_trig_vs_time_1_sum().bins_size();ibin++) {
      double count = partials_gc.ped_trig_vs_time_1_sum().bins(ibin);
      if(count>0) {
        mean_hist->set_bins(ibin, mean_hist->bins(ibin)/count);
        var_hist->set_bins(ibin, var_hist->bins(ibin)/count - SQR(mean_hist->bins(ibin)));
      }
    }
    calin::math::histogram::sparsify(partials_gc.ped_trig_vs_time_1_sum(),
      results_g->add_ped_trigger_full_wf_count_vs_time());
    calin::math::histogram::sparsify(*mean_hist,
      results_g->add_ped_trigger_full_wf_mean_vs_time());
    calin::math::histogram::sparsify(*var_hist,
      results_g->add_ped_trigger_full_wf_var_vs_time());
    delete var_hist;
    delete mean_hist;
  }

  results_g->add_ped_trigger_event_count(partials_gc.ped_trig_num_events());
  if(partials_gc.ped_trig_num_events() > 0) {
    results_g->add_ped_trigger_full_wf_mean(
      double(partials_gc.ped_trig_full_wf_sum())/double(partials_gc.ped_trig_num_events()));
    results_g->add_ped_trigger_full_wf_var(cov_i64_gen(
      partials_gc.ped_trig_full_wf_sumsq(), partials_gc.ped_trig_num_events(),
      partials_gc.ped_trig_full_wf_sum(), partials_gc.ped_trig_num_events(),
      partials_gc.ped_trig_full_wf_sum(), partials_gc.ped_trig_num_events()));

    results_g->add_ped_trigger_ped_win_mean(
      double(partials_gc.ped_trig_ped_win_sum())/double(partials_gc.ped_trig_num_events()));
    results_g->add_ped_trigger_ped_win_var(cov_i64_gen(
      partials_gc.ped_trig_ped_win_sumsq(), partials_gc.ped_trig_num_events(),
      partials_gc.ped_trig_ped_win_sum(), partials_gc.ped_trig_num_events(),
      partials_gc.ped_trig_ped_win_sum(), partials_gc.ped_trig_num_events()));

    results_g->add_ped_trigger_sig_win_mean(
      double(partials_gc.ped_trig_sig_win_sum())/double(partials_gc.ped_trig_num_events()));
    results_g->add_ped_trigger_sig_win_var(cov_i64_gen(
      partials_gc.ped_trig_sig_win_sumsq(), partials_gc.ped_trig_num_events(),
      partials_gc.ped_trig_sig_win_sum(), partials_gc.ped_trig_num_events(),
      partials_gc.ped_trig_sig_win_sum(), partials_gc.ped_trig_num_events()));

    results_g->add_ped_trigger_opt_win_mean(
      double(partials_gc.ped_trig_opt_win_sum())/double(partials_gc.ped_trig_num_events()));
    results_g->add_ped_trigger_opt_win_var(cov_i64_gen(
      partials_gc.ped_trig_opt_win_sumsq(), partials_gc.ped_trig_num_events(),
      partials_gc.ped_trig_opt_win_sum(), partials_gc.ped_trig_num_events(),
      partials_gc.ped_trig_opt_win_sum(), partials_gc.ped_trig_num_events()));
  } else {
    results_g->add_ped_trigger_full_wf_mean(0.0);
    results_g->add_ped_trigger_full_wf_var(0.0);
    results_g->add_ped_trigger_ped_win_mean(0.0);
    results_g->add_ped_trigger_ped_win_var(0.0);
    results_g->add_ped_trigger_sig_win_mean(0.0);
    results_g->add_ped_trigger_sig_win_var(0.0);
    results_g->add_ped_trigger_opt_win_mean(0.0);
    results_g->add_ped_trigger_opt_win_var(0.0);
  }
  results_g->add_ped_trigger_num_wf_clipped(partials_gc.ped_trig_num_wf_clipped());

  results_g->add_ext_trigger_event_count(partials_gc.ext_trig_num_events());
  if(partials_gc.ext_trig_num_events() > 0) {
    results_g->add_ext_trigger_sig_win_mean(
      double(partials_gc.ext_trig_sig_win_sum())/double(partials_gc.ext_trig_num_events()));
    results_g->add_ext_trigger_sig_win_var(cov_i64_gen(
      partials_gc.ext_trig_sig_win_sumsq(), partials_gc.ext_trig_num_events(),
      partials_gc.ext_trig_sig_win_sum(), partials_gc.ext_trig_num_events(),
      partials_gc.ext_trig_sig_win_sum(), partials_gc.ext_trig_num_events()));

    results_g->add_ext_trigger_opt_win_mean(
      double(partials_gc.ext_trig_opt_win_sum())/double(partials_gc.ext_trig_num_events()));
    results_g->add_ext_trigger_opt_win_var(cov_i64_gen(
      partials_gc.ext_trig_opt_win_sumsq(), partials_gc.ext_trig_num_events(),
      partials_gc.ext_trig_opt_win_sum(), partials_gc.ext_trig_num_events(),
      partials_gc.ext_trig_opt_win_sum(), partials_gc.ext_trig_num_events()));
  } else {
    results_g->add_ext_trigger_sig_win_mean(0.0);
    results_g->add_ext_trigger_sig_win_var(0.0);
    results_g->add_ext_trigger_opt_win_mean(0.0);
    results_g->add_ext_trigger_opt_win_var(0.0);
  }
  results_g->add_ext_trigger_num_wf_clipped(partials_gc.ext_trig_num_wf_clipped());

  results_g->add_phy_trigger_event_count(partials_gc.phys_trig_num_events());
  results_g->add_phy_trigger_num_wf_clipped(partials_gc.phys_trig_num_wf_clipped());

  results_g->add_int_trigger_event_count(partials_gc.int_trig_num_events());
  results_g->add_int_trigger_num_wf_clipped(partials_gc.int_trig_num_wf_clipped());
}

void SimpleChargeStatsParallelEventVisitor::integrate_dual_gain_partials(
  calin::ix::diagnostics::simple_charge_stats::DualGainSimpleChargeStats* results_g,
  const calin::ix::diagnostics::simple_charge_stats::PartialDualGainChannelSimpleChargeStats& partials_gc)
{
  if(partials_gc.has_all_max_sample_1()) {
    auto* mean_hist = new calin::ix::math::histogram::Histogram1DData;
    auto* var_hist = new calin::ix::math::histogram::Histogram1DData;
    mean_hist->CopyFrom(partials_gc.all_max_sample_q());
    var_hist->CopyFrom(partials_gc.all_max_sample_q2());
    for(int ibin=0;ibin<partials_gc.all_max_sample_1().bins_size();ibin++) {
      double count = partials_gc.all_max_sample_1().bins(ibin);
      if(count>0) {
        mean_hist->set_bins(ibin, mean_hist->bins(ibin)/count);
        var_hist->set_bins(ibin, var_hist->bins(ibin)/count - SQR(mean_hist->bins(ibin)));
      }
    }
    calin::math::histogram::sparsify(partials_gc.all_max_sample_1(), results_g->add_all_max_sample_count());
    calin::math::histogram::sparsify(*mean_hist, results_g->add_all_max_sample_mean());
    calin::math::histogram::sparsify(*var_hist, results_g->add_all_max_sample_var());
    delete var_hist;
    delete mean_hist;
  }

  if(partials_gc.has_all_opt_sum_1()) {
    auto* mean_hist = new calin::ix::math::histogram::Histogram1DData;
    auto* var_hist = new calin::ix::math::histogram::Histogram1DData;
    mean_hist->CopyFrom(partials_gc.all_opt_sum_q());
    var_hist->CopyFrom(partials_gc.all_opt_sum_q2());
    for(int ibin=0;ibin<partials_gc.all_opt_sum_1().bins_size();ibin++) {
      double count = partials_gc.all_opt_sum_1().bins(ibin);
      if(count>0) {
        mean_hist->set_bins(ibin, mean_hist->bins(ibin)/count);
        var_hist->set_bins(ibin, var_hist->bins(ibin)/count - SQR(mean_hist->bins(ibin)));
      }
    }
    calin::math::histogram::sparsify(partials_gc.all_opt_sum_1(), results_g->add_all_opt_sum_count());
    calin::math::histogram::sparsify(*mean_hist, results_g->add_all_opt_sum_mean());
    calin::math::histogram::sparsify(*var_hist, results_g->add_all_opt_sum_var());
    delete var_hist;
    delete mean_hist;
  }
}

void SimpleChargeStatsParallelEventVisitor::integrate_one_gain_camera_partials(
  calin::ix::diagnostics::simple_charge_stats::OneGainSimpleChargeStats* results_g,
  const calin::ix::diagnostics::simple_charge_stats::PartialOneGainCameraSimpleChargeStats& partials_g)
{
  results_g->set_ext_trigger_all_channel_count(partials_g.ext_trig_all_num_events());
  if(partials_g.ext_trig_all_num_events() > 0) {
    results_g->set_ext_trigger_all_channel_sig_win_mean(
      double(partials_g.ext_trig_all_sig_win_sum())/double(partials_g.ext_trig_all_num_events()));
    results_g->set_ext_trigger_all_channel_sig_win_var(cov_double_gen(
      partials_g.ext_trig_all_sig_win_sumsq(), partials_g.ext_trig_all_num_events(),
      partials_g.ext_trig_all_sig_win_sum(), partials_g.ext_trig_all_num_events(),
      partials_g.ext_trig_all_sig_win_sum(), partials_g.ext_trig_all_num_events()));

    results_g->set_ext_trigger_all_channel_opt_win_mean(
      double(partials_g.ext_trig_all_opt_win_sum())/double(partials_g.ext_trig_all_num_events()));
    results_g->set_ext_trigger_all_channel_opt_win_var(cov_double_gen(
      partials_g.ext_trig_all_opt_win_sumsq(), partials_g.ext_trig_all_num_events(),
      partials_g.ext_trig_all_opt_win_sum(), partials_g.ext_trig_all_num_events(),
      partials_g.ext_trig_all_opt_win_sum(), partials_g.ext_trig_all_num_events()));
  } else {
    results_g->add_ext_trigger_sig_win_mean(0.0);
    results_g->add_ext_trigger_sig_win_var(0.0);
    results_g->add_ext_trigger_opt_win_mean(0.0);
    results_g->add_ext_trigger_opt_win_var(0.0);
  }

  if(partials_g.has_all_trig_pedwin_vs_time_1_sum()) {
    auto* mean_hist = new calin::ix::math::histogram::Histogram1DData;
    auto* var_hist = new calin::ix::math::histogram::Histogram1DData;
    mean_hist->CopyFrom(partials_g.all_trig_pedwin_vs_time_q_sum());
    var_hist->CopyFrom(partials_g.all_trig_pedwin_vs_time_q2_sum());
    for(int ibin=0;ibin<partials_g.all_trig_pedwin_vs_time_1_sum().bins_size();ibin++) {
      double count = partials_g.all_trig_pedwin_vs_time_1_sum().bins(ibin);
      if(count>0) {
        mean_hist->set_bins(ibin, mean_hist->bins(ibin)/count);
        var_hist->set_bins(ibin, var_hist->bins(ibin)/count - SQR(mean_hist->bins(ibin)));
      }
    }
    calin::math::histogram::sparsify(partials_g.all_trig_pedwin_vs_time_1_sum(),
      results_g->mutable_camera_all_trigger_ped_win_count_vs_time());
    calin::math::histogram::sparsify(*mean_hist,
      results_g->mutable_camera_all_trigger_ped_win_mean_vs_time());
    calin::math::histogram::sparsify(*var_hist,
      results_g->mutable_camera_all_trigger_ped_win_var_vs_time());
    delete var_hist;
    delete mean_hist;
  }

  if(partials_g.has_ped_trig_vs_time_1_sum()) {
    auto* mean_hist = new calin::ix::math::histogram::Histogram1DData;
    auto* var_hist = new calin::ix::math::histogram::Histogram1DData;
    mean_hist->CopyFrom(partials_g.ped_trig_vs_time_q_sum());
    var_hist->CopyFrom(partials_g.ped_trig_vs_time_q2_sum());
    for(int ibin=0;ibin<partials_g.ped_trig_vs_time_1_sum().bins_size();ibin++) {
      double count = partials_g.ped_trig_vs_time_1_sum().bins(ibin);
      if(count>0) {
        mean_hist->set_bins(ibin, mean_hist->bins(ibin)/count);
        var_hist->set_bins(ibin, var_hist->bins(ibin)/count - SQR(mean_hist->bins(ibin)));
      }
    }
    calin::math::histogram::sparsify(partials_g.ped_trig_vs_time_1_sum(),
      results_g->mutable_camera_ped_trigger_full_wf_count_vs_time());
    calin::math::histogram::sparsify(*mean_hist,
      results_g->mutable_camera_ped_trigger_full_wf_mean_vs_time());
    calin::math::histogram::sparsify(*var_hist,
      results_g->mutable_camera_ped_trigger_full_wf_var_vs_time());
    delete var_hist;
    delete mean_hist;
  }

}

void SimpleChargeStatsParallelEventVisitor::dump_single_gain_channel_hists_to_partials(
  const SingleGainChannelHists& hists,
  calin::ix::diagnostics::simple_charge_stats::PartialOneGainChannelSimpleChargeStats* partials)
{
  auto* hp = hists.all_pedwin_1_sum_vs_time->dump_as_proto();
  partials->mutable_all_trig_pedwin_vs_time_1_sum()->IntegrateFrom(*hp);

  hists.all_pedwin_q_sum_vs_time->dump_as_proto(hp);
  partials->mutable_all_trig_pedwin_vs_time_q_sum()->IntegrateFrom(*hp);

  hists.all_pedwin_q2_sum_vs_time->dump_as_proto(hp);
  partials->mutable_all_trig_pedwin_vs_time_q2_sum()->IntegrateFrom(*hp);

  hists.ped_wf_1_sum_vs_time->dump_as_proto(hp);
  partials->mutable_ped_trig_vs_time_1_sum()->IntegrateFrom(*hp);

  hists.ped_wf_q_sum_vs_time->dump_as_proto(hp);
  partials->mutable_ped_trig_vs_time_q_sum()->IntegrateFrom(*hp);

  hists.ped_wf_q2_sum_vs_time->dump_as_proto(hp);
  partials->mutable_ped_trig_vs_time_q2_sum()->IntegrateFrom(*hp);

  delete hp;
}

void SimpleChargeStatsParallelEventVisitor::dump_dual_gain_channel_hists_to_partials(
  const DualGainChannelHists& hists,
  calin::ix::diagnostics::simple_charge_stats::PartialDualGainChannelSimpleChargeStats* partials)
{
  auto* hp = hists.all_max_sample_1->dump_as_proto();
  partials->mutable_all_max_sample_1()->IntegrateFrom(*hp);

  hists.all_max_sample_q->dump_as_proto(hp);
  partials->mutable_all_max_sample_q()->IntegrateFrom(*hp);

  hists.all_max_sample_q2->dump_as_proto(hp);
  partials->mutable_all_max_sample_q2()->IntegrateFrom(*hp);

  hists.all_opt_sum_1->dump_as_proto(hp);
  partials->mutable_all_opt_sum_1()->IntegrateFrom(*hp);

  hists.all_opt_sum_q->dump_as_proto(hp);
  partials->mutable_all_opt_sum_q()->IntegrateFrom(*hp);

  hists.all_opt_sum_q2->dump_as_proto(hp);
  partials->mutable_all_opt_sum_q2()->IntegrateFrom(*hp);

  delete hp;
}

void SimpleChargeStatsParallelEventVisitor::dump_single_gain_camera_hists_to_partials(
  const SingleGainChannelHists& hists,
  calin::ix::diagnostics::simple_charge_stats::PartialOneGainCameraSimpleChargeStats* partials)
{
  auto* hp = hists.all_pedwin_1_sum_vs_time->dump_as_proto();
  partials->mutable_all_trig_pedwin_vs_time_1_sum()->IntegrateFrom(*hp);

  hists.all_pedwin_q_sum_vs_time->dump_as_proto(hp);
  partials->mutable_all_trig_pedwin_vs_time_q_sum()->IntegrateFrom(*hp);

  hists.all_pedwin_q2_sum_vs_time->dump_as_proto(hp);
  partials->mutable_all_trig_pedwin_vs_time_q2_sum()->IntegrateFrom(*hp);

  hists.ped_wf_1_sum_vs_time->dump_as_proto(hp);
  partials->mutable_ped_trig_vs_time_1_sum()->IntegrateFrom(*hp);

  hists.ped_wf_q_sum_vs_time->dump_as_proto(hp);
  partials->mutable_ped_trig_vs_time_q_sum()->IntegrateFrom(*hp);

  hists.ped_wf_q2_sum_vs_time->dump_as_proto(hp);
  partials->mutable_ped_trig_vs_time_q2_sum()->IntegrateFrom(*hp);

  delete hp;
}

void SimpleChargeStatsParallelEventVisitor::SingleGainChannelHists::insert_from_and_clear(
  SimpleChargeStatsParallelEventVisitor::SingleGainChannelHists* from)
{
  all_pedwin_1_sum_vs_time->insert_hist(*from->all_pedwin_1_sum_vs_time);
  from->all_pedwin_1_sum_vs_time->clear();

  all_pedwin_q_sum_vs_time->insert_hist(*from->all_pedwin_q_sum_vs_time);
  from->all_pedwin_q_sum_vs_time->clear();

  all_pedwin_q2_sum_vs_time->insert_hist(*from->all_pedwin_q2_sum_vs_time);
  from->all_pedwin_q2_sum_vs_time->clear();

  ped_wf_1_sum_vs_time->insert_hist(*from->ped_wf_1_sum_vs_time);
  from->ped_wf_1_sum_vs_time->clear();

  ped_wf_q_sum_vs_time->insert_hist(*from->ped_wf_q_sum_vs_time);
  from->ped_wf_q_sum_vs_time->clear();

  ped_wf_q2_sum_vs_time->insert_hist(*from->ped_wf_q2_sum_vs_time);
  from->ped_wf_q2_sum_vs_time->clear();
}

void SimpleChargeStatsParallelEventVisitor::merge_time_histograms_if_necessary()
{
  // This function added to do on-the-fly merging of the time histograms if they get very
  // large. Only used on very long runs with more than 1000 bins (corresponding to 5000 
  // seconds by default), to avoid memory problems in multi-threaded envrionment, since these
  // histograms are duplicated across all the threads

  if(parent_ == nullptr or camera_hists_->high_gain->all_pedwin_1_sum_vs_time->size()<1000) {
    return;
  }

  std::lock_guard<std::mutex> lock { parent_->on_the_fly_merge_lock_ };

  for(int ichan = 0; ichan<chan_hists_.size(); ichan++) {
    if(chan_hists_[ichan]->high_gain) {
      parent_->chan_hists_[ichan]->high_gain->insert_from_and_clear(chan_hists_[ichan]->high_gain);
    }
    if(chan_hists_[ichan]->low_gain) {
      parent_->chan_hists_[ichan]->low_gain->insert_from_and_clear(chan_hists_[ichan]->low_gain);
    }
  }
  if(camera_hists_->high_gain) {
    parent_->camera_hists_->high_gain->insert_from_and_clear(camera_hists_->high_gain);
  }
  if(camera_hists_->low_gain) {
    parent_->camera_hists_->low_gain->insert_from_and_clear(camera_hists_->low_gain);
  }
}

bool SimpleChargeStatsParallelEventVisitor::leave_telescope_run(
  calin::ix::provenance::chronicle::ProcessingRecord* processing_record)
{
  for(int ichan = 0; ichan<partials_.channel_size(); ichan++) {
    dump_single_gain_channel_hists_to_partials(*chan_hists_[ichan]->high_gain,
      partials_.mutable_channel(ichan)->mutable_high_gain());
    delete chan_hists_[ichan]->high_gain;
    chan_hists_[ichan]->high_gain = nullptr;
    if(has_dual_gain_ and chan_hists_[ichan]->low_gain) {
      dump_single_gain_channel_hists_to_partials(*chan_hists_[ichan]->low_gain,
        partials_.mutable_channel(ichan)->mutable_low_gain());
      delete chan_hists_[ichan]->low_gain;
      chan_hists_[ichan]->low_gain = nullptr;
    }
    if(has_dual_gain_ and chan_hists_[ichan]->dual_gain) {
      dump_dual_gain_channel_hists_to_partials(*chan_hists_[ichan]->dual_gain,
        partials_.mutable_channel(ichan)->mutable_dual_gain());
      delete chan_hists_[ichan]->dual_gain;
      chan_hists_[ichan]->dual_gain = nullptr;
    }
  }

  dump_single_gain_camera_hists_to_partials(*camera_hists_->high_gain,
    partials_.mutable_camera()->mutable_high_gain());
  if(has_dual_gain_ and camera_hists_->low_gain) {
    dump_single_gain_camera_hists_to_partials(*camera_hists_->low_gain,
      partials_.mutable_camera()->mutable_low_gain());
  }
  auto* hp = camera_hists_->num_channel_triggered_hist->dump_as_proto();
  partials_.mutable_camera()->mutable_num_channel_triggered_hist()->IntegrateFrom(*hp);
  delete hp;

  hp = camera_hists_->num_contiguous_channel_triggered_hist->dump_as_proto();
  partials_.mutable_camera()->mutable_num_contiguous_channel_triggered_hist()->IntegrateFrom(*hp);
  delete hp;

  hp = camera_hists_->phys_trig_num_channel_triggered_hist->dump_as_proto();
  partials_.mutable_camera()->mutable_phys_trig_num_channel_triggered_hist()->IntegrateFrom(*hp);
  delete hp;

  hp = camera_hists_->phys_trig_num_contiguous_channel_triggered_hist->dump_as_proto();
  partials_.mutable_camera()->mutable_phys_trig_num_contiguous_channel_triggered_hist()->IntegrateFrom(*hp);
  delete hp;

  hp = camera_hists_->muon_candidate_num_channel_triggered_hist->dump_as_proto();
  partials_.mutable_camera()->mutable_muon_candidate_num_channel_triggered_hist()->IntegrateFrom(*hp);
  delete hp;

  if(parent_)return true;

  for(int ichan = 0; ichan<partials_.channel_size(); ichan++) {
    auto& partials_chan = partials_.channel(ichan);
    integrate_one_gain_partials(results_.mutable_high_gain(), partials_chan.high_gain());
    if(has_dual_gain_) {
      integrate_one_gain_partials(results_.mutable_low_gain(), partials_chan.low_gain());
      integrate_dual_gain_partials(results_.mutable_dual_gain(), partials_chan.dual_gain());
    }
    if(partials_.camera().num_event_trigger_hitmap_found() > 0) {
      results_.add_channel_triggered_count(partials_chan.all_trig_num_events_triggered());
      results_.add_muon_candidate_channel_triggered_count(partials_chan.muon_candidate_num_events_triggered());
      results_.add_phy_trigger_few_neighbor_channel_triggered_count(partials_chan.phy_trig_few_neighbor_channel_triggered_count());
    }
  }
  integrate_one_gain_camera_partials(results_.mutable_high_gain(), partials_.camera().high_gain());
  if(has_dual_gain_) {
    integrate_one_gain_camera_partials(results_.mutable_low_gain(), partials_.camera().low_gain());
  }
  results_.mutable_num_channel_triggered_hist()->IntegrateFrom(
    partials_.camera().num_channel_triggered_hist());
  results_.mutable_num_contiguous_channel_triggered_hist()->IntegrateFrom(
    partials_.camera().num_contiguous_channel_triggered_hist());
  results_.mutable_phy_trigger_num_channel_triggered_hist()->IntegrateFrom(
    partials_.camera().phys_trig_num_channel_triggered_hist());
  results_.mutable_phy_trigger_num_contiguous_channel_triggered_hist()->IntegrateFrom(
    partials_.camera().phys_trig_num_contiguous_channel_triggered_hist());
  results_.mutable_muon_candidate_num_channel_triggered_hist()->IntegrateFrom(
    partials_.camera().muon_candidate_num_channel_triggered_hist());

  partials_.Clear();
  return true;
}

void SimpleChargeStatsParallelEventVisitor::record_one_gain_channel_data(
  const calin::ix::iact_data::telescope_event::TelescopeEvent* event,
  const calin::iact_data::waveform_treatment_event_visitor::OptimalWindowSumWaveformTreatmentParallelEventVisitor* sum_visitor,
  unsigned ichan, double elapsed_event_time,
  calin::ix::diagnostics::simple_charge_stats::PartialOneGainChannelSimpleChargeStats* one_gain_stats,
  SingleGainChannelHists* one_gain_hists,
  unsigned& nsum, int64_t& opt_sum, int64_t& sig_sum, int64_t& bkg_sum, int64_t& wf_sum,
  int wf_clipping_value)
{
  one_gain_stats->increment_all_trig_num_events();

  int64_t ped_win_sum = sum_visitor->array_chan_bkg_win_sum()[ichan];
  int64_t sqr_ped_win_sum = SQR(ped_win_sum);
  one_gain_stats->increment_all_trig_ped_win_sum(ped_win_sum);
  one_gain_stats->increment_all_trig_ped_win_sumsq(sqr_ped_win_sum);
  if(one_gain_hists) {
    one_gain_hists->all_pedwin_1_sum_vs_time->insert(elapsed_event_time);
    one_gain_hists->all_pedwin_q_sum_vs_time->insert(elapsed_event_time, ped_win_sum);
    one_gain_hists->all_pedwin_q2_sum_vs_time->insert(elapsed_event_time, sqr_ped_win_sum);
  }
  int64_t wf_all_sum = sum_visitor->array_chan_all_sum()[ichan];
  int64_t sig_win_sum = sum_visitor->array_chan_sig_win_sum()[ichan];
  int64_t bkg_win_sum = sum_visitor->array_chan_bkg_win_sum()[ichan];
  int64_t opt_win_sum = sum_visitor->array_chan_opt_win_sum()[ichan];
  if(event->trigger_type() == calin::ix::iact_data::telescope_event::TRIGGER_PEDESTAL) {
    int64_t sqr_wf_all_sum = SQR(wf_all_sum);
    one_gain_stats->increment_ped_trig_num_events();
    one_gain_stats->increment_ped_trig_full_wf_sum(wf_all_sum);
    one_gain_stats->increment_ped_trig_full_wf_sumsq(sqr_wf_all_sum);
    one_gain_stats->increment_ped_trig_ped_win_sum(ped_win_sum);
    one_gain_stats->increment_ped_trig_ped_win_sumsq(sqr_ped_win_sum);
    int64_t sqr_sig_win_sum = SQR(sig_win_sum);
    one_gain_stats->increment_ped_trig_sig_win_sum(sig_win_sum);
    one_gain_stats->increment_ped_trig_sig_win_sumsq(sqr_sig_win_sum);
    int64_t sqr_opt_win_sum = SQR(opt_win_sum);
    one_gain_stats->increment_ped_trig_opt_win_sum(opt_win_sum);
    one_gain_stats->increment_ped_trig_opt_win_sumsq(sqr_opt_win_sum);
    if(one_gain_hists) {
      one_gain_hists->ped_wf_1_sum_vs_time->insert(elapsed_event_time);
      one_gain_hists->ped_wf_q_sum_vs_time->insert(elapsed_event_time, wf_all_sum);
      one_gain_hists->ped_wf_q2_sum_vs_time->insert(elapsed_event_time, sqr_wf_all_sum);
    }
  } else if(event->trigger_type() == calin::ix::iact_data::telescope_event::TRIGGER_EXTERNAL_FLASHER) {
    int64_t sqr_sig_win_sum = SQR(sig_win_sum);
    one_gain_stats->increment_ext_trig_num_events();
    one_gain_stats->increment_ext_trig_sig_win_sum(sig_win_sum);
    one_gain_stats->increment_ext_trig_sig_win_sumsq(sqr_sig_win_sum);
    int64_t sqr_opt_win_sum = SQR(opt_win_sum);
    one_gain_stats->increment_ext_trig_opt_win_sum(opt_win_sum);
    one_gain_stats->increment_ext_trig_opt_win_sumsq(sqr_opt_win_sum);
  } else if(event->trigger_type() == calin::ix::iact_data::telescope_event::TRIGGER_PHYSICS) {
    one_gain_stats->increment_phys_trig_num_events();
  } else if(event->trigger_type() == calin::ix::iact_data::telescope_event::TRIGGER_INTERNAL_FLASHER) {
    one_gain_stats->increment_int_trig_num_events();
  }
  int chan_max = sum_visitor->array_chan_max()[ichan];
  if(chan_max >= wf_clipping_value) {
    one_gain_stats->increment_all_trig_num_wf_clipped();
    if(event->trigger_type() == calin::ix::iact_data::telescope_event::TRIGGER_PEDESTAL) {
      one_gain_stats->increment_ped_trig_num_wf_clipped();
    } else if(event->trigger_type() == calin::ix::iact_data::telescope_event::TRIGGER_EXTERNAL_FLASHER) {
      one_gain_stats->increment_ext_trig_num_wf_clipped();
    } else if(event->trigger_type() == calin::ix::iact_data::telescope_event::TRIGGER_PHYSICS) {
      one_gain_stats->increment_phys_trig_num_wf_clipped();
    } else if(event->trigger_type() == calin::ix::iact_data::telescope_event::TRIGGER_INTERNAL_FLASHER) {
      one_gain_stats->increment_int_trig_num_wf_clipped();
    }
  }
  ++nsum;
  opt_sum += opt_win_sum;
  sig_sum += sig_win_sum;
  bkg_sum += bkg_win_sum;
  wf_sum += wf_all_sum;
}

void SimpleChargeStatsParallelEventVisitor::record_dual_gain_channel_data(unsigned ichan,
  calin::ix::diagnostics::simple_charge_stats::PartialDualGainChannelSimpleChargeStats* dual_gain_stats,
  DualGainChannelHists* dual_gain_hists)
{
  if(dual_gain_hists) {
    int lg_max_sample = low_gain_visitor_->array_chan_max()[ichan];
    int hg_max_sample = high_gain_visitor_->array_chan_max()[ichan];
    int64_t hg_max_sample_sq = SQR(hg_max_sample);

    dual_gain_hists->all_max_sample_1->insert(lg_max_sample);
    dual_gain_hists->all_max_sample_q->insert(lg_max_sample, hg_max_sample);
    dual_gain_hists->all_max_sample_q2->insert(lg_max_sample, hg_max_sample_sq);

    int lg_opt_sum = low_gain_visitor_->array_chan_opt_win_sum()[ichan];
    int hg_opt_sum = high_gain_visitor_->array_chan_opt_win_sum()[ichan];
    int64_t hg_opt_sum_sq = SQR(hg_opt_sum);

    dual_gain_hists->all_opt_sum_1->insert(lg_opt_sum);
    dual_gain_hists->all_opt_sum_q->insert(lg_opt_sum, hg_opt_sum);
    dual_gain_hists->all_opt_sum_q2->insert(lg_opt_sum, hg_opt_sum_sq);
  }
}

void SimpleChargeStatsParallelEventVisitor::record_one_visitor_data(
  uint64_t seq_index, const calin::ix::iact_data::telescope_event::TelescopeEvent* event,
  const calin::iact_data::waveform_treatment_event_visitor::OptimalWindowSumWaveformTreatmentParallelEventVisitor* sum_visitor,
  calin::ix::diagnostics::simple_charge_stats::PartialSimpleChargeStats* partials)
{
  double elapsed_event_time = event->elapsed_event_time().time_ns() * 1e-9;

  if(sum_visitor and sum_visitor->is_same_event(seq_index)) {
    unsigned nsum_hg = 0;
    int64_t opt_sum_hg = 0;
    int64_t sig_sum_hg = 0;
    int64_t bkg_sum_hg = 0;
    int64_t all_sum_hg = 0;

    unsigned nsum_lg = 0;
    int64_t opt_sum_lg = 0;
    int64_t sig_sum_lg = 0;
    int64_t bkg_sum_lg = 0;
    int64_t all_sum_lg = 0;
    for(unsigned ichan=0; ichan<sum_visitor->nchan(); ichan++) {
      auto* pc = partials->mutable_channel(ichan);
      switch(sum_visitor->array_chan_signal_type()[ichan]) {
      case calin::ix::iact_data::telescope_event::SIGNAL_UNIQUE_GAIN:
      case calin::ix::iact_data::telescope_event::SIGNAL_HIGH_GAIN:
        record_one_gain_channel_data(event, sum_visitor, ichan, elapsed_event_time,
          pc->mutable_high_gain(), chan_hists_[ichan]->high_gain,
          nsum_hg, opt_sum_hg, sig_sum_hg, bkg_sum_hg, all_sum_hg, config_.high_gain_wf_clipping_value());
        break;
      case calin::ix::iact_data::telescope_event::SIGNAL_LOW_GAIN:
        record_one_gain_channel_data(event, sum_visitor, ichan, elapsed_event_time,
          pc->mutable_low_gain(), chan_hists_[ichan]->low_gain,
          nsum_lg, opt_sum_lg, sig_sum_lg, bkg_sum_lg, all_sum_lg, config_.low_gain_wf_clipping_value());
        break;
      case calin::ix::iact_data::telescope_event::SIGNAL_NONE:
      default:
        // do nothing
        break;
      }
    }
    if(nsum_hg == sum_visitor->nchan()) {
      camera_hists_->high_gain->all_pedwin_1_sum_vs_time->insert(elapsed_event_time);
      camera_hists_->high_gain->all_pedwin_q_sum_vs_time->insert(elapsed_event_time, bkg_sum_hg);
      camera_hists_->high_gain->all_pedwin_q2_sum_vs_time->insert(elapsed_event_time, bkg_sum_hg*bkg_sum_hg);
      if(event->trigger_type() == calin::ix::iact_data::telescope_event::TRIGGER_EXTERNAL_FLASHER) {
        auto* pcam = partials->mutable_camera()->mutable_high_gain();
        pcam->increment_ext_trig_all_num_events();
        pcam->increment_ext_trig_all_sig_win_sum(sig_sum_hg);
        pcam->increment_ext_trig_all_sig_win_sumsq(SQR(sig_sum_hg));
        pcam->increment_ext_trig_all_opt_win_sum(opt_sum_hg);
        pcam->increment_ext_trig_all_opt_win_sumsq(SQR(opt_sum_hg));
      } else if(event->trigger_type() == calin::ix::iact_data::telescope_event::TRIGGER_PEDESTAL) {
        camera_hists_->high_gain->ped_wf_1_sum_vs_time->insert(elapsed_event_time);
        camera_hists_->high_gain->ped_wf_q_sum_vs_time->insert(elapsed_event_time, all_sum_hg);
        camera_hists_->high_gain->ped_wf_q2_sum_vs_time->insert(elapsed_event_time, all_sum_hg*all_sum_hg);
      }
    }
    if(nsum_lg == sum_visitor->nchan()) {
      camera_hists_->low_gain->all_pedwin_1_sum_vs_time->insert(elapsed_event_time);
      camera_hists_->low_gain->all_pedwin_q_sum_vs_time->insert(elapsed_event_time, bkg_sum_lg);
      camera_hists_->low_gain->all_pedwin_q2_sum_vs_time->insert(elapsed_event_time, bkg_sum_lg*bkg_sum_lg);
      if(event->trigger_type() == calin::ix::iact_data::telescope_event::TRIGGER_EXTERNAL_FLASHER) {
        auto* pcam = partials->mutable_camera()->mutable_low_gain();
        pcam->increment_ext_trig_all_num_events();
        pcam->increment_ext_trig_all_sig_win_sum(sig_sum_lg);
        pcam->increment_ext_trig_all_sig_win_sumsq(SQR(sig_sum_lg));
        pcam->increment_ext_trig_all_opt_win_sum(opt_sum_lg);
        pcam->increment_ext_trig_all_opt_win_sumsq(SQR(opt_sum_lg));
      } else if(event->trigger_type() == calin::ix::iact_data::telescope_event::TRIGGER_PEDESTAL) {
        camera_hists_->low_gain->ped_wf_1_sum_vs_time->insert(elapsed_event_time);
        camera_hists_->low_gain->ped_wf_q_sum_vs_time->insert(elapsed_event_time, all_sum_lg);
        camera_hists_->low_gain->ped_wf_q2_sum_vs_time->insert(elapsed_event_time, all_sum_lg*all_sum_lg);
      }
    }
  }
}

void SimpleChargeStatsParallelEventVisitor::record_dual_visitor_data(uint64_t seq_index)
{
  if(high_gain_visitor_->is_same_event(seq_index) and
      low_gain_visitor_->is_same_event(seq_index)) {
    for(unsigned ichan=0; ichan<low_gain_visitor_->nchan(); ichan++) {
      auto* pc = partials_.mutable_channel(ichan);
      if(high_gain_visitor_->array_chan_signal_type()[ichan] == calin::ix::iact_data::telescope_event::SIGNAL_HIGH_GAIN and
          low_gain_visitor_->array_chan_signal_type()[ichan] == calin::ix::iact_data::telescope_event::SIGNAL_LOW_GAIN) {
        record_dual_gain_channel_data(ichan, pc->mutable_dual_gain(), chan_hists_[ichan]->dual_gain);
      }
    }
  }
}

bool SimpleChargeStatsParallelEventVisitor::visit_telescope_event(uint64_t seq_index,
  calin::ix::iact_data::telescope_event::TelescopeEvent* event)
{
  if(high_gain_visitor_) {
    record_one_visitor_data(seq_index, event, high_gain_visitor_, &partials_);
  }
  if(low_gain_visitor_) {
    record_one_visitor_data(seq_index, event, low_gain_visitor_, &partials_);
    if(high_gain_visitor_) {
      record_dual_visitor_data(seq_index);
    }
  }
  if(event->has_trigger_map()) {
    if(event->trigger_map().hit_channel_id_size()>0) {
      partials_.mutable_camera()->increment_num_event_trigger_hitmap_found();
    }
    for(auto ichan : event->trigger_map().hit_channel_id()) {
      partials_.mutable_channel(ichan)->increment_all_trig_num_events_triggered();
      partials_.mutable_channel(ichan)->increment_muon_candidate_num_events_triggered(
        event->is_muon_candidate());
    }
    camera_hists_->num_channel_triggered_hist->insert(event->trigger_map().hit_channel_id_size());
    int num_contiguous_channel_triggered = 0;
    if(event->trigger_map().hit_channel_id_size() > 0) {
      unsigned nisland = calin::iact_data::algorithms::find_channel_islands(
        *data_order_camera_, reinterpret_cast<const int*>(event->trigger_map().hit_channel_id().data()),
        event->trigger_map().hit_channel_id_size(), channel_island_id_.data(),
        channel_island_count_.data());
      num_contiguous_channel_triggered = channel_island_count_[0]; // guaranteed to have at least one island
      for(unsigned iisland=1; iisland<nisland; iisland++) {
        num_contiguous_channel_triggered = std::max(num_contiguous_channel_triggered, channel_island_count_[iisland]);
      }
    }
    camera_hists_->num_contiguous_channel_triggered_hist->insert(num_contiguous_channel_triggered);
    if(event->trigger_type() == calin::ix::iact_data::telescope_event::TRIGGER_PHYSICS) {
      camera_hists_->phys_trig_num_channel_triggered_hist->insert(event->trigger_map().hit_channel_id_size());
      camera_hists_->phys_trig_num_contiguous_channel_triggered_hist->insert(num_contiguous_channel_triggered);
      if(num_contiguous_channel_triggered < config_.nearest_neighbor_nchannel_threshold()) {
        for(auto ichan : event->trigger_map().hit_channel_id()) {
          partials_.mutable_channel(ichan)->increment_phy_trig_few_neighbor_channel_triggered_count();
        }
      }
    }
    if(event->is_muon_candidate()) {
      camera_hists_->muon_candidate_num_channel_triggered_hist->insert(event->trigger_map().hit_channel_id_size());
    }
  }
  merge_time_histograms_if_necessary();
  return true;
}

bool SimpleChargeStatsParallelEventVisitor::merge_results()
{
  if(parent_) {
    parent_->partials_.IntegrateFrom(partials_);
  }
  return true;
}

calin::ix::diagnostics::simple_charge_stats::SimpleChargeStats*
SimpleChargeStatsParallelEventVisitor::simple_charge_stats(
  calin::ix::diagnostics::simple_charge_stats::SimpleChargeStats* stats) const
{
  if(stats == nullptr)stats = results_.New();
  stats->CopyFrom(results_);
  return stats;
}

calin::ix::diagnostics::simple_charge_stats::SimpleChargeStatsConfig
SimpleChargeStatsParallelEventVisitor::default_config()
{
  calin::ix::diagnostics::simple_charge_stats::SimpleChargeStatsConfig config;
  config.set_high_gain_wf_clipping_value(4095);
  config.set_low_gain_wf_clipping_value(4095);
  config.set_nearest_neighbor_nchannel_threshold(3);
  config.set_ped_time_hist_resolution(5.0);
  config.set_channel_ped_time_hist_range(86400.0);
  config.set_camera_ped_time_hist_range(86400.0);
  config.set_dual_gain_sample_resolution(1.0);
  config.set_dual_gain_sum_resolution(5.0);
  return config;
}
