/*

   calin/diagnostics/stage1.hpp -- Stephen Fegan -- 2020-03-28

   Stage 1 analysis - calculate distributions and perform some low-level
                      diagnostics on events

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

#include <util/log.hpp>
#include <util/file.hpp>
#include <util/timestamp.hpp>
#include <io/json.hpp>
#include <diagnostics/stage1.hpp>
#include <provenance/anthology.hpp>
#include <diagnostics/waveform_psd_vcl.hpp>

using namespace calin::util::log;
using namespace calin::diagnostics::stage1;

namespace {
  char LOGO[] =
    "  ____                ___       ____  \n"
    " / / /    _________ _/ (_)___   \\ \\ \\ \n" \
    "/ / /    / ___/ __ `/ / / __ \\   \\ \\ \\   __              \n" \
    "\\ \\ \\   / /__/ /_/ / / / / / /   / / /  (__|_ _. _  _  /|\n" \
    " \\_\\_\\  \\___/\\__,_/_/_/_/ /_/   /_/_/   __)|_(_|(_|(/_  |\n" \
    "                                                ._|      \n";
}

Stage1ParallelEventVisitor::Stage1ParallelEventVisitor(const calin::ix::diagnostics::stage1::Stage1Config& config):
  FilteredDelegatingParallelEventVisitor(), config_(config)
{
  hg_sum_pev_ = calin::iact_data::waveform_treatment_event_visitor::
    OptimalWindowSumWaveformTreatmentParallelEventVisitor::New(config_.high_gain_opt_sum());

  if(config_.has_low_gain_opt_sum()) {
    lg_sum_pev_ = calin::iact_data::waveform_treatment_event_visitor::
      OptimalWindowSumWaveformTreatmentParallelEventVisitor::New(config_.low_gain_opt_sum(),
        calin::iact_data::waveform_treatment_event_visitor::
        OptimalWindowSumWaveformTreatmentParallelEventVisitor::LOW_GAIN);
  } else {
    lg_sum_pev_ = calin::iact_data::waveform_treatment_event_visitor::
      OptimalWindowSumWaveformTreatmentParallelEventVisitor::New(config_.high_gain_opt_sum(),
        calin::iact_data::waveform_treatment_event_visitor::
        OptimalWindowSumWaveformTreatmentParallelEventVisitor::LOW_GAIN);
  }

  run_info_pev_ = new calin::diagnostics::run_info::RunInfoDiagnosticsParallelEventVisitor(
    config_.run_info());

  charge_stats_pev_ = new calin::diagnostics::simple_charge_stats::
    SimpleChargeStatsParallelEventVisitor(hg_sum_pev_, lg_sum_pev_, config_.simple_charge_stats());

  this->add_visitor(hg_sum_pev_);
  this->add_visitor(lg_sum_pev_);
  this->add_visitor(run_info_pev_);
  this->add_visitor(charge_stats_pev_);

  if(config_.enable_mean_waveform()) {
    wf_mean_phy_pev_ = new calin::diagnostics::waveform::WaveformSumParallelEventVisitor(config.calculate_waveform_variance());
    wf_mean_ped_pev_ = new calin::diagnostics::waveform::WaveformSumParallelEventVisitor(config.calculate_waveform_variance());
    wf_mean_ext_pev_ = new calin::diagnostics::waveform::WaveformSumParallelEventVisitor(config.calculate_waveform_variance());
    wf_mean_int_pev_ = new calin::diagnostics::waveform::WaveformSumParallelEventVisitor(config.calculate_waveform_variance());

    this->add_physics_trigger_visitor(wf_mean_phy_pev_, "Physics triggers");
    this->add_pedestal_trigger_visitor(wf_mean_ped_pev_, "Pedestal triggers");
    this->add_external_flasher_trigger_visitor(wf_mean_ext_pev_, "External-flasher triggers");
    this->add_internal_flasher_trigger_visitor(wf_mean_int_pev_, "Internal-flasher triggers");
  }

  if(config_.enable_simple_waveform_hists()) {
    charge_hists_phy_pev_ = new calin::diagnostics::simple_charge_hists::
      SimpleChargeHistsParallelEventVisitor(hg_sum_pev_, lg_sum_pev_,
        config_.phy_trigger_waveform_hists());
    charge_hists_ped_pev_ = new calin::diagnostics::simple_charge_hists::
      SimpleChargeHistsParallelEventVisitor(hg_sum_pev_, lg_sum_pev_,
        config_.ped_trigger_waveform_hists());
    charge_hists_ext_pev_ = new calin::diagnostics::simple_charge_hists::
      SimpleChargeHistsParallelEventVisitor(hg_sum_pev_, lg_sum_pev_,
        config_.ext_trigger_waveform_hists());
    charge_hists_int_pev_ = new calin::diagnostics::simple_charge_hists::
      SimpleChargeHistsParallelEventVisitor(hg_sum_pev_, lg_sum_pev_,
        config_.int_trigger_waveform_hists());

    this->add_physics_trigger_visitor(charge_hists_phy_pev_, "Physics triggers");
    this->add_pedestal_trigger_visitor(charge_hists_ped_pev_, "Pedestal triggers");
    this->add_external_flasher_trigger_visitor(charge_hists_ext_pev_, "External-flasher triggers");
    this->add_internal_flasher_trigger_visitor(charge_hists_int_pev_, "Internal-flasher triggers");
  }

  if(config_.enable_l0_trigger_bit_waveform_hists()) {
    charge_hists_trig_bit_set_pev_ = new calin::diagnostics::simple_charge_hists::
      SimpleChargeHistsParallelEventVisitor(hg_sum_pev_, nullptr,
        config_.l0_trigger_bit_waveform_hists(),
        new calin::diagnostics::simple_charge_hists::SimpleChargeHistsTriggerBitFilter(
          /*trigger_bit_status_required_for_accept=*/true),/*adopt_filter=*/true);
    charge_hists_trig_bit_clr_pev_ = new calin::diagnostics::simple_charge_hists::
      SimpleChargeHistsParallelEventVisitor(hg_sum_pev_, nullptr,
        config_.l0_trigger_bit_waveform_hists(),
        new calin::diagnostics::simple_charge_hists::SimpleChargeHistsTriggerBitFilter(
          /*trigger_bit_status_required_for_accept=*/false),/*adopt_filter=*/true);
    this->add_visitor(charge_hists_trig_bit_set_pev_, "L0 trigger-bit set");
    this->add_visitor(charge_hists_trig_bit_clr_pev_, "L0 trigger-bit clear");
  }

  if(config_.enable_all_waveform_psd()) {
    wf_psd_phy_pev_ = calin::diagnostics::waveform::WaveformPSDParallelVisitor::New();
    this->add_physics_trigger_visitor(wf_psd_phy_pev_, "Physics triggers");
    wf_psd_ped_pev_ = calin::diagnostics::waveform::WaveformPSDParallelVisitor::New();
    this->add_pedestal_trigger_visitor(wf_psd_ped_pev_, "Pedestal triggers");
    wf_psd_ext_pev_ = calin::diagnostics::waveform::WaveformPSDParallelVisitor::New();
    this->add_external_flasher_trigger_visitor(wf_psd_ext_pev_, "External-flasher triggers");
    wf_psd_int_pev_ = calin::diagnostics::waveform::WaveformPSDParallelVisitor::New();
    this->add_internal_flasher_trigger_visitor(wf_psd_int_pev_, "Internal-flasher triggers");
  } else if(config_.enable_pedestal_waveform_psd()) {
    wf_psd_ped_pev_ = calin::diagnostics::waveform::WaveformPSDParallelVisitor::New();
    this->add_pedestal_trigger_visitor(wf_psd_ped_pev_, "Pedestal triggers");
  }

  if(config_.enable_clock_regression()) {
    clock_regression_pev_ = new calin::diagnostics::clock_regression::
      ClockRegressionParallelEventVisitor(config_.clock_regression());
    this->add_visitor(clock_regression_pev_);
  }

  if(config_.enable_write_reduced_event_file()) {
    reduced_event_writer_pev_ = new calin::diagnostics::reduced_event_writer::
      ReducedEventWriterParallelEventVisitor(hg_sum_pev_, lg_sum_pev_, config_.reduced_event_writer());
    this->add_visitor(reduced_event_writer_pev_);
  }
}

Stage1ParallelEventVisitor::~Stage1ParallelEventVisitor()
{
  delete charge_hists_phy_pev_;
  delete charge_hists_ped_pev_;
  delete charge_hists_ext_pev_;
  delete charge_hists_int_pev_;
  delete charge_hists_trig_bit_set_pev_;
  delete charge_hists_trig_bit_clr_pev_;
  delete wf_mean_int_pev_;
  delete wf_mean_ext_pev_;
  delete wf_mean_ped_pev_;
  delete wf_mean_phy_pev_;
  delete charge_stats_pev_;
  delete run_info_pev_;
  delete lg_sum_pev_;
  delete hg_sum_pev_;
  delete clock_regression_pev_;
  delete wf_psd_phy_pev_;
  delete wf_psd_ped_pev_;
  delete wf_psd_ext_pev_;
  delete wf_psd_int_pev_;
  delete nectarcam_ancillary_data_;
  delete reduced_event_writer_pev_;
}

bool Stage1ParallelEventVisitor::visit_telescope_run(
  const calin::ix::iact_data::telescope_run_configuration::TelescopeRunConfiguration* run_config,
  calin::iact_data::event_visitor::EventLifetimeManager* event_lifetime_manager,
  calin::ix::provenance::chronicle::ProcessingRecord* processing_record)
{
  LOG(INFO) << LOGO;
  delete nectarcam_ancillary_data_;
  nectarcam_ancillary_data_ = nullptr;
  run_config_ = run_config;
  if(processing_record) {
    processing_record->set_type("Stage1ParallelEventVisitor");
    processing_record->set_description("Stage 1 data reduction");
    auto* config_json = processing_record->add_config();
    config_json->set_type(config_.GetTypeName());
    config_json->set_json(calin::io::json::encode_protobuf_to_json_string(config_));
  }
  return FilteredDelegatingParallelEventVisitor::visit_telescope_run(
    run_config, event_lifetime_manager, processing_record);
}

bool Stage1ParallelEventVisitor::leave_telescope_run(
  calin::ix::provenance::chronicle::ProcessingRecord* processing_record)
{
  bool good = FilteredDelegatingParallelEventVisitor::leave_telescope_run(processing_record);

  if(config_.enable_ancillary_data()) {
    int64_t start_time = run_info_pev_->min_event_time() / int64_t(1000000000);
    int64_t end_time = run_info_pev_->max_event_time() / int64_t(1000000000);
    if(start_time > end_time) {
      start_time = run_config_->run_start_time().time_ns() / int64_t(1000000000);
      end_time = start_time + 3600; // default to getting 1 hour of data
    } else {
      start_time = std::min(start_time, run_config_->run_start_time().time_ns() / int64_t(1000000000));
    }
    start_time = std::max(start_time-30, int64_t(0));
    end_time = end_time+30;

    std::string db_filename;

    switch(run_config_->camera_layout().camera_type()) {
      case calin::ix::iact_data::instrument_layout::CameraLayout::NECTARCAM:
      case calin::ix::iact_data::instrument_layout::CameraLayout::NECTARCAM_TESTBENCH_19CHANNEL:
      case calin::ix::iact_data::instrument_layout::CameraLayout::NECTARCAM_TESTBENCH_61CHANNEL:
        db_filename = nectarcam_ancillary_database_filename(run_config_->filename(), 
          run_config_->run_start_time().time_ns(), config_.ancillary_database(),
          config_.ancillary_database_directory());
        if(calin::util::file::is_file(db_filename)) {
          nectarcam_ancillary_data_ =
            calin::iact_data::nectarcam_ancillary_data::
              retrieve_nectarcam_ancillary_data(db_filename, run_config_->telescope_id(),
                start_time, end_time);
        } else {
          LOG(WARNING) << "NectarCAM ancillary database not found: " << db_filename;
        }
        break;
      case calin::ix::iact_data::instrument_layout::CameraLayout::LSTCAM:
        break;
      default:
        break;
    }
  }

  return good;
}

std::string Stage1ParallelEventVisitor::
nectarcam_ancillary_database_filename(const std::string run_filename, uint64_t run_start_time_ns, 
  const std::string forced_filename, const std::string forced_directory)
{
  std::string db_filename = forced_filename;
  if(db_filename.empty()) {
    db_filename = forced_directory;
    if(db_filename.empty()) {
      db_filename = calin::util::file::dirname(run_filename);
    }
    std::string utdate = calin::util::file::basename(calin::util::file::dirname(run_filename));
    db_filename += "/nectarcam_monitoring_db_";
    if(utdate.size() == 8) {
      db_filename += utdate.substr(0,4);
      db_filename += "-";
      db_filename += utdate.substr(4,2);
      db_filename += "-";
      db_filename += utdate.substr(6,2);
    } else {
      db_filename += calin::util::timestamp::
        Timestamp(run_start_time_ns).as_string().substr(0,10);
    }
    db_filename += ".sqlite";
  }
  return db_filename;
}

calin::ix::diagnostics::stage1::Stage1* Stage1ParallelEventVisitor::stage1_results(
  calin::ix::diagnostics::stage1::Stage1* stage1) const
{
  if(stage1 == nullptr)stage1 = new calin::ix::diagnostics::stage1::Stage1;

  run_info_pev_->run_config(stage1->mutable_run_config());
  run_info_pev_->run_info(stage1->mutable_run_info());
  charge_stats_pev_->simple_charge_stats(stage1->mutable_charge_stats());

  if(wf_mean_phy_pev_ and this->visitor_saw_event(wf_mean_phy_pev_)) {
    wf_mean_phy_pev_->mean_waveforms(stage1->mutable_mean_wf_physics());
  }
  if(wf_mean_ped_pev_ and this->visitor_saw_event(wf_mean_ped_pev_)) {
    wf_mean_ped_pev_->mean_waveforms(stage1->mutable_mean_wf_pedestal());
  }
  if(wf_mean_ext_pev_ and this->visitor_saw_event(wf_mean_ext_pev_)) {
    wf_mean_ext_pev_->mean_waveforms(stage1->mutable_mean_wf_external_flasher());
  }
  if(wf_mean_int_pev_ and this->visitor_saw_event(wf_mean_int_pev_)) {
    wf_mean_int_pev_->mean_waveforms(stage1->mutable_mean_wf_internal_flasher());
  }

  if(charge_hists_phy_pev_ and this->visitor_saw_event(charge_hists_phy_pev_)) {
    charge_hists_phy_pev_->simple_charge_hists(stage1->mutable_wf_hists_physics());
  }
  if(charge_hists_ped_pev_ and this->visitor_saw_event(charge_hists_ped_pev_)) {
    charge_hists_ped_pev_->simple_charge_hists(stage1->mutable_wf_hists_pedestal());
  }
  if(charge_hists_ext_pev_ and this->visitor_saw_event(charge_hists_ext_pev_)) {
    charge_hists_ext_pev_->simple_charge_hists(stage1->mutable_wf_hists_external_flasher());
  }
  if(charge_hists_int_pev_ and this->visitor_saw_event(charge_hists_int_pev_)) {
    charge_hists_int_pev_->simple_charge_hists(stage1->mutable_wf_hists_internal_flasher());
  }

  if(charge_hists_trig_bit_set_pev_ and this->visitor_saw_event(charge_hists_trig_bit_set_pev_)) {
    charge_hists_trig_bit_set_pev_->simple_charge_hists(stage1->mutable_wf_hists_l0_trigger_bit_set());
  }
  if(charge_hists_trig_bit_clr_pev_ and this->visitor_saw_event(charge_hists_trig_bit_clr_pev_)) {
    charge_hists_trig_bit_clr_pev_->simple_charge_hists(stage1->mutable_wf_hists_l0_trigger_bit_clear());
  }

  if(clock_regression_pev_) {
    clock_regression_pev_->clock_regression(stage1->mutable_clock_regression());
  }

  if(wf_psd_phy_pev_ and this->visitor_saw_event(wf_psd_phy_pev_)) {
    wf_psd_phy_pev_->psd(stage1->mutable_psd_wf_physics());
  }
  if(wf_psd_ped_pev_ and this->visitor_saw_event(wf_psd_ped_pev_)) {
    wf_psd_ped_pev_->psd(stage1->mutable_psd_wf_pedestal());
  }
  if(wf_psd_ext_pev_ and this->visitor_saw_event(wf_psd_ext_pev_)) {
    wf_psd_ext_pev_->psd(stage1->mutable_psd_wf_external_flasher());
  }
  if(wf_psd_int_pev_ and this->visitor_saw_event(wf_psd_int_pev_)) {
    wf_psd_int_pev_->psd(stage1->mutable_psd_wf_internal_flasher());
  }

  stage1->set_run_number(stage1->run_config().run_number());
  stage1->set_run_start_time(stage1->run_config().run_start_time().time_ns());
  stage1->set_run_start_time_string(
    calin::util::timestamp::Timestamp(stage1->run_config().run_start_time().time_ns()).as_string());
  stage1->set_telescope_id(stage1->run_config().telescope_id());
  stage1->set_filename(stage1->run_config().filename());
  stage1->set_run_duration(
    std::max(stage1->run_info().max_event_time()-stage1->run_info().min_event_time(), int64_t(-1)));
  stage1->set_run_duration_sec((stage1->run_duration()==-1) ? -1.0: double(stage1->run_duration())*1e-9);
  stage1->set_num_events_found(stage1->run_info().num_events_found());

  stage1->set_num_physics_triggers(stage1->run_info().num_mono_trigger());
  stage1->set_num_pedestal_triggers(stage1->run_info().num_pedestal_trigger());
  stage1->set_num_external_calibration_triggers(stage1->run_info().num_external_calibration_trigger());
  stage1->set_num_internal_calibration_trigger(stage1->run_info().num_internal_calibration_trigger());

  stage1->mutable_config()->CopyFrom(config_);
  calin::provenance::anthology::get_current_anthology(stage1->mutable_provenance_anthology());

  if(nectarcam_ancillary_data_) {
    stage1->mutable_nectarcam()->mutable_ancillary_data()->CopyFrom(*nectarcam_ancillary_data_);
  }

  return stage1;
}

calin::ix::diagnostics::stage1::Stage1Config Stage1ParallelEventVisitor::default_config()
{
  calin::ix::diagnostics::stage1::Stage1Config cfg;
  cfg.set_enable_mean_waveform(true);
  cfg.set_enable_simple_waveform_hists(true);
  cfg.set_enable_l0_trigger_bit_waveform_hists(true);
  cfg.set_enable_ancillary_data(true);
  cfg.set_enable_clock_regression(true);
  cfg.set_enable_pedestal_waveform_psd(true);
  cfg.set_enable_all_waveform_psd(false);
  cfg.set_enable_write_reduced_event_file(false);

  cfg.mutable_high_gain_opt_sum()->CopyFrom(
    calin::iact_data::waveform_treatment_event_visitor::OptimalWindowSumWaveformTreatmentParallelEventVisitor::default_config());
  // We don't set "cfg.mutable_low_gain_opt_sum" which forces its config to come from high-gain

  cfg.mutable_run_info()->CopyFrom(
    calin::diagnostics::run_info::RunInfoDiagnosticsParallelEventVisitor::default_config());

  cfg.mutable_simple_charge_stats()->CopyFrom(
    calin::diagnostics::simple_charge_stats::SimpleChargeStatsParallelEventVisitor::default_config());

  cfg.mutable_phy_trigger_waveform_hists()->CopyFrom(
    calin::diagnostics::simple_charge_hists::SimpleChargeHistsParallelEventVisitor::phy_trig_default_config());

  cfg.mutable_ped_trigger_waveform_hists()->CopyFrom(
    calin::diagnostics::simple_charge_hists::SimpleChargeHistsParallelEventVisitor::ped_trig_default_config());

  cfg.mutable_ext_trigger_waveform_hists()->CopyFrom(
    calin::diagnostics::simple_charge_hists::SimpleChargeHistsParallelEventVisitor::ext_trig_default_config());

  cfg.mutable_int_trigger_waveform_hists()->CopyFrom(
    calin::diagnostics::simple_charge_hists::SimpleChargeHistsParallelEventVisitor::int_trig_default_config());

  cfg.mutable_l0_trigger_bit_waveform_hists()->CopyFrom(
    calin::diagnostics::simple_charge_hists::SimpleChargeHistsParallelEventVisitor::l0_trig_bits_default_config());

  cfg.mutable_clock_regression()->CopyFrom(
    calin::diagnostics::clock_regression::ClockRegressionParallelEventVisitor::default_config());

  cfg.mutable_reduced_event_writer()->CopyFrom(
    calin::diagnostics::reduced_event_writer::ReducedEventWriterParallelEventVisitor::default_config());

  return cfg;
}
