/*

   calin/iact_data/parallel_event_dispatcher.cpp -- Stephen Fegan -- 2018-02-08

   A parallel-only dispatcher of run and event data

   Copyright 2018, Stephen Fegan <sfegan@llr.in2p3.fr>
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

#include <chrono>
#include <type_traits>
#include <iomanip>

#include <util/string.hpp>
#include <util/log.hpp>
#include <util/timestamp.hpp>
#include <iact_data/zfits_actl_data_source.hpp>
#include <iact_data/cta_actl_event_decoder.hpp>
#include <iact_data/parallel_event_dispatcher.hpp>
#include <io/data_source.hpp>
#include <io/buffered_data_source.hpp>
#include <util/file.hpp>

using namespace calin::util::string;
using namespace calin::util::log;
using calin::util::timestamp::Timestamp;
using namespace calin::iact_data::event_dispatcher;
using namespace calin::io::data_source;
using namespace calin::iact_data::telescope_data_source;
using namespace calin::iact_data::cta_data_source;
using namespace calin::ix::iact_data::telescope_event;
using namespace calin::ix::iact_data::telescope_run_configuration;
using calin::iact_data::event_visitor::ParallelEventVisitor;

ParallelEventDispatcher::ParallelEventDispatcher()
{
  // nothing to see here
}

ParallelEventDispatcher::~ParallelEventDispatcher()
{
  for(auto ivisitor : adopted_visitors_)delete ivisitor;

}

void ParallelEventDispatcher::add_event_to_keep(
  calin::ix::iact_data::telescope_event::TelescopeEvent* event,
  uint64_t seq_index, google::protobuf::Arena* arena)
{
  if(event_keep_.find(event) != event_keep_.end())
    throw std::logic_error("add_event_to_keep: event already on managed event list");
  managed_event me { 0, seq_index, arena };
  event_keep_.emplace(event, me);
}

void ParallelEventDispatcher::keep_event(
  calin::ix::iact_data::telescope_event::TelescopeEvent* event)
{
  auto ifind = event_keep_.find(event);
  if(ifind == event_keep_.end())
    throw std::logic_error("keep_event: event not on managed event list");
  ifind->second.usage_count++;
  // std::cerr << "Keep: " << event << " count " << ifind->second.usage_count << '\n';
}

void ParallelEventDispatcher::release_event(
  calin::ix::iact_data::telescope_event::TelescopeEvent* event)
{
  auto ifind = event_keep_.find(event);
  if(ifind == event_keep_.end())
    throw std::logic_error("release_event: event not on managed event list");
  ifind->second.usage_count--;
  // std::cerr << "Release: " << event << " count " << ifind->second.usage_count << '\n';
  if(ifind->second.usage_count == 0) {
    // std::cerr << "Freeing: " << event << " arena: " << ifind->second.arena
    //   << " seq: " << ifind->second.seq_index << '\n';
    if(ifind->second.arena)delete ifind->second.arena;
    else delete event;
    event_keep_.erase(ifind);
  }
}

void ParallelEventDispatcher::
add_visitor(ParallelEventVisitor* visitor,
  const std::string& processing_record_comment, bool adopt_visitor)
{
  visitors_.emplace_back(visitor, processing_record_comment);
  if(adopt_visitor)adopted_visitors_.emplace_back(visitor);
}

void ParallelEventDispatcher::
process_run(TelescopeRandomAccessDataSourceWithRunConfig* src,
  unsigned log_frequency, int nthread)
{
  TelescopeRunConfiguration* run_config = src->get_run_configuration();
  process_run(src, run_config, log_frequency, nthread);
  delete run_config;
}

void ParallelEventDispatcher::
process_run(std::vector<calin::iact_data::telescope_data_source::
  TelescopeRandomAccessDataSourceWithRunConfig*> src_list,
  unsigned log_frequency)
{
  if(src_list.empty())
    throw std::runtime_error("process_run: empty data source list");
  TelescopeRunConfiguration* run_config = src_list[0]->get_run_configuration();
  for(unsigned isrc=1; isrc<src_list.size(); isrc++) {
    TelescopeRunConfiguration* from_run_config =
      src_list[isrc]->get_run_configuration();
    merge_run_config(run_config, *from_run_config);
    delete from_run_config;
  }
  std::vector<calin::io::data_source::DataSource<
    calin::ix::iact_data::telescope_event::TelescopeEvent>*> src_list_upcast;
  for(auto* src: src_list)src_list_upcast.push_back(src);
  this->process_run(src_list_upcast, run_config, log_frequency);
  delete run_config;
}

void ParallelEventDispatcher::
process_run(std::vector<calin::iact_data::telescope_data_source::
  TelescopeDataSourceWithRunConfig*> src_list, unsigned log_frequency)
{
  if(src_list.empty())
    throw std::runtime_error("process_run: empty data source list");
  TelescopeRunConfiguration* run_config = src_list[0]->get_run_configuration();
  for(unsigned isrc=1; isrc<src_list.size(); isrc++) {
    TelescopeRunConfiguration* from_run_config =
      src_list[isrc]->get_run_configuration();
    merge_run_config(run_config, *from_run_config);
    delete from_run_config;
  }
  std::vector<calin::io::data_source::DataSource<
    calin::ix::iact_data::telescope_event::TelescopeEvent>*> src_list_upcast;
  for(auto* src: src_list)src_list_upcast.push_back(src);
  this->process_run(src_list_upcast, run_config, log_frequency);
  delete run_config;
}

void ParallelEventDispatcher::process_run(calin::io::data_source::DataSource<
    calin::ix::iact_data::telescope_event::TelescopeEvent>* src,
  calin::ix::iact_data::
    telescope_run_configuration::TelescopeRunConfiguration* run_config,
  unsigned log_frequency, int nthread)
{
  auto start_time = std::chrono::system_clock::now();
  std::atomic<uint_fast64_t> ndispatched { 0 };

  write_initial_log_message(run_config, nthread);
  dispatch_run_configuration(run_config, /*register_processor=*/ true);
  if(nthread <= 0)
  {
    do_dispatcher_loop(src, log_frequency, start_time, ndispatched);
  }
  else
  {
    io::data_source::UnidirectionalBufferedDataSourcePump<TelescopeEvent> pump(src);
    do_parallel_dispatcher_loops(run_config, &pump, nthread, log_frequency,
      start_time, ndispatched);
  }
  dispatch_leave_run();
  write_final_log_message(log_frequency, start_time, ndispatched);
}

void ParallelEventDispatcher::
process_run(std::vector<calin::io::data_source::DataSource<
    calin::ix::iact_data::telescope_event::TelescopeEvent>*> src_list,
  calin::ix::iact_data::
    telescope_run_configuration::TelescopeRunConfiguration* run_config,
  unsigned log_frequency)
{
  if(src_list.empty())
    throw std::runtime_error("process_run: empty data source list");

  auto start_time = std::chrono::system_clock::now();
  std::atomic<uint_fast64_t> ndispatched { 0 };

  write_initial_log_message(run_config, src_list.size());
  dispatch_run_configuration(run_config, /*register_processor=*/ true);
  if(src_list.size() == 1)
  {
    do_dispatcher_loop(src_list[0], log_frequency, start_time, ndispatched);
  }
  else
  {
    io::data_source::VectorDataSourceFactory<TelescopeEvent> src_factory(src_list);
    do_parallel_dispatcher_loops(run_config, &src_factory, src_list.size(),
      log_frequency, start_time, ndispatched);
  }
  dispatch_leave_run();
  write_final_log_message(log_frequency, start_time, ndispatched);
}

#ifdef CALIN_HAVE_CTA_CAMERASTOACTL
void ParallelEventDispatcher::
process_cta_zfits_run(const std::string& filename,
  const calin::ix::iact_data::event_dispatcher::EventDispatcherConfig& config)
{
  auto zfits_config = config.zfits();
  auto* cta_file = new CTAZFITSDataSource(filename, config.decoder(), zfits_config);
  TelescopeRunConfiguration* run_config = cta_file->get_run_configuration();

  auto fragments = cta_file->all_fragment_names();
  if(fragments.empty()) {
    // This should never happen (I guess) as we should already have had an exception
    throw std::runtime_error("process_cta_zfits_run: file not found: " + filename);
  }

  unsigned nthread = std::min(std::max(config.nthread(), 1U), unsigned(fragments.size()));

  if(nthread == 1) {
    try {
      process_run(cta_file, run_config, config.log_frequency());
    } catch(...) {
      delete cta_file;
      delete run_config;
      throw;
    }
  } else {
    zfits_config.set_file_fragment_stride(
      nthread*std::max(1U, zfits_config.file_fragment_stride()));

    std::vector<calin::iact_data::telescope_data_source::
      TelescopeDataSource*> src_list(nthread);
    try {
      for(unsigned ithread=0; ithread<nthread; ithread++) {
        src_list[ithread] =
          new CTAZFITSDataSource(fragments[ithread], cta_file, zfits_config);
      }
      delete cta_file;
      cta_file = nullptr;
      process_run(src_list, run_config, config.log_frequency());
    } catch(...) {
      for(auto* src: src_list)delete src;
      delete cta_file;
      delete run_config;
      throw;
    }
    for(auto* src: src_list)delete src;
  }
  delete run_config;
}

void ParallelEventDispatcher::
process_cta_zmq_run(const std::vector<std::string>& endpoints,
  const calin::ix::iact_data::event_dispatcher::EventDispatcherConfig& config)
{
  if(endpoints.empty())
    throw std::runtime_error("process_run: empty endpoints list");
  std::vector<calin::iact_data::telescope_data_source::
    TelescopeDataSourceWithRunConfig*> src_list;
  try {
    for(const auto& endpoint: endpoints) {
      for(unsigned ithread=0; ithread<std::max(config.nthread(),1U); ++ithread) {
        auto* rsrc = new calin::iact_data::zfits_actl_data_source::
          ZMQACTL_R1_CameraEventDataSource(endpoint, config.zmq());
        auto* decoder = new calin::iact_data::cta_actl_event_decoder::
          CTA_ACTL_R1_CameraEventDecoder(endpoint, config.run_number(), config.decoder());
        auto* src = new calin::iact_data::actl_event_decoder::
          DecodedACTL_R1_CameraEventDataSourceWithRunConfig(rsrc, decoder,
            /* adopt_actl_src = */ false, /* adopt_decoder = */ true);
        src_list.emplace_back(src);
      }
    }
    process_run(src_list, config.log_frequency());
  } catch(...) {
    for(auto* src: src_list)delete src;
    throw;
  }
  for(auto* src: src_list)delete src;
}

void ParallelEventDispatcher::
process_cta_zmq_run(const std::string& endpoint,
  const calin::ix::iact_data::event_dispatcher::EventDispatcherConfig& config)
{
  std::vector<std::string> endpoints;
  endpoints.emplace_back(endpoint);
  process_cta_zmq_run(endpoints, config);
}
#endif // defined(CALIN_HAVE_CTA_CAMERASTOACTL)

calin::ix::iact_data::event_dispatcher::EventDispatcherConfig
ParallelEventDispatcher::default_config()
{
  calin::ix::iact_data::event_dispatcher::EventDispatcherConfig config;
  config.set_log_frequency(10000);
  config.set_nthread(1);
  config.set_run_number(0);
#ifdef CALIN_HAVE_CTA_CAMERASTOACTL
  config.mutable_decoder()->CopyFrom(
    calin::iact_data::cta_actl_event_decoder::CTA_ACTL_R1_CameraEventDecoder::default_config());
  config.mutable_zfits()->CopyFrom(
    calin::iact_data::cta_data_source::CTAZFITSDataSource::default_config());
  config.mutable_zmq()->CopyFrom(
    calin::iact_data::zfits_actl_data_source::ZMQACTL_R1_CameraEventDataSource::default_config());
#endif // defined(CALIN_HAVE_CTA_CAMERASTOACTL)
  return config;
}

void ParallelEventDispatcher::
dispatch_event(uint64_t seq_index, TelescopeEvent* event)
{
  for(const auto& iv : visitors_) {
    iv.visitor->visit_telescope_event(seq_index, event);
  }
}

void ParallelEventDispatcher::do_parallel_dispatcher_loops(
  calin::ix::iact_data::
    telescope_run_configuration::TelescopeRunConfiguration* run_config,
  calin::io::data_source::DataSourceFactory<
    calin::ix::iact_data::telescope_event::TelescopeEvent>* src_factory,
  unsigned nthread, unsigned log_frequency,
  const std::chrono::system_clock::time_point& start_time,
  std::atomic<uint_fast64_t>& ndispatched)
{
  std::vector<ParallelEventDispatcher*> sub_dispatchers;
  for(unsigned ithread=0;ithread<nthread;ithread++)
  {
    auto* d = new ParallelEventDispatcher;
    sub_dispatchers.emplace_back(d);

    std::map<ParallelEventVisitor*,ParallelEventVisitor*>
      antecedent_visitors;
    for(const auto& iv : visitors_) {
      ParallelEventVisitor* sv = iv.visitor->new_sub_visitor(antecedent_visitors);
      if(sv != nullptr) {
        d->add_visitor(sv, true);
      }
      antecedent_visitors[iv.visitor] = sv;
    }
    d->dispatch_run_configuration(run_config, /*register_processor=*/ false);
  }

  std::vector<std::thread> threads;
  std::atomic<unsigned> threads_active { 0 };
  std::atomic<unsigned> exceptions_raised { 0 };

  // Go go gadget threads
  for(auto* d : sub_dispatchers)
  {
    ++threads_active;
    threads.emplace_back([d,src_factory,&threads_active,&exceptions_raised,log_frequency,start_time,&ndispatched](){
      auto* bsrc = src_factory->new_data_source();
      try {
        d->do_dispatcher_loop(bsrc, log_frequency, start_time, ndispatched);
      } catch(const std::exception& x) {
        util::log::LOG(util::log::FATAL) << x.what();
        ++exceptions_raised;
      }
      delete bsrc;
      --threads_active;
    });
  }

  for(auto& i : threads)i.join();

  if(exceptions_raised) {
    for(auto* d : sub_dispatchers) {
      delete d;
    }
    throw std::runtime_error("Exception(s) thrown in threaded dispatcher loop");
  }

  for(auto* d : sub_dispatchers)
  {
    d->dispatch_leave_run();
    d->dispatch_merge_results();
    delete d;
  }
}

void ParallelEventDispatcher::do_dispatcher_loop(
  calin::io::data_source::DataSource<
    calin::ix::iact_data::telescope_event::TelescopeEvent>* src,
  unsigned log_frequency,
  const std::chrono::system_clock::time_point& start_time,
  std::atomic<uint_fast64_t>& ndispatched)
{
  using namespace std::chrono;
  google::protobuf::Arena* arena = nullptr;
  uint64_t seq_index;
  while(TelescopeEvent* event = src->get_next(seq_index, &arena))
  {
    unsigned ndispatched_val = ndispatched.fetch_add(1);
    if(log_frequency and ndispatched_val and ndispatched_val % log_frequency == 0)
    {
      auto dt = std::chrono::system_clock::now() - start_time;
      LOG(INFO) << "Dispatched "
        << to_string_with_commas(ndispatched_val) << " events in "
        << to_string_with_commas(double(duration_cast<milliseconds>(dt).count())*0.001,3) << " sec";
    }

    add_event_to_keep(event, seq_index, arena);
    keep_event(event);
    dispatch_event(seq_index, event);
    release_event(event);
    arena = nullptr;
  }
}

void ParallelEventDispatcher::write_initial_log_message(
  calin::ix::iact_data::telescope_run_configuration::TelescopeRunConfiguration* run_config,
  int nthread)
{
  auto logger = LOG(INFO);
  logger << "Dispatching " << run_config->filename() << "\n";

  if(nthread) {
    logger << "Using " << nthread << " threads to process ";
  } else {
    logger << "Processing ";
  }

  if(run_config->file_size() > 0) {
    logger << std::setprecision(3) << double(run_config->file_size())*1e-9 << " GB (";
  }
  logger << run_config->fragment_filename_size();
  if(run_config->fragment_filename_size()==1) {
    logger << " file fragment";
  } else {
    logger << " file fragments";
  }
  if(run_config->file_size() > 0) {
    logger << ")";
  }

  if(run_config->run_number() > 0 and run_config->run_start_time().time_ns()>0) {
    logger << "\nRun number: " << run_config->run_number() << ", run start time: "
      << Timestamp(run_config->run_start_time().time_ns()).as_string(/* utc= */ true);
  } else if(run_config->run_number()) {
    logger << "\nRun number: " << run_config->run_number();
  } else if(run_config->run_start_time().time_ns()>0) {
    logger << "\nRun start time: "
      << Timestamp(run_config->run_start_time().time_ns()).as_string();
  }
}

void ParallelEventDispatcher::write_final_log_message(
  unsigned log_frequency, const std::chrono::system_clock::time_point& start_time,
  std::atomic<uint_fast64_t>& ndispatched)
{
  using namespace std::chrono;
  if(log_frequency)
  {
    auto dt = system_clock::now() - start_time;
    if(ndispatched) {
      LOG(INFO) << "Dispatched "
        << to_string_with_commas(uint64_t(ndispatched)) << " events in "
        << to_string_with_commas(double(duration_cast<milliseconds>(dt).count())*0.001,3) << " sec, "
        << to_string_with_commas(duration_cast<microseconds>(dt).count()/ndispatched)
        << " us/event (finished)";
    } else {
      LOG(INFO) << "Dispatched "
        << to_string_with_commas(uint64_t(ndispatched)) << " events in "
        << to_string_with_commas(double(duration_cast<milliseconds>(dt).count())*0.001,3) << " sec (finished)";
    }
  }
}

void ParallelEventDispatcher::
dispatch_run_configuration(TelescopeRunConfiguration* run_config, bool register_processor)
{
  for(auto& iv : visitors_) {
    if(register_processor) {
      iv.processing_record = calin::provenance::chronicle::
        register_processing_start(__PRETTY_FUNCTION__, iv.processing_record_comment);
      if(not run_config->filename().empty()) {
        iv.processing_record->add_primary_inputs(run_config->filename());
      }
      iv.processing_record->set_instance(calin::util::string::instance_identifier(iv.visitor));
    }
    iv.visitor->visit_telescope_run(run_config, this, iv.processing_record);
  }
}

void ParallelEventDispatcher::dispatch_leave_run()
{
  for(auto& iv : visitors_) {
    iv.visitor->leave_telescope_run(iv.processing_record);
    if(iv.processing_record) {
      calin::provenance::chronicle::register_processing_finish(iv.processing_record);
      iv.processing_record = nullptr;
    }
  }
}

void ParallelEventDispatcher::
dispatch_merge_results()
{
  for(const auto& iv : visitors_) {
    iv.visitor->merge_results();
  }
}

namespace {
  template<typename T> bool equal(const T& a, const T& b) {
    if(a.size() != b.size())return false;
    return std::equal(a.begin(), a.end(), b.begin());
  }
}

bool ParallelEventDispatcher::merge_run_config(calin::ix::iact_data::
    telescope_run_configuration::TelescopeRunConfiguration* to,
  const calin::ix::iact_data::
    telescope_run_configuration::TelescopeRunConfiguration& from)
{
  if(to->run_number() != from.run_number())
    throw std::runtime_error("merge_run_config: Mismatch of run_number");
  if(to->telescope_id() != from.telescope_id())
    throw std::runtime_error("merge_run_config: Mismatch of telescope_id");
  if(not equal(to->configured_channel_index(), from.configured_channel_index()))
    throw std::runtime_error("merge_run_config: Mismatch of configured_channel_index");
  if(not equal(to->configured_channel_id(), from.configured_channel_id()))
    throw std::runtime_error("merge_run_config: Mismatch of configured_channel_id");
  if(not equal(to->configured_module_index(), from.configured_module_index()))
    throw std::runtime_error("merge_run_config: Mismatch of configured_module_index");
  if(not equal(to->configured_module_id(), from.configured_module_id()))
    throw std::runtime_error("merge_run_config: Mismatch of configured_module_id");
  if(to->num_samples() != from.num_samples())
    throw std::runtime_error("merge_run_config: Mismatch of num_samples");
  std::string to_camera = to->camera_layout().SerializeAsString();
  std::string from_camera = from.camera_layout().SerializeAsString();
  if(to_camera != from_camera)
    throw std::runtime_error("merge_run_config: Mismatch of camera_layout");
  if(to->has_nectarcam() != from.has_nectarcam()
      or (to->has_nectarcam() and to->nectarcam().SerializeAsString() != from.nectarcam().SerializeAsString()))
    throw std::runtime_error("merge_run_config: Mismatch of NectarCAM specific elements");
  if(to->has_lstcam() != from.has_lstcam()
      or (to->has_lstcam() and to->lstcam().SerializeAsString() != from.lstcam().SerializeAsString()))
    throw std::runtime_error("merge_run_config: Mismatch of LSTCAM specific elements");

  to->set_filename(std::min(to->filename(), from.filename()));
  to->mutable_run_start_time()->set_time_ns(std::min(
    to->run_start_time().time_ns(), from.run_start_time().time_ns()));
  for(const auto& iff : from.fragment_filename())to->add_fragment_filename(iff);
  return true;
}
