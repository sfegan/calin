/*

   calin/iact_data/cta_acada_event_decoder_r1v0.cpp -- Stephen Fegan -- 2018-11-23

   A supplier of single telescope data from CTA DAQ data files

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

#include <string>

#include <util/log.hpp>
#include <iact_data/cta_acada_event_decoder.hpp>
#include <iact_data/nectarcam_acada_event_decoder.hpp>
#include <iact_data/lstcam_acada_event_decoder.hpp>

using namespace calin::ix::iact_data::cta_data_source;
using namespace calin::iact_data::acada_event_decoder;
using namespace calin::iact_data::cta_acada_event_decoder;
using namespace calin::iact_data::nectarcam_acada_event_decoder;
using namespace calin::iact_data::lstcam_acada_event_decoder;
using namespace calin::pattern::delegation;

CTA_ACADACameraEventDecoder_R1v0::
CTA_ACADACameraEventDecoder_R1v0(ACADACameraEventDecoder_R1v0* decoder,
    bool adopt_decoder):
  ACADACameraEventDecoder_R1v0(), Delegator<ACADACameraEventDecoder_R1v0>(decoder, adopt_decoder)
{
  // nothing to see here
}

CTA_ACADACameraEventDecoder_R1v0::
CTA_ACADACameraEventDecoder_R1v0(const std::string& filename, unsigned run_number,
    const calin::ix::iact_data::cta_data_source::CTACameraEventDecoderConfig& config):
  ACADACameraEventDecoder_R1v0(), Delegator<ACADACameraEventDecoder_R1v0>(nullptr),
  filename_(filename), run_number_(run_number), config_(config)
{
  // nothing to see here
}

CTA_ACADACameraEventDecoder_R1v0::~CTA_ACADACameraEventDecoder_R1v0()
{
  // nothing to see here
}

bool CTA_ACADACameraEventDecoder_R1v0::decode(
  calin::ix::iact_data::telescope_event::TelescopeEvent* event,
  const calin::iact_data::acada_data_source::ACADA_MessageSet_R1v0& cta_messages)
{
  ensure_deligate(cta_messages);    
  return this->delegate()->decode(event, cta_messages);
}

bool CTA_ACADACameraEventDecoder_R1v0::decode_run_config(
  calin::ix::iact_data::telescope_run_configuration::
    TelescopeRunConfiguration* run_config,
  const calin::iact_data::acada_data_source::ACADA_MessageSet_R1v0& cta_messages)
{
  ensure_deligate(cta_messages);    
  return this->delegate()->decode_run_config(run_config, cta_messages);
}

void CTA_ACADACameraEventDecoder_R1v0::ensure_deligate(
  const calin::iact_data::acada_data_source::ACADA_MessageSet_R1v0& cta_messages)
{
  const header_type* cta_run_header = cta_messages.header;
  const event_type* cta_event = cta_messages.event;

  if(this->delegate()==nullptr) {
    if((config_.camera_type() == NECTARCAM) or
      ((config_.camera_type() == AUTO_DETECT) and
        ((cta_event!=nullptr and cta_event->has_nectarcam()) or
        (cta_run_header!=nullptr and cta_run_header->has_nectarcam()))))
      this->set_delegate(new NectarCam_ACADACameraEventDecoder_R1v0(filename_, run_number_, config_.nectarcam()),true);
    else if((config_.camera_type() == LSTCAM) or
      ((config_.camera_type() == AUTO_DETECT) and
        ((cta_event!=nullptr and cta_event->has_lstcam()) or
        (cta_run_header!=nullptr and cta_run_header->has_lstcam()))))
      this->set_delegate(new LSTCam_ACADACameraEventDecoder_R1v0(filename_, run_number_, config_.lstcam()),true);
    else
      throw std::runtime_error("CTA_ACADACameraEventDecoder_R1v0: event does not "
        "have NectarCAM or LSTCam extensions");
  } else if(config_.camera_type() == AUTO_DETECT
      and dynamic_cast<LSTCam_ACADACameraEventDecoder_R1v0*>(this->delegate())
      and ((cta_event!=nullptr and cta_event->has_nectarcam()) or
        (cta_run_header!=nullptr and cta_run_header->has_nectarcam()))) {
    this->set_delegate(new NectarCam_ACADACameraEventDecoder_R1v0(filename_, run_number_, config_.nectarcam()),true);
  } else if(config_.camera_type() == AUTO_DETECT
      and dynamic_cast<NectarCam_ACADACameraEventDecoder_R1v0*>(this->delegate())
      and ((cta_event!=nullptr and cta_event->has_lstcam()) or
        (cta_run_header!=nullptr and cta_run_header->has_lstcam()))) {
    this->set_delegate(new LSTCam_ACADACameraEventDecoder_R1v0(filename_, run_number_, config_.lstcam()),true);
  }
}

calin::ix::iact_data::cta_data_source::CTACameraEventDecoderConfig
CTA_ACADACameraEventDecoder_R1v0::default_config()
{
  calin::ix::iact_data::cta_data_source::CTACameraEventDecoderConfig config;
  config.mutable_nectarcam()->CopyFrom(
    NectarCam_ACADACameraEventDecoder_R1v0::default_config());
  config.mutable_lstcam()->CopyFrom(
    LSTCam_ACADACameraEventDecoder_R1v0::default_config());
  return config;
}

CTA_ACADACameraEventDecoder_R1v0* CTA_ACADACameraEventDecoder_R1v0::clone() const {
  return new CTA_ACADACameraEventDecoder_R1v0(this->delegate()->clone(), /* adopt_deligate = */ true);
}
