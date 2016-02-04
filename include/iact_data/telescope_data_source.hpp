/*

   calin/iact_data/telescope_data_source.hpp -- Stephen Fegan -- 2016-01-08

   A supplier of single telescope data, for example:
   ix::iact_data::telescope_event::TelescopeEvent

   Copyright 2016, Stephen Fegan <sfegan@llr.in2p3.fr>
   LLR, Ecole polytechnique, CNRS/IN2P3, Universite Paris-Saclay

   This file is part of "calin"

   "calin" is free software: you can redistribute it and/or modify it
   under the terms of the GNU General Public License version 2 or
   later, as published by the Free Software Foundation.

   "calin" is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.

*/

#pragma once

#include <calin_global_definitions.hpp>
#include <io/data_source.hpp>
#include <iact_data/telescope_event.pb.h>

namespace calin { namespace iact_data { namespace telescope_data_source {

CALIN_TYPEALIAS(TelescopeDataSource,
  calin::io::data_source::DataSource<
    calin::ix::iact_data::telescope_event::TelescopeEvent>);

CALIN_TYPEALIAS(TelescopeRandomAccessDataSource,
  calin::io::data_source::RandomAccessDataSource<
    calin::ix::iact_data::telescope_event::TelescopeEvent>);

CALIN_TYPEALIAS(RawFileTelescopeDataSource,
  calin::io::data_source::ProtobufFileDataSource<
    calin::ix::iact_data::telescope_event::TelescopeEvent>);

CALIN_TYPEALIAS(TelescopeDataSink,
  calin::io::data_source::DataSink<
    calin::ix::iact_data::telescope_event::TelescopeEvent>);

CALIN_TYPEALIAS(RawFileTelescopeDataSink,
  calin::io::data_source::ProtobufFileDataSink<
    calin::ix::iact_data::telescope_event::TelescopeEvent>);



} } } // namespace calin::iact_data::telescope_data_source

#ifndef SWIG
#ifndef CALIN_TELESCOPE_DATA_SOURCE_NO_EXTERN
extern template class calin::io::data_source::DataSource<
  calin::ix::iact_data::telescope_event::TelescopeEvent>;
extern template class calin::io::data_source::RandomAccessDataSource<
  calin::ix::iact_data::telescope_event::TelescopeEvent>;
extern template class calin::io::data_source::ProtobufFileDataSource<
  calin::ix::iact_data::telescope_event::TelescopeEvent>;
extern template class calin::io::data_source::DataSink<
  calin::ix::iact_data::telescope_event::TelescopeEvent>;
extern template class calin::io::data_source::ProtobufFileDataSink<
  calin::ix::iact_data::telescope_event::TelescopeEvent>;
#endif // #ifdef CALIN_TELESCOPE_DATA_SOURCE_NO_EXTERN
#endif