# calin/src/io/CMakeLists.txt -- Stephen Fegan
#
# Copyright 2015, Stephen Fegan <sfegan@llr.in2p3.fr>
# Laboratoire Leprince-Ringuet, CNRS/IN2P3, Ecole Polytechnique, Institut Polytechnique de Paris
#
# This file is part of "calin"
#
# "calin" is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License version 2 or later, as published by
# the Free Software Foundation.
#
# "calin" is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

set(CALIN_TARGET_LIBRARY calin_io)
set(CALIN_SOURCES sql_statement.cpp sqlite3_statement.cpp
	sql_serializer.cpp sqlite3_serializer.cpp
	data_source.cpp chained_data_source.cpp packet_stream.cpp
	zmq_inproc_push_pull.cpp json.cpp options_processor.cpp)

include_directories(${Python3_INCLUDE_DIRS})
add_library(${CALIN_TARGET_LIBRARY} SHARED ${CALIN_SOURCES})
target_link_libraries(${CALIN_TARGET_LIBRARY} calin_util calin_provenance calin_proto_io
	${SQLITE3_LIBRARY} ${ZMQ_LIBRARIES} ${PROTOBUF_LIBRARY} -lutil)

install(TARGETS ${CALIN_TARGET_LIBRARY} DESTINATION ${CALIN_LIB_INSTALL_DIR})
