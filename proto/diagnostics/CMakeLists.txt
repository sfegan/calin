# calin/proto/diagnostics/CMakeLists.txt -- Stephen Fegan
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

set(CALIN_SWIG_PROTO waveform.proto module.proto counter.proto
	event_number.proto functional.proto value_capture.proto range.proto
	run_info.proto simple_charge_stats.proto simple_charge_hists.proto
	clock_regression.proto stage1.proto reduced_event_writer.proto reduced_event.proto)
set(CALIN_PACKAGE diagnostics)
set(CALIN_TARGET_LIBRARY calin_proto_${CALIN_PACKAGE})

make_proto_modules(${CALIN_TARGET_LIBRARY} ${CALIN_PACKAGE} ${CALIN_SWIG_PROTO}
	range_mif.cpp)
target_link_libraries(${CALIN_TARGET_LIBRARY} calin_proto_math
	calin_proto_iact_data calin_proto_provenance)
make_proto_swig_modules(${CALIN_TARGET_LIBRARY}
	${CALIN_PROTO_SWIG_INSTALL_DIR}/${CALIN_PACKAGE} ${CALIN_SWIG_PROTO})
