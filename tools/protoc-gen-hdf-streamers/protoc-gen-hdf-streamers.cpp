/*

   calin/tools/protoc-gen-hdf-streamers/protoc-gen-hdf_streamers.hpp -- Stephen Fegan -- 2025-02-01

   Procobuf compiler plugin for generating HDF streamers

   Copyright 2025, Stephen Fegan <sfegan@llr.in2p3.fr>
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

#include <google/protobuf/compiler/plugin.h>
#include "hdf_streamers_generator.hpp"

int main(int argc, char** argv) {
  calin::tools::hdf_streamers_generator::HDFStreamersGenerator generator;
  return google::protobuf::compiler::PluginMain(argc, argv, &generator);
}
