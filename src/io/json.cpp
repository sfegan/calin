/*

   calin/io/json.cpp -- Stephen Fegan -- 2021-08-08

   JSON IO functions

   Copyright 2021, Stephen Fegan <sfegan@llr.in2p3.fr>
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

#include <fstream>
#include <sstream>
#include <string>

#include <google/protobuf/descriptor.h>
#include <google/protobuf/message.h>
#include <google/protobuf/util/json_util.h>

#include <io/json.hpp>
#include <util/string.hpp>
#include <util/file.hpp>
#include <provenance/chronicle.hpp>

#include <libgen.h>

#include <sstream>
#include <iomanip>

// https://stackoverflow.com/questions/7724448/simple-json-string-escape-for-c

std::string calin::io::json::escape_json(const std::string &s) {
  std::ostringstream o;
  for (auto c = s.cbegin(); c != s.cend(); c++) {
    switch (*c) {
    case '"': o << "\\\""; break;
    case '\\': o << "\\\\"; break;
    case '\b': o << "\\b"; break;
    case '\f': o << "\\f"; break;
    case '\n': o << "\\n"; break;
    case '\r': o << "\\r"; break;
    case '\t': o << "\\t"; break;
    default:
      if ('\x00' <= *c && *c <= '\x1f') {
        o << "\\u"
          << std::hex << std::setw(4) << std::setfill('0') << (int)*c;
      } else {
        o << *c;
      }
    }
  }
  return o.str();
}

std::string calin::io::json::json_for_dictionary(
  const std::vector<std::pair<std::string,std::string> >& keyval)
{
  std::string s = "{\n";
  for(unsigned i=0;i<keyval.size();++i) {
    const auto& kv = keyval[i];
    std::string comma = (i==keyval.size()-1)?"":",";
    s += " \"" + kv.first + "\": " + kv.second + comma + "\n";
  }
  s+= "}\n";
  return s;
}

std::string calin::io::json::
encode_protobuf_to_json_string(const google::protobuf::Message& message)
{
  google::protobuf::util::JsonPrintOptions opt;
  opt.add_whitespace = true;
  opt.always_print_primitive_fields = true;
  std::string s;
  if(!google::protobuf::util::MessageToJsonString(message, &s, opt).ok()) {
    throw std::runtime_error("Could not encode message as JSON");
  }
  return s;
}

void calin::io::json::save_protobuf_to_json_file(const std::string& filename,
  const google::protobuf::Message& message)
{
  google::protobuf::util::JsonPrintOptions opt;
  opt.add_whitespace = true;
  opt.always_print_primitive_fields = true;
  std::string s;
  if(!google::protobuf::util::MessageToJsonString(message, &s, opt).ok())
    throw std::runtime_error("Could not encode message as JSON");
  auto file_record = calin::provenance::chronicle::register_file_open(filename,
      calin::ix::provenance::chronicle::AT_WRITE, __PRETTY_FUNCTION__);
  std::ofstream stream(filename);
  if(!stream)throw std::runtime_error("Could not open file: "+filename);
  stream << s;
  if(!stream)throw std::runtime_error("Error writing file: "+filename);
  calin::provenance::chronicle::register_file_close(file_record);
}

void calin::io::json::decode_protobuf_from_json_string(const std::string& buffer_str,
  google::protobuf::Message* message)
{
  google::protobuf::util::JsonParseOptions opt;
  opt.ignore_unknown_fields = false;
  if(!google::protobuf::util::JsonStringToMessage(buffer_str, message, opt).ok()) {
    throw std::runtime_error("Could not decode JSON string");
  }
}

void calin::io::json::load_protobuf_from_json_file(const std::string& filename,
  google::protobuf::Message* message)
{
  auto file_record = calin::provenance::chronicle::register_file_open(filename,
      calin::ix::provenance::chronicle::AT_READ, __PRETTY_FUNCTION__);
  std::ifstream stream(filename);
  if(!stream)throw std::runtime_error("Could not open file: "+filename);
  std::stringstream buffer;
  buffer << stream.rdbuf();
  google::protobuf::util::JsonParseOptions opt;
  opt.ignore_unknown_fields = false;
  if(!google::protobuf::util::
    JsonStringToMessage(buffer.str(), message, opt).ok())
    throw std::runtime_error("Could not decode JSON file: "+filename);
  calin::provenance::chronicle::register_file_close(file_record);
}
