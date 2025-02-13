/*

   calin/tools/protoc_gen_swig/swig_generator.cpp -- Stephen Fegan -- 2015-12-03

   Code generator for SWIG interface files

   Copyright 2015, Stephen Fegan <sfegan@llr.in2p3.fr>
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

#include <iostream>
#include <sstream>
#include <numeric>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/descriptor.pb.h>
#include <google/protobuf/io/printer.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <util/string.hpp>
#include <calin.pb.h>
#include "swig_generator.hpp"

using std::string;
using namespace calin::util::string;
using namespace calin::tools::protoc_gen_swig;
using google::protobuf::io::Printer;

SwigGenerator::~SwigGenerator()
{
  // nothing to see here
}

namespace {

string pb_to_gen_filename(string pb_filename, string extension = ".pb.i")
{
  return pb_filename.substr(0,pb_filename.find_last_of('.')) + extension;
}

void print_open_namespace(Printer* I,
  const google::protobuf::FileDescriptor * file)
{
  std::vector<string> package_bits = split(file->package(), '.');
  if(!package_bits.empty())
  {
    for(auto ibit : package_bits)I->Print("namespace $bit$ { ","bit",ibit);
    I->Print("\n");
  }
}

void print_close_namespace(Printer* I,
  const google::protobuf::FileDescriptor * file)
{
  std::vector<string> package_bits = split(file->package(), '.');
  if(!package_bits.empty())
  {
    for(auto ibit : package_bits)I->Print("} ");
    I->Print("// namespace $ns$\n","ns",join(package_bits,"::"));
  }
}

void print_includes(Printer* I, const google::protobuf::FileDescriptor * file,
                    string directive, string extension, bool include_self)
{
  if(include_self)
    I->Print("$directive$<$file$>\n", "directive", directive, "file",
             pb_to_gen_filename(file->name(),extension));
  for(int i=0; i<file->dependency_count(); i++)
    I->Print("$directive$<$file$>\n","directive", directive, "file",
             pb_to_gen_filename(file->dependency(i)->name(),extension));
  for(int i=0; i<file->public_dependency_count(); i++)
    I->Print("$directive$<$file$>\n","directive", directive, "file",
             pb_to_gen_filename(file->public_dependency(i)->name(),extension));
  for(int i=0; i<file->weak_dependency_count(); i++)
    I->Print("$directive$<$file$>\n","directive", directive, "file",
             pb_to_gen_filename(file->weak_dependency(i)->name(),extension));
}

// Make the name of the class - handling nested types
string class_name(const google::protobuf::Descriptor* d)
{
  string class_name = d->name();
  while((d = d->containing_type()))class_name = d->name() + "_" + class_name;
  return class_name;
}

std::string stream_writer_name(const google::protobuf::Descriptor* d) 
{
  return class_name(d) + "_StreamWriter";
}

std::string stream_reader_name(const google::protobuf::Descriptor* d) 
{
  return class_name(d) + "_StreamReader";
}

// Make the full type for an enum for use as an argument
string enum_type(const google::protobuf::EnumDescriptor* d,
  const google::protobuf::Descriptor* d_referrer = nullptr)
{
  string enum_type = d->name();
  const google::protobuf::Descriptor* d_sub = d->containing_type();
  while(d_sub)
  {
    if(d_sub == d_referrer)return enum_type;
    enum_type = d_sub->name() + "::" + enum_type;
    d_sub = d_sub->containing_type();
  }
  if(d_referrer and d_referrer->file()->package() == d->file()->package())
    return enum_type;
  else return join(split(d->file()->package(),'.'), "::") + "::" + enum_type;
}

// Make the full type for a message for use as an argument
string class_type(const google::protobuf::Descriptor* d,
  const google::protobuf::Descriptor* d_referrer = nullptr)
{
  string class_type = d->name();
  const google::protobuf::Descriptor* d_sub = d;
  while((d_sub = d_sub->containing_type()))
  {
    if(d_sub == d_referrer)return class_type;
    class_type = d_sub->name() + "_" + class_type;
  }
  if(d_referrer and d_referrer->file()->package() == d->file()->package())
    return class_type;
  else return join(split(d->file()->package(),'.'), "::") + "::" + class_type;
}

void print_fwd_decl(Printer* I, const google::protobuf::Descriptor* d)
{
  for(int i=0; i<d->nested_type_count(); i++)
    print_fwd_decl(I, d->nested_type(i));
  string the_class_name = class_name(d);
  I->Print("class $class_name$;\n", "class_name", the_class_name);
}

string CamelCase(const string& s, bool uc_next = true)
{
  string t;
  for(auto c : s)
  {
    if(c == '_')uc_next=true;
    else if(uc_next)t += std::toupper(c), uc_next=false;
    else t += c;
  }
  return t;
}

string ALLCAPSCASE(const string& s)
{
  string t;
  for(auto c : s)t += std::toupper(c);
  return t;
}

string allsmallcase(const string& s)
{
  string t;
  for(auto c : s)t += std::tolower(c);
  return t;
}

void print_enum(Printer* I, const google::protobuf::EnumDescriptor* e)
{
  std::map<string, string> vars;
  vars["enum_name"] = e->name();
  I->Print(vars,
    "\n"
    "enum $enum_name$ {\n");
  I->Indent();
  int jmax = 0;
  int jmin = 0;
  for(int j=0;j<e->value_count();j++)
  {
    auto* v = e->value(j);
    if(v->number() > e->value(jmax)->number())jmax = j;
    if(v->number() < e->value(jmin)->number())jmin = j;
    vars["value_name"] = v->name();
    vars["value_number"] = std::to_string(v->number());
    vars["comma"] = (j==e->value_count()-1)?string():string(",");
    I->Print(vars,"$value_name$ = $value_number$$comma$\n");
  }
  I->Outdent();
  if(e->containing_type())vars["static"] = "static ";
  else vars["static"] = "";
  vars["min_val"] = e->value(jmin)->name();
  vars["max_val"] = e->value(jmax)->name();
  I->Print(vars, "};\n\n"
    "$static$bool $enum_name$_IsValid(int value);\n"
    "$static$const std::string& $enum_name$_Name($enum_name$ value);\n"
    "$static$bool $enum_name$_Parse(const std::string& name, $enum_name$ *CALIN_INT_OUTPUT);\n"
    "$static$const $enum_name$ $enum_name$_MIN = $min_val$;\n"
    "$static$const $enum_name$ $enum_name$_MAX = $max_val$;\n");
}

string field_type(const google::protobuf::FieldDescriptor* f,
  const google::protobuf::Descriptor* d)
{
  if(f->type() == google::protobuf::FieldDescriptor::TYPE_MESSAGE)
    return class_type(f->message_type(), d);
  else if(f->type() == google::protobuf::FieldDescriptor::TYPE_ENUM)
    return enum_type(f->enum_type(), d);
  else if(f->type() == google::protobuf::FieldDescriptor::TYPE_STRING or
          f->type() == google::protobuf::FieldDescriptor::TYPE_BYTES)
    return "std::string";
  else return f->cpp_type_name();
}

string field_type_const_in(const google::protobuf::FieldDescriptor* f,
  const google::protobuf::Descriptor* d)
{
  if(f->type() == google::protobuf::FieldDescriptor::TYPE_MESSAGE)
    return "const " + class_type(f->message_type(), d) + "*";
  else if(f->type() == google::protobuf::FieldDescriptor::TYPE_STRING or
     f->type() == google::protobuf::FieldDescriptor::TYPE_BYTES)
    return "const std::string&";
  else return field_type(f,d);
}

bool is_type_compatible_with_numpy(google::protobuf::FieldDescriptor::Type type)
{
  switch(type)
  {
  case google::protobuf::FieldDescriptor::TYPE_DOUBLE:    // fall through
  case google::protobuf::FieldDescriptor::TYPE_FLOAT:     // fall through
  case google::protobuf::FieldDescriptor::TYPE_INT64:     // fall through
  case google::protobuf::FieldDescriptor::TYPE_UINT64:    // fall through
  case google::protobuf::FieldDescriptor::TYPE_INT32:     // fall through
  case google::protobuf::FieldDescriptor::TYPE_FIXED64:   // fall through
  case google::protobuf::FieldDescriptor::TYPE_FIXED32:   // fall through
  case google::protobuf::FieldDescriptor::TYPE_UINT32:    // fall through
  case google::protobuf::FieldDescriptor::TYPE_SFIXED32:  // fall through
  case google::protobuf::FieldDescriptor::TYPE_SFIXED64:  // fall through
  case google::protobuf::FieldDescriptor::TYPE_SINT32:    // fall through
  case google::protobuf::FieldDescriptor::TYPE_SINT64:    // fall through
    return true;
  default:
    return false;
  }
  assert(0);
  return false;
}

void print_message(Printer* I, const google::protobuf::Descriptor* d)
{
  for(int i=0; i<d->nested_type_count(); i++)
    if(!d->nested_type(i)->options().map_entry())
      print_message(I, d->nested_type(i));

  string the_class_name = class_name(d);
  string the_class_type = class_type(d);
  I->Print(
    "\n"
    "%newobject $class_name$::New() const;\n"
    "%newobject $class_name$::Clone() const;\n"
    "%newobject $class_name$::NewHDFStreamWriter;\n"
    "%newobject $class_name$::NewHDFStreamReader;\n"
    "\n"
    "class $class_name$;\n"
    "\n"
    "class $stream_writer_name$\n"
    "{\n"
    "public:\n"
    "  virtual ~$stream_writer_name$();\n"
    "  virtual uint64_t nrow() = 0;\n"
    "  virtual void write(const $class_name$& m) = 0;\n"
    "  virtual void flush() = 0;\n"
    "};\n"
    "class $stream_reader_name$\n"
    "{\n"
    "public:\n"
    "  virtual ~$stream_reader_name$();\n"
    "  virtual uint64_t nrow() = 0;\n"
    "  virtual bool read(uint64_t irow, $class_name$* m) = 0;\n"
    "  virtual bool preload(uint64_t start, uint64_t count) = 0;\n"
    "};\n"
    "\n"
    "class $class_name$ : public google::protobuf::Message \n"
    "{\n"
    " public:\n"
    "  $class_name$();\n"
    "  ~$class_name$();\n"
    "  $class_name$(const $class_name$& other);\n"
    "  $class_name$* New() const;\n"
    "  $class_name$* New(google::protobuf::Arena*) const;\n"
    "  %extend {\n"
    "    $class_name$* Clone() const {\n"
    "      $class_type$* the_clone = $$self->New();\n"
    "      the_clone->MergeFrom(*$$self); return the_clone; }\n"
    "    $class_name$* Clone(google::protobuf::Arena* arena) const {\n"
    "      $class_type$* the_clone = $$self->New(arena);\n"
    "      the_clone->MergeFrom(*$$self); return the_clone; }\n"
    "    bool Equals(const $class_name$& other) const {\n"
    "      return google::protobuf::util::MessageDifferencer::Equals(*$$self,other); }\n"
    "    %pythoncode %{"
    "\n"
    "      def __getstate__(self):\n"
    "        return self.SerializeAsString()\n"
    "\n"
    "      def __setstate__(self, serialized_data):\n"
    "        if not hasattr(self,\"this\"):\n"
    "          self.__init__()\n"
    "        self.ParseFromString(serialized_data)\n"
    "    %}\n"
    "  }\n"
    "  void Swap($class_name$* other);\n"
    "  const google::protobuf::Descriptor* GetDescriptor() const;\n"
    "\n"
    "  static const google::protobuf::Descriptor* descriptor();\n"
    "  static const $class_name$& default_instance();\n"
    "  static $stream_writer_name$* NewHDFStreamWriter(const std::string& filename, const std::string& groupname, bool truncate = false);\n" 
    "  static $stream_reader_name$* NewHDFStreamReader(const std::string& filename, const std::string& groupname);\n",
    "class_name", the_class_name,
    "class_type", the_class_type,
    "stream_writer_name", stream_writer_name(d),
    "stream_reader_name", stream_reader_name(d));

  const google::protobuf::MessageOptions* mopt = &d->options();
  const calin::MessageOptions* cmo = &mopt->GetExtension(calin::CMO);
  if(cmo->message_integration_function() != calin::MessageOptions::MIF_NONE)
    I->Print(
      "\n"
      "  void IntegrateFrom(const $class_name$& from);\n",
      "class_name", the_class_name);

  I->Indent();

  // Typedefs for nested types
  if(d->nested_type_count())
  {
    for(int i=0; i<d->nested_type_count(); i++)
      if(!d->nested_type(i)->options().map_entry())
      {
        I->Print("\n");
        break;
      }
    for(int i=0; i<d->nested_type_count(); i++)
      if(!d->nested_type(i)->options().map_entry())
        I->Print("typedef $local$ $full$;\n",
          "local", d->nested_type(i)->name(),
          "full", class_name(d->nested_type(i)));
  }

  // Enums
  for(int i=0;i<d->enum_type_count(); i++)
    print_enum(I, d->enum_type(i));

  // Oneofs
  for(int i=0;i<d->oneof_decl_count();i++)
  {
    auto* oo = d->oneof_decl(i);
    std::map<string, string> vars;
    vars["oo_name"] = oo->name();
    vars["oo_cc_name"] = CamelCase(oo->name());
    vars["oo_ac_name"] = ALLCAPSCASE(oo->name());
    I->Print(vars,"\n"
             "enum $oo_cc_name$Case {\n");
    I->Indent();
    for(int j=0;j<oo->field_count();j++)
    {
      auto* f = oo->field(j);
      vars["field_cc_name"] = CamelCase(f->name());
      vars["field_number"] = std::to_string(f->number());
      I->Print(vars,"k$field_cc_name$ = $field_number$,\n");
    }
    I->Print(vars, "$oo_ac_name$_NOT_SET = 0\n");
    I->Outdent();
    I->Print(vars,
      "};\n\n"
      "$oo_cc_name$Case $oo_name$_case() const;\n");
  }

  // Fields
  for(int i=0;i<d->field_count();i++)
  {
    auto* f = d->field(i);
    const google::protobuf::FieldOptions* fopt { &f->options() };

    std::map<string, string> vars;
    vars["id"]          = std::to_string(f->index());
    vars["name"]        = allsmallcase(f->name());
    vars["type"]        = field_type(f, d);
    vars["type_in"]     = field_type_const_in(f, d);
    vars["class_name"]  = the_class_name;
    vars["scoped_type"] = field_type(f, nullptr);

    I->Print(vars,
      "\n"
      "// Field: $name$ = $id$ [$type$]\n");

    if(f->is_map())
    {
      assert(f->message_type());
      assert(f->message_type()->field_count()==2);
      const auto* fi = f->message_type()->field(0);
      const auto* fv = f->message_type()->field(1);
      vars["key_type"] = field_type(fi, d);
      vars["key_type_in"] = field_type_const_in(fi, d);
      vars["type"] = field_type(fv, d);
      vars["type_in"] = field_type_const_in(fv, d);
      I->Print("%extend {\n");
      I->Indent();
      I->Print(vars,
        "int $name$_size() const {\n"
        "  return $$self->$name$().size(); }\n"
        "bool $name$_has_key($key_type_in$ key) const {\n"
        "  return $$self->$name$().count(key)>0; }\n"
        "std::vector<$key_type$> $name$_keys() const {\n"
        "  std::vector<$key_type$> keys;\n"
        "  for(const auto& i : $$self->$name$())keys.emplace_back(i.first);\n"
        "  return keys; }\n");
      if(fv->type() == google::protobuf::FieldDescriptor::TYPE_MESSAGE)
      {
        I->Print(vars,
          "const $type$& const_$name$($key_type_in$ key) const {\n"
          "  return $$self->$name$().at(key); }\n"
          "$type$& mutable_$name$($key_type_in$ key) {\n"
          "  return (*$$self->mutable_$name$())[key]; }\n"
          "$type$& $name$($key_type_in$ key) {\n"
          "  return (*$$self->mutable_$name$())[key]; }\n");
      }
      else if(fv->type() == google::protobuf::FieldDescriptor::TYPE_BYTES)
      {
        I->Print(vars,
          "%extend {\n"
          "  void $name$($key_type_in$ key, std::string& CALIN_BYTES_OUT) const {\n"
          "    CALIN_BYTES_OUT = $$self->$name$().at(key); }\n"
          "  void set_$name$($key_type_in$ key, const std::string& CALIN_BYTES_IN) {\n"
          "    (*$$self->mutable_$name$())[key] = CALIN_BYTES_IN; }\n"
          "};\n");
      }
      else // POD type
      {
        I->Print(vars,
          "$type$ $name$($key_type_in$ key) const {\n"
          "  return $$self->$name$().at(key); }\n"
          "void set_$name$($key_type_in$ key, $type_in$ val) {\n"
          "  (*$$self->mutable_$name$())[key] = val; }\n");
      }
      I->Outdent();
      I->Print("};\n");
    }
    else if(f->is_repeated())
    {
      I->Print(vars, "int $name$_size() const;\n");
      if(f->type() == google::protobuf::FieldDescriptor::TYPE_MESSAGE)
      {
        I->Print("%extend {\n");
        I->Indent();
        I->Print(vars,
          "const $type$& const_$name$(int index) const {\n"
          "  verify_range(index, $$self->$name$_size());\n"
          "  return $$self->$name$(index); }\n"
          "$type$* mutable_$name$(int index) {\n"
          "  verify_range(index, $$self->$name$_size());\n"
          "  return $$self->mutable_$name$(index); }\n"
          "$type$* $name$(int index) {\n"
          "  verify_range(index, $$self->$name$_size());\n"
          "  return $$self->mutable_$name$(index); }\n"
          "std::vector<const $scoped_type$*> const_$name$() {\n"
          "  const auto& array = $$self->$name$();\n"
          "  std::vector<const $scoped_type$*> OUTPUT(array.size());\n"
          "  std::copy(array.data(), array.data()+array.size(), OUTPUT.begin());\n"
          "  return OUTPUT;\n"
          "}\n"
          "std::vector<$scoped_type$*> mutable_$name$() {\n"
          "  auto* array = $$self->mutable_$name$();\n"
          "  std::vector<$scoped_type$*> OUTPUT(array->size());\n"
          "  std::copy(array->mutable_data(), array->mutable_data()+array->size(), OUTPUT.begin());\n"
          "  return OUTPUT;\n"
          "}\n"
          "std::vector<$scoped_type$*> $name$() {\n"
          "  auto* array = $$self->mutable_$name$();\n"
          "  std::vector<$scoped_type$*> OUTPUT(array->size());\n"
          "  std::copy(array->mutable_data(), array->mutable_data()+array->size(), OUTPUT.begin());\n"
          "  return OUTPUT;\n"
          "}\n"
          "void set_$name$(const std::vector<const $scoped_type$*>& INPUT) {\n"
          "  $$self->clear_$name$();\n"
          "  for(const auto* m : INPUT)$$self->add_$name$()->MergeFrom(*m);\n"
          "}\n");
        I->Outdent();
        I->Print("};\n");
        I->Print(vars, "$type$* add_$name$();\n");
      }
      else if(f->type() == google::protobuf::FieldDescriptor::TYPE_BYTES)
      {
        I->Print("%extend {\n");
        I->Indent();
        I->Print(vars,
          "void $name$(int index, std::string& CALIN_BYTES_OUT) const {\n"
          "  verify_range(index, $$self->$name$_size());\n"
          "  CALIN_BYTES_OUT = $$self->$name$(index); }\n"
          "void set_$name$(int index, const std::string& CALIN_BYTES_IN) {\n"
          "  verify_range(index, $$self->$name$_size());\n"
          "  $$self->set_$name$(index, CALIN_BYTES_IN); }\n"
          "void add_$name$(const std::string& CALIN_BYTES_IN) {\n"
          "  $$self->add_$name$(CALIN_BYTES_IN); }\n");
        I->Outdent();
        I->Print("};\n");
      }
      else if(f->type() == google::protobuf::FieldDescriptor::TYPE_STRING)
      {
        I->Print("%extend {\n");
        I->Indent();
        I->Print(vars,
          "std::string $name$(int index) const {\n"
          "  verify_range(index, $$self->$name$_size());\n"
          "  return $$self->$name$(index); }\n"
          "void set_$name$(int index, const std::string& INPUT) {\n"
          "  verify_range(index, $$self->$name$_size());\n"
          "  $$self->set_$name$(index, INPUT); }\n"
          "void $name$(std::vector<std::string> &OUTPUT) {\n"
          "  const auto& array = $$self->$name$();\n"
          "  OUTPUT.resize(array.size());\n"
          "  std::copy(array.begin(), array.end(), OUTPUT.begin());\n"
          "};\n"
          "void set_$name$(const std::vector<std::string>& INPUT) {\n"
          "  $$self->clear_$name$();\n"
          "  for(const auto& s : INPUT)$$self->add_$name$(s);\n"
          "}\n");
        I->Outdent();
        I->Print("};\n");
        I->Print(vars, "void add_$name$(const std::string& INPUT);\n");
      }
      else // POD type
      {
        I->Print("%extend {\n");
        I->Indent();
        I->Print(vars,
          "$type$ $name$(int index) const {\n"
          "  verify_range(index, $$self->$name$_size());\n"
          "  return $$self->$name$(index); }\n"
          "void set_$name$(int index, $type_in$ INPUT) {\n"
          "  verify_range(index, $$self->$name$_size());\n"
          "  $$self->set_$name$(index, INPUT); }\n");
        if(is_type_compatible_with_numpy(f->type()))
          I->Print(vars,
            "void set_$name$(intptr_t DIM1, $type$* IN_ARRAY1) {\n"
            "  auto* array = $$self->mutable_$name$();\n"
            "  if(array->size()>DIM1)array->Truncate(DIM1);\n"
            "  for(int i=0;i<array->size();i++)array->Set(i,IN_ARRAY1[i]);\n"
            "  while(array->size()<DIM1)array->Add(IN_ARRAY1[array->size()]);\n"
            "}\n"
            "void $name$(intptr_t* DIM1, $type$** ARGOUTVIEWM_ARRAY1) {\n"
            "  const auto& array = $$self->$name$();\n"
            "  *DIM1 = array.size();\n"
            "  *ARGOUTVIEWM_ARRAY1 = ($type$*)malloc(*DIM1 * sizeof($type$));\n"
            "  std::copy(array.begin(), array.end(), *ARGOUTVIEWM_ARRAY1);\n"
            "};\n"
            "void $name$_copy(intptr_t* DIM1, $type$** ARGOUTVIEWM_ARRAY1) {\n"
            "  const auto& array = $$self->$name$();\n"
            "  *DIM1 = array.size();\n"
            "  *ARGOUTVIEWM_ARRAY1 = ($type$*)malloc(*DIM1 * sizeof($type$));\n"
            "  std::copy(array.begin(), array.end(), *ARGOUTVIEWM_ARRAY1);\n"
            "};\n"
            "void $name$_view(intptr_t* DIM1, $type$** ARGOUTVIEW_ARRAY1) {\n"
            "  auto* array = $$self->mutable_$name$();\n"
            "  *DIM1 = array->size();\n"
            "  *ARGOUTVIEW_ARRAY1 = ($type$*)array->mutable_data();\n"
            "};\n");
        I->Outdent();
        I->Print("};\n");
        I->Print(vars, "void add_$name$($type_in$ INPUT);\n");
      }
    }
    else // Is a singular type
    {
      if(f->type() == google::protobuf::FieldDescriptor::TYPE_MESSAGE)
      {
        I->Print(vars,
          "bool has_$name$() const;\n"
          "%extend {\n");
        I->Indent();
        I->Print(vars,
          "const $type$& const_$name$() { \n"
          "  return $$self->$name$(); }\n"
          "$type$* $name$() {\n"
          "  return $$self->mutable_$name$(); }\n");
        I->Outdent();
        I->Print(vars,
          "}\n"
          "$type$* mutable_$name$();\n");
      }
      else if(f->type() == google::protobuf::FieldDescriptor::TYPE_BYTES)
      {
        I->Print(vars, "%extend {\n");
        I->Indent();
        I->Print(vars,
          "void $name$(std::string& CALIN_BYTES_OUT) const {\n"
          "  CALIN_BYTES_OUT = $$self->$name$(); }\n"
          "void set_$name$(const std::string& CALIN_BYTES_IN) {\n"
          "  $$self->set_$name$(CALIN_BYTES_IN); }\n");
        I->Outdent();
        I->Print(vars, "};\n");
      }
      else // string or POD type
      {
        I->Print(vars,
          "$type$ $name$() const;\n"
          "void set_$name$($type_in$ INPUT);\n");
        if(fopt->HasExtension(calin::CFO)) {
          if(fopt->GetExtension(calin::CFO).is_counter()) {
            I->Print(vars,
              "$type$ increment_$name$();\n"
              "$type$ increment_$name$_if(bool);\n");
          }
        }

      }
    }

    I->Print(vars, "void clear_$name$();\n");

    I->Print("%extend {\n");
    I->Indent();

    if(fopt->HasExtension(calin::CFO))
    {
      if(!fopt->GetExtension(calin::CFO).desc().empty())
      {
        vars["desc"] = string_escape(fopt->GetExtension(calin::CFO).desc());
        I->Print(vars,
          "static std::string $name$_desc() { return \"$desc$\"; }\n");
      }

      if(!fopt->GetExtension(calin::CFO).units().empty())
      {
        vars["units"] = string_escape(fopt->GetExtension(calin::CFO).units());
        I->Print(vars,
          "static std::string $name$_units() { return \"$units$\"; }\n");
      }
    }

    if(vars.find("desc") == vars.end())
      I->Print(vars, "static void $name$_desc() { }\n");
    if(vars.find("units") == vars.end())
      I->Print(vars, "static void $name$_units() { }\n");

    I->Outdent();
    I->Print(vars, "};\n");
  }

  // End of class
  I->Outdent();
  I->Print("};\n");

  I->Print("\n");
  print_close_namespace(I,d->file());

  string package_name = d->file()->package();
  for(size_t ifind = package_name.find_first_of('.'); ifind != string::npos;
      ifind = package_name.find_first_of('.', ifind+=2))
    package_name.replace(ifind, 1, "::");

  string package_name_cc = d->file()->package();
  for(size_t ifind = package_name_cc.find_first_of('.'); ifind != string::npos;
      ifind = package_name_cc.find_first_of('.', ifind))
    package_name_cc[ifind++] = '_';

  // I->Print(
  //   "\n"
  //   "%template(Vector_$package_name_cc$_$class_name$)\n"
  //   "  std::vector<$package_name$::$class_name$>;\n",
  //   "package_name", package_name,
  //   "package_name_cc", CamelCase(package_name_cc),
  //   "class_name", the_class_name);
  I->Print(
    "\n"
    "%template(Vector_Ptr_$package_name_cc$_$class_name$)\n"
    "  std::vector<$package_name$::$class_name$*>;\n",
    "package_name", package_name,
    "package_name_cc", CamelCase(package_name_cc),
    "class_name", the_class_name);
  I->Print(
    "\n"
    "%template(Vector_ConstPtr_$package_name_cc$_$class_name$)\n"
    "  std::vector<const $package_name$::$class_name$*>;\n",
    "package_name", package_name,
    "package_name_cc", CamelCase(package_name_cc),
    "class_name", the_class_name);

  I->Print("\n");
  print_open_namespace(I,d->file());
}

} // anonymous namespace

bool SwigGenerator::
Generate(const google::protobuf::FileDescriptor * file,
         const string & parameter,
         google::protobuf::compiler::GeneratorContext *context,
         string * error) const
{
  auto I_stream = context->Open(pb_to_gen_filename(file->name(),".pb.i"));
  Printer* I = new Printer(I_stream,'$');

  I->Print("// Auto-generated from \"$file$\". "
           "Do not edit by hand.\n\n","file",file->name());

  std::map<string,string> vars;
  if(file->package().find('.') != string::npos)
  {
    auto index = file->package().find_last_of('.');
    I->Print("%module (package=\"$package$\") $module$\n",
             "package", file->package().substr(0,index),
             "module", file->package().substr(index+1));
  }
  else
  {
    I->Print("%module $module$\n",
             "module", file->package());
  }

  I->Print("\n%{\n");
  I->Indent();
  I->Print(
    "#include<cstdint>\n"
    "#include<string>\n"
    "#include<vector>\n"
    "#include<memory>\n"
    "#include<numeric>\n"
    "#include<map>\n"
    "#include<stdexcept>\n"
    "#include<google/protobuf/message.h>\n"
    "#include<google/protobuf/descriptor.h>\n"
    "#include<google/protobuf/util/message_differencer.h>\n");
  print_includes(I, file, "#include", ".pb.h", true);
  I->Print("\n#define SWIG_FILE_WITH_INIT\n\n");
  I->Print(
    "namespace {\n"
    "  void verify_range(int& index, int size) __attribute__((unused));\n\n"
    "  void verify_range(int& index, int size) {\n"
    "    assert(size>=0);\n"
    "    if(index<-size)throw std::range_error(\"Index out of range: \"\n"
    "      + std::to_string(index) + \" < -\" + std::to_string(size));\n"
    "    else if(index>=size)throw std::range_error(\"Index out of range: \"\n"
    "      + std::to_string(index) + \" >= \" + std::to_string(size));\n"
    "    else if(index<0)index += size; }\n"
    "} // private namespace\n");
  I->Outdent();
  I->Print("%}\n\n");

  I->Print("%init %{\n"
           "  import_array();\n"
           "%}\n\n");

  I->Print("%include<calin_typemaps.i>\n");
  I->Print("%import<calin_global_definitions.i>\n");

  I->Print("%include<typemaps.i>\n");
  I->Print("%include<calin_numpy.i>\n");
  I->Print("%include<calin_stdint.i>\n");
  I->Print("%include<std_string.i>\n");
  //I->Print("%include<std_vector.i>\n");
  //I->Print("%include<std_map.i>\n");

  I->Print("%import<google_protobuf.i>\n");
  print_includes(I, file, "%import", ".pb.i", false);

  I->Print("\n"
           "#define int32 int32_t\n"
           "#define uint32 uint32_t\n"
           "#define int64 int64_t\n"
           "#define uint64 uint64_t\n");

  // Print open namespace
  I->Print("\n");
  print_open_namespace(I, file);

  // Print forward declaration for all message classes
  for(int i=0;i<file->message_type_count();i++)
  {
    if(i==0)I->Print("\n");
    print_fwd_decl(I, file->message_type(i));
  }

  // Print enum definitions from main scope
  for(int i=0;i<file->enum_type_count(); i++)
    print_enum(I, file->enum_type(i));

  // Print classes for all messages
  for(int i=0;i<file->message_type_count();i++)
    print_message(I, file->message_type(i));

  // Print close namespace
  I->Print("\n");
  print_close_namespace(I, file);

  delete I;
  delete I_stream;
  return true;
}
