//-*-mode:swig;-*-

/*

   calin/io/zmq_inproc_push_pull.i -- Stephen Fegan -- 2018-11-15

   SWIG interface file for calin.io.zmq_inproc_push_pull

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

%module (package="calin.io", threads=1) zmq_inproc_push_pull
%feature(autodoc,2);

%nothread;

%{
#include "io/zmq_inproc_push_pull.hpp"
#define SWIG_FILE_WITH_INIT
  %}

%init %{
  import_array();
%}

%include "calin_typemaps.i"
%import "calin_global_definitions.i"

%newobject new_puller;
%newobject new_pusher;

%apply std::string &CALIN_BYTES_OUT { std::string &data_pull };
%apply const std::string &CALIN_BYTES_IN { const std::string& data_push };

%thread; // Release Python GIL for all functions here (since some use threads)
%include "io/zmq_inproc_push_pull.hpp"
%nothread;
