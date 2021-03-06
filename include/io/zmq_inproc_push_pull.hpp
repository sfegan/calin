/*

   calin/io/zmq_inproc_push_pull.hpp -- Stephen Fegan -- 2016-03-01

   A class to implement ZMQ inprox push/pull sockets

   Copyright 2016, Stephen Fegan <sfegan@llr.in2p3.fr>
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

#pragma once

#include <string>
#include <memory>
#include <atomic>

#include <zmq.h>

namespace calin { namespace io { namespace zmq_inproc {

enum class ZMQProtocol { PUSH_PULL, PUB_SUB };
enum class ZMQBindOrConnect { BIND, CONNECT };

void* new_zmq_ctx();
void destroy_zmq_ctx(void* zmq_cxt);

class ZMQPusher
{
public:
  ZMQPusher(void* zmq_ctx, const std::string& endpoint, int buffer_size = 100,
    ZMQBindOrConnect bind_or_connect = ZMQBindOrConnect::BIND,
    ZMQProtocol protocol = ZMQProtocol::PUSH_PULL);
  bool push(const std::string& data_push, bool dont_wait = false);
#ifndef SWIG
  bool push(const void* data, unsigned size, bool dont_wait = false);
#endif
  void* socket() { return socket_.get(); }
  zmq_pollitem_t pollitem() { return { socket_.get(), 0, ZMQ_POLLOUT, 0 }; }
private:
  std::unique_ptr<void,int(*)(void*)> socket_;
};

class ZMQPuller
{
public:
  ZMQPuller(void* zmq_ctx, const std::string& endpoint, int buffer_size = 100,
    ZMQBindOrConnect bind_or_connect = ZMQBindOrConnect::CONNECT,
    ZMQProtocol protocol = ZMQProtocol::PUSH_PULL);
  bool pull(std::string& data_pull, bool dont_wait = false);
#ifndef SWIG
  bool pull(zmq_msg_t* msg, bool dont_wait = false);
  bool pull(void* data, unsigned buffer_size, unsigned& bytes_received,
     bool dont_wait = false);
  bool pull_assert_size(void* data, unsigned buffer_size,
      bool dont_wait = false);
#endif
  uint64_t nbytes_received() const { return nbytes_pulled_; }
  void* socket() { return socket_.get(); }
  bool wait_for_data(long timeout_ms = -1);
  // Returns 0 if timeout, -1 if error, 1 if this is ready or 2 if puller2 is ready
  int wait_for_data_multi_source(ZMQPuller* puller2, long timeout_ms = -1);
  zmq_pollitem_t pollitem() { return { socket_.get(), 0, ZMQ_POLLIN, 0 }; }
private:
  std::unique_ptr<void,int(*)(void*)> socket_;
  uint64_t nbytes_pulled_ = 0;
};

class ZMQInprocPushPull
{
public:
  ZMQInprocPushPull(void* extern_ctx, unsigned address_index,
    ZMQProtocol protocol = ZMQProtocol::PUSH_PULL, unsigned buffer_size = 100);
  ZMQInprocPushPull(unsigned buffer_size = 100, ZMQInprocPushPull* shared_ctx = nullptr,
    ZMQProtocol protocol = ZMQProtocol::PUSH_PULL);
  ZMQInprocPushPull(ZMQInprocPushPull* shared_ctx, unsigned buffer_size = 100,
      ZMQProtocol protocol = ZMQProtocol::PUSH_PULL):
    ZMQInprocPushPull(buffer_size, shared_ctx, protocol) { }
  ~ZMQInprocPushPull();

  ZMQPuller* new_puller(ZMQBindOrConnect bind_or_connect = ZMQBindOrConnect::CONNECT);
  ZMQPusher* new_pusher(ZMQBindOrConnect bind_or_connect = ZMQBindOrConnect::BIND);

  void* zmq_ctx() { return zmq_ctx_; }
  unsigned address_index() { return address_index_; }
  std::string address();

  unsigned num_puller() const { return num_puller_;  }
  unsigned num_pusher() const { return num_pusher_;  }

private:
  std::atomic<unsigned> zmq_ctx_address_ { 0 };
  unsigned buffer_size_ = 100;
  void* my_zmq_ctx_ = nullptr;
  void* zmq_ctx_ = nullptr;
  unsigned address_index_ = 0;
  ZMQProtocol protocol_ = ZMQProtocol::PUSH_PULL;
  std::atomic<unsigned> num_pusher_ { 0 };
  std::atomic<unsigned> num_puller_ { 0 };
};

} } } // namespace calin::io::zmq_inproc
