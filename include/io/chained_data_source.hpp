/*

   calin/io/chained_data_source.hpp -- Stephen Fegan -- 2016-01-15

   ChaninedFile(RandomAccess)DataSource - chain a number of file-based
   (RandomAccess)DataSource together

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

#include <util/log.hpp>
#include <io/data_source.hpp>

namespace calin { namespace io { namespace data_source {

class FragmentList
{
public:
  virtual ~FragmentList();

  virtual unsigned current_fragment_index() const = 0;
  virtual unsigned num_fragments() const = 0;
  virtual unsigned num_missing_fragments() const = 0;
  virtual std::string fragment_name(unsigned index) const = 0;

  std::string current_fragment_name() const {
    return fragment_name(current_fragment_index());
  }
  std::vector<std::string> all_fragment_names() const {
    std::vector<std::string> fragments(num_fragments());
    for(unsigned i=0;i<num_fragments();i++)fragments[i] = fragment_name(i);
    return fragments;
  }
};

template<typename DST> class DataSourceOpener
{
public:
  CALIN_TYPEALIAS(data_source_type, DST);
  virtual ~DataSourceOpener() { }
  virtual unsigned num_sources() const = 0;
  virtual unsigned num_missing_sources() const = 0;
  virtual std::string source_name(unsigned isource) const = 0;
  std::vector<std::string> source_names() const {
    std::vector<std::string> source_names;
    for(unsigned isource=0; isource<this->num_sources(); isource++)
      source_names.push_back(this->source_name(isource));
    return source_names;
  }
  virtual DST* open(unsigned isource) = 0;
};

template<typename DST> class BasicChainedDataSource:
  public DST, public FragmentList
{
public:
  CALIN_TYPEALIAS(data_type, typename DST::data_type);

  BasicChainedDataSource(DataSourceOpener<DST>* opener,
      bool adopt_opener = false):
    DST(), opener_(opener), adopt_opener_(adopt_opener)
  {
    open_file();
  }

  ~BasicChainedDataSource()
  {
    if(adopt_opener_)delete opener_;
    delete source_;
  }

  data_type* get_next(uint64_t& seq_index_out,
    google::protobuf::Arena** arena = nullptr) override
  {
    uint64_t unused_index = 0;
    while(isource_ < opener_->num_sources())
    {
      if(data_type* next = source_->get_next(unused_index, arena))
      {
        seq_index_out = seq_index_;
        ++seq_index_;
        return next;
      }
      ++isource_;
      open_file();
    }
    return nullptr;
  }

  unsigned current_fragment_index() const override { return isource_; }
  unsigned num_fragments() const override { return opener_->num_sources(); }
  unsigned num_missing_fragments() const override { return opener_->num_missing_sources(); }
  std::string fragment_name(unsigned index) const override {
    return opener_->source_name(index); }

protected:
  virtual void open_file()
  {
    if(source_)delete source_;
    source_ = nullptr;
    if(isource_ < opener_->num_sources())
    {
      source_ = opener_->open(isource_);
      // Throw exception if opener doesn't throw its own
      if(source_ == nullptr)throw std::runtime_error("Could not open source " +
        std::to_string(isource_));
    }
  }

  DataSourceOpener<DST>* opener_ = nullptr;
  bool adopt_opener_ = false;
  unsigned isource_ = 0;
  DST* source_ = nullptr;
  uint64_t seq_index_ = 0;
};

#ifndef SWIG
template<typename T> using ChainedDataSource =
  BasicChainedDataSource<DataSource<T> >;
#endif

template<typename RADST> class BasicChainedRandomAccessDataSource:
  public BasicChainedDataSource<RADST>
{
public:
  CALIN_TYPEALIAS(data_type, typename RADST::data_type);

  BasicChainedRandomAccessDataSource(DataSourceOpener<RADST>* opener,
    bool adopt_opener = false):
    BasicChainedDataSource<RADST>(opener, adopt_opener)
  {
    // Base class can't call virtual open_file function so we complete it
    if(source_)add_source_index(source_);
  }

  virtual ~BasicChainedRandomAccessDataSource() {
    /* nothing to see here */ }

  virtual uint64_t size() override
  {
    for(unsigned i = chained_file_index_.size(); i<opener_->num_sources(); i++)
    {
      std::unique_ptr<RADST> source(opener_->open(i));
      // Throw exception in case opener doesn't do so itself
      if(!source)throw std::runtime_error("Could not open file " +
        std::to_string(i));
      add_source_index(source.get());
    }

    if(opener_->num_sources())return chained_file_index_.back();
    else return 0;
  }

  void set_next_index(uint64_t next_index) override
  {
    if(opener_->num_sources() == 0)return;

    if(next_index < chained_file_index_.back())
    {
      // next_index belongs to one of the files we have already indexed
      auto file_index = std::upper_bound(chained_file_index_.begin(),
        chained_file_index_.end(), next_index);
      unsigned new_isource = int(file_index-chained_file_index_.begin());
      if(new_isource != isource_)
      {
        isource_ = new_isource;
        BasicChainedDataSource<RADST>::open_file();
      }
    }
    else
    {
      while(isource_<opener_->num_sources() and
        chained_file_index_.back()<=next_index)
      {
        ++isource_;
        open_file();
      }
    }

    if(source_)
    {
      seq_index_ = next_index;
      if(isource_ == 0)
        source_->set_next_index(next_index);
      else
        source_->set_next_index(next_index - chained_file_index_[isource_-1]);
    }
  }

protected:
  using BasicChainedDataSource<RADST>::source_;
  using BasicChainedDataSource<RADST>::isource_;
  using BasicChainedDataSource<RADST>::opener_;
  using BasicChainedDataSource<RADST>::seq_index_;

  void add_source_index(RADST* source)
  {
    if(chained_file_index_.empty())
      chained_file_index_.push_back(source->size());
    else
      chained_file_index_.push_back(source->size()+chained_file_index_.back());
  }

  void open_file() override
  {
    BasicChainedDataSource<RADST>::open_file();
    if(source_ and isource_ >= chained_file_index_.size())
    {
      add_source_index(source_);
    }
  }

  std::vector<uint64_t> chained_file_index_;
};

} } } // namespace calin::io::data_source
