// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: net.proto

#ifndef PROTOBUF_net_2eproto__INCLUDED
#define PROTOBUF_net_2eproto__INCLUDED

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 2006000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 2006001 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)

// Internal implementation detail -- do not call these.
void  protobuf_AddDesc_net_2eproto();
void protobuf_AssignDesc_net_2eproto();
void protobuf_ShutdownFile_net_2eproto();

class NetParamter;

// ===================================================================

class NetParamter : public ::google::protobuf::Message {
 public:
  NetParamter();
  virtual ~NetParamter();

  NetParamter(const NetParamter& from);

  inline NetParamter& operator=(const NetParamter& from) {
    CopyFrom(from);
    return *this;
  }

  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _unknown_fields_;
  }

  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return &_unknown_fields_;
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const NetParamter& default_instance();

  void Swap(NetParamter* other);

  // implements Message ----------------------------------------------

  NetParamter* New() const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const NetParamter& from);
  void MergeFrom(const NetParamter& from);
  void Clear();
  bool IsInitialized() const;

  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const;
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  public:
  ::google::protobuf::Metadata GetMetadata() const;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // required fixed32 word_dim = 1;
  inline bool has_word_dim() const;
  inline void clear_word_dim();
  static const int kWordDimFieldNumber = 1;
  inline ::google::protobuf::uint32 word_dim() const;
  inline void set_word_dim(::google::protobuf::uint32 value);

  // required fixed32 hidden_dim = 2;
  inline bool has_hidden_dim() const;
  inline void clear_hidden_dim();
  static const int kHiddenDimFieldNumber = 2;
  inline ::google::protobuf::uint32 hidden_dim() const;
  inline void set_hidden_dim(::google::protobuf::uint32 value);

  // required fixed32 bptt_truncate = 3;
  inline bool has_bptt_truncate() const;
  inline void clear_bptt_truncate();
  static const int kBpttTruncateFieldNumber = 3;
  inline ::google::protobuf::uint32 bptt_truncate() const;
  inline void set_bptt_truncate(::google::protobuf::uint32 value);

  // required fixed32 epoch = 4;
  inline bool has_epoch() const;
  inline void clear_epoch();
  static const int kEpochFieldNumber = 4;
  inline ::google::protobuf::uint32 epoch() const;
  inline void set_epoch(::google::protobuf::uint32 value);

  // optional double learingRate = 5;
  inline bool has_learingrate() const;
  inline void clear_learingrate();
  static const int kLearingRateFieldNumber = 5;
  inline double learingrate() const;
  inline void set_learingrate(double value);

  // repeated double U = 6 [packed = true];
  inline int u_size() const;
  inline void clear_u();
  static const int kUFieldNumber = 6;
  inline double u(int index) const;
  inline void set_u(int index, double value);
  inline void add_u(double value);
  inline const ::google::protobuf::RepeatedField< double >&
      u() const;
  inline ::google::protobuf::RepeatedField< double >*
      mutable_u();

  // repeated double W = 7 [packed = true];
  inline int w_size() const;
  inline void clear_w();
  static const int kWFieldNumber = 7;
  inline double w(int index) const;
  inline void set_w(int index, double value);
  inline void add_w(double value);
  inline const ::google::protobuf::RepeatedField< double >&
      w() const;
  inline ::google::protobuf::RepeatedField< double >*
      mutable_w();

  // repeated double V = 8 [packed = true];
  inline int v_size() const;
  inline void clear_v();
  static const int kVFieldNumber = 8;
  inline double v(int index) const;
  inline void set_v(int index, double value);
  inline void add_v(double value);
  inline const ::google::protobuf::RepeatedField< double >&
      v() const;
  inline ::google::protobuf::RepeatedField< double >*
      mutable_v();

  // @@protoc_insertion_point(class_scope:NetParamter)
 private:
  inline void set_has_word_dim();
  inline void clear_has_word_dim();
  inline void set_has_hidden_dim();
  inline void clear_has_hidden_dim();
  inline void set_has_bptt_truncate();
  inline void clear_has_bptt_truncate();
  inline void set_has_epoch();
  inline void clear_has_epoch();
  inline void set_has_learingrate();
  inline void clear_has_learingrate();

  ::google::protobuf::UnknownFieldSet _unknown_fields_;

  ::google::protobuf::uint32 _has_bits_[1];
  mutable int _cached_size_;
  ::google::protobuf::uint32 word_dim_;
  ::google::protobuf::uint32 hidden_dim_;
  ::google::protobuf::uint32 bptt_truncate_;
  ::google::protobuf::uint32 epoch_;
  double learingrate_;
  ::google::protobuf::RepeatedField< double > u_;
  mutable int _u_cached_byte_size_;
  ::google::protobuf::RepeatedField< double > w_;
  mutable int _w_cached_byte_size_;
  ::google::protobuf::RepeatedField< double > v_;
  mutable int _v_cached_byte_size_;
  friend void  protobuf_AddDesc_net_2eproto();
  friend void protobuf_AssignDesc_net_2eproto();
  friend void protobuf_ShutdownFile_net_2eproto();

  void InitAsDefaultInstance();
  static NetParamter* default_instance_;
};
// ===================================================================


// ===================================================================

// NetParamter

// required fixed32 word_dim = 1;
inline bool NetParamter::has_word_dim() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void NetParamter::set_has_word_dim() {
  _has_bits_[0] |= 0x00000001u;
}
inline void NetParamter::clear_has_word_dim() {
  _has_bits_[0] &= ~0x00000001u;
}
inline void NetParamter::clear_word_dim() {
  word_dim_ = 0u;
  clear_has_word_dim();
}
inline ::google::protobuf::uint32 NetParamter::word_dim() const {
  // @@protoc_insertion_point(field_get:NetParamter.word_dim)
  return word_dim_;
}
inline void NetParamter::set_word_dim(::google::protobuf::uint32 value) {
  set_has_word_dim();
  word_dim_ = value;
  // @@protoc_insertion_point(field_set:NetParamter.word_dim)
}

// required fixed32 hidden_dim = 2;
inline bool NetParamter::has_hidden_dim() const {
  return (_has_bits_[0] & 0x00000002u) != 0;
}
inline void NetParamter::set_has_hidden_dim() {
  _has_bits_[0] |= 0x00000002u;
}
inline void NetParamter::clear_has_hidden_dim() {
  _has_bits_[0] &= ~0x00000002u;
}
inline void NetParamter::clear_hidden_dim() {
  hidden_dim_ = 0u;
  clear_has_hidden_dim();
}
inline ::google::protobuf::uint32 NetParamter::hidden_dim() const {
  // @@protoc_insertion_point(field_get:NetParamter.hidden_dim)
  return hidden_dim_;
}
inline void NetParamter::set_hidden_dim(::google::protobuf::uint32 value) {
  set_has_hidden_dim();
  hidden_dim_ = value;
  // @@protoc_insertion_point(field_set:NetParamter.hidden_dim)
}

// required fixed32 bptt_truncate = 3;
inline bool NetParamter::has_bptt_truncate() const {
  return (_has_bits_[0] & 0x00000004u) != 0;
}
inline void NetParamter::set_has_bptt_truncate() {
  _has_bits_[0] |= 0x00000004u;
}
inline void NetParamter::clear_has_bptt_truncate() {
  _has_bits_[0] &= ~0x00000004u;
}
inline void NetParamter::clear_bptt_truncate() {
  bptt_truncate_ = 0u;
  clear_has_bptt_truncate();
}
inline ::google::protobuf::uint32 NetParamter::bptt_truncate() const {
  // @@protoc_insertion_point(field_get:NetParamter.bptt_truncate)
  return bptt_truncate_;
}
inline void NetParamter::set_bptt_truncate(::google::protobuf::uint32 value) {
  set_has_bptt_truncate();
  bptt_truncate_ = value;
  // @@protoc_insertion_point(field_set:NetParamter.bptt_truncate)
}

// required fixed32 epoch = 4;
inline bool NetParamter::has_epoch() const {
  return (_has_bits_[0] & 0x00000008u) != 0;
}
inline void NetParamter::set_has_epoch() {
  _has_bits_[0] |= 0x00000008u;
}
inline void NetParamter::clear_has_epoch() {
  _has_bits_[0] &= ~0x00000008u;
}
inline void NetParamter::clear_epoch() {
  epoch_ = 0u;
  clear_has_epoch();
}
inline ::google::protobuf::uint32 NetParamter::epoch() const {
  // @@protoc_insertion_point(field_get:NetParamter.epoch)
  return epoch_;
}
inline void NetParamter::set_epoch(::google::protobuf::uint32 value) {
  set_has_epoch();
  epoch_ = value;
  // @@protoc_insertion_point(field_set:NetParamter.epoch)
}

// optional double learingRate = 5;
inline bool NetParamter::has_learingrate() const {
  return (_has_bits_[0] & 0x00000010u) != 0;
}
inline void NetParamter::set_has_learingrate() {
  _has_bits_[0] |= 0x00000010u;
}
inline void NetParamter::clear_has_learingrate() {
  _has_bits_[0] &= ~0x00000010u;
}
inline void NetParamter::clear_learingrate() {
  learingrate_ = 0;
  clear_has_learingrate();
}
inline double NetParamter::learingrate() const {
  // @@protoc_insertion_point(field_get:NetParamter.learingRate)
  return learingrate_;
}
inline void NetParamter::set_learingrate(double value) {
  set_has_learingrate();
  learingrate_ = value;
  // @@protoc_insertion_point(field_set:NetParamter.learingRate)
}

// repeated double U = 6 [packed = true];
inline int NetParamter::u_size() const {
  return u_.size();
}
inline void NetParamter::clear_u() {
  u_.Clear();
}
inline double NetParamter::u(int index) const {
  // @@protoc_insertion_point(field_get:NetParamter.U)
  return u_.Get(index);
}
inline void NetParamter::set_u(int index, double value) {
  u_.Set(index, value);
  // @@protoc_insertion_point(field_set:NetParamter.U)
}
inline void NetParamter::add_u(double value) {
  u_.Add(value);
  // @@protoc_insertion_point(field_add:NetParamter.U)
}
inline const ::google::protobuf::RepeatedField< double >&
NetParamter::u() const {
  // @@protoc_insertion_point(field_list:NetParamter.U)
  return u_;
}
inline ::google::protobuf::RepeatedField< double >*
NetParamter::mutable_u() {
  // @@protoc_insertion_point(field_mutable_list:NetParamter.U)
  return &u_;
}

// repeated double W = 7 [packed = true];
inline int NetParamter::w_size() const {
  return w_.size();
}
inline void NetParamter::clear_w() {
  w_.Clear();
}
inline double NetParamter::w(int index) const {
  // @@protoc_insertion_point(field_get:NetParamter.W)
  return w_.Get(index);
}
inline void NetParamter::set_w(int index, double value) {
  w_.Set(index, value);
  // @@protoc_insertion_point(field_set:NetParamter.W)
}
inline void NetParamter::add_w(double value) {
  w_.Add(value);
  // @@protoc_insertion_point(field_add:NetParamter.W)
}
inline const ::google::protobuf::RepeatedField< double >&
NetParamter::w() const {
  // @@protoc_insertion_point(field_list:NetParamter.W)
  return w_;
}
inline ::google::protobuf::RepeatedField< double >*
NetParamter::mutable_w() {
  // @@protoc_insertion_point(field_mutable_list:NetParamter.W)
  return &w_;
}

// repeated double V = 8 [packed = true];
inline int NetParamter::v_size() const {
  return v_.size();
}
inline void NetParamter::clear_v() {
  v_.Clear();
}
inline double NetParamter::v(int index) const {
  // @@protoc_insertion_point(field_get:NetParamter.V)
  return v_.Get(index);
}
inline void NetParamter::set_v(int index, double value) {
  v_.Set(index, value);
  // @@protoc_insertion_point(field_set:NetParamter.V)
}
inline void NetParamter::add_v(double value) {
  v_.Add(value);
  // @@protoc_insertion_point(field_add:NetParamter.V)
}
inline const ::google::protobuf::RepeatedField< double >&
NetParamter::v() const {
  // @@protoc_insertion_point(field_list:NetParamter.V)
  return v_;
}
inline ::google::protobuf::RepeatedField< double >*
NetParamter::mutable_v() {
  // @@protoc_insertion_point(field_mutable_list:NetParamter.V)
  return &v_;
}


// @@protoc_insertion_point(namespace_scope)

#ifndef SWIG
namespace google {
namespace protobuf {


}  // namespace google
}  // namespace protobuf
#endif  // SWIG

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_net_2eproto__INCLUDED
