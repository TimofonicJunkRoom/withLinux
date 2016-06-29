/**
 @file ds.hpp
 @author Copyright (C) 2016 Zhou Mo
 @reference GNU GCC: G++ STL Souce Code (stl_list.h)
 */
#ifndef _DS_H
#define _DS_H

#include <iostream>
#include <cassert>
#include <string>

namespace DS {

  using std::cout;
  using std::endl;

  bool debug = 0;

// --- class _list_node ---

template <typename Tp>
class _list_node {
  private:
    Tp data_;
    _list_node<Tp> * prev_;
    _list_node<Tp> * next_;
  public:
    _list_node (Tp data, _list_node<Tp> * next = NULL, _list_node<Tp> * prev = NULL);
    void dump (void) const;
    _list_node<Tp> * prev() const;
    _list_node<Tp> * next() const;
    void setprev (_list_node<Tp> * newprev);
    void setnext (_list_node<Tp> * newnext);
    Tp data (void) const;
}; // class _list_node

template <typename Tp>
_list_node<Tp>::_list_node (Tp data,
       _list_node<Tp> * next,
       _list_node<Tp> * prev)
{
  data_ = data;
  prev_ = prev;
  next_ = next;
}

template <typename Tp> void
_list_node<Tp>::dump(void) const
{
  std::cout << this << " _list_node: data " << data_ << " prev " << prev_
    << " next " << next_ << std::endl;
}

template <typename Tp> _list_node<Tp> *
_list_node<Tp>::prev(void) const
{ return prev_; }

template <typename Tp> _list_node<Tp> *
_list_node<Tp>::next(void) const
{ return next_; }

template <typename Tp> void
_list_node<Tp>::setprev (_list_node<Tp> * newprev)
{ prev_ = newprev; }

template <typename Tp> void
_list_node<Tp>::setnext (_list_node<Tp> * newnext)
{ next_ = newnext; }

template <typename Tp> Tp
_list_node<Tp>::data (void) const
{ return data_; }

// --- class list ---

template <typename Tp>
class list {
  private:
    _list_node<Tp> * head_;
    _list_node<Tp> * tail_;
    size_t size_;
  public:
    list (void);
    size_t size (void) const;
    size_t _checklink (void) const;
    void dump (bool all = 0) const;
    _list_node<Tp> * get (long index = 0) const; // index starts from 0
    _list_node<Tp> * insert (Tp data, long index = 0);
    _list_node<Tp> * append (Tp data);
    _list_node<Tp> * append (Tp data, long index);
    void remove (long index);
    void purge (void); // remove all nodes
    _list_node<Tp> * head (void) const;
    _list_node<Tp> * tail (void) const;
}; // class list

template <typename Tp>
list<Tp>::list (void)
{
  head_ = NULL;
  tail_ = NULL;
  size_ = 0;
}

template <typename Tp> size_t
list<Tp>::size (void) const
{ return size_; }

template <typename Tp> size_t
list<Tp>::_checklink (void) const
{
  if (debug) cout << "(checklink)" << endl;
  if (size_ == 0) return size_;
  size_t _count_forward = 0;
  size_t _count_backward = 0;
  _list_node<Tp> * cursor = NULL;
  // forward count
  if (debug) cout << "(forward)" << endl;
  cursor = head_;
  while (cursor != NULL) {
    ++_count_forward;
    cursor = cursor->next();
  }
  // backward count
  if (debug) cout << "(backward)" << endl;
  cursor = tail_;
  while (cursor != NULL) {
    ++_count_backward;
    cursor = cursor->prev();
  }
  if (debug) cout << "(ok)" << endl;
  assert(_count_forward == _count_backward);
  assert(_count_forward == size_);
  return size_;
}

template <typename Tp> _list_node<Tp> * 
list<Tp>::get (long index) const
{
  // process index and check
  size_t target = (size_t)(index >= 0 ? index :
    ((size_ + index > 0) ? size_ + index : 0));
  if (debug) cout << "(get idnex " << index << " target " << target << ")" << endl;
  if (size_ == 0) {
    cout << "empty list, ignore get at index " << target << endl;
    return NULL;
  }
  if (target >= size_) {
    std::cout << "warn: list index " << target << " out of bounds" << endl;
    return NULL;
  }
  // find the target starting from head_
  _list_node<Tp> * cursor = head_;
  size_t counter = 0;
  while (cursor != NULL && counter != target) {
    cursor = cursor->next();
    ++counter;
  } 
  return cursor;
}

template <typename Tp> void
list<Tp>::dump(bool all) const
{
  std::cout << this << " list: size " << size_ << " head " << head_
    << " tail " << tail_ << std::endl;
  if (all) {
    _list_node<Tp> * cursor = head_;
    while (cursor != NULL) {
      cursor->dump();
      cursor = cursor->next();
    }
  }
}

template <typename Tp> _list_node<Tp> *
list<Tp>::insert (Tp data, long index)
{
  _list_node<Tp> * node = new _list_node<Tp> (data);
  if (size_ == 0) {
    if (debug) cout << "(empty list insert)" << endl;
    head_ = node;
    tail_ = node;
    ++size_;
  } else {
    if (debug) cout << "(checklink)" << endl;
    _checklink();
    _list_node<Tp> * cursor = get(index);
    // surgery forward chain and backward chain
    if (debug) cout << "(surgery)" << endl;
    node->setprev(cursor->prev());
    node->setnext(cursor);
    if (cursor->prev() != NULL) cursor->prev()->setnext(node);
    cursor->setprev(node);
    // update list
    if (debug) cout << "(update)" << endl;
    ++size_;
    if (node->prev() == NULL) head_ = node;
    if (node->next() == NULL) tail_ = node;
  }
  return node;
}

template <typename Tp> _list_node<Tp> *
list<Tp>::append (Tp data, long index)
{
  if (debug) cout << "(append to index " << index << ")" << endl;
  _list_node<Tp> * node = new _list_node<Tp> (data);
  if (size_ == 0) {
    head_ = node;
    tail_ = node;
    ++size_;
  } else {
    _checklink();
    _list_node<Tp> * cursor = get(index);
    // surgery forward chain and backward chain
    node->setprev(cursor);
    node->setnext(cursor->next());
    if (cursor->next() != NULL) cursor->next()->setprev(node);
    cursor->setnext(node);
    // update list
    ++size_;
    if (node->prev() == NULL) head_ = node;
    if (node->next() == NULL) tail_ = node;
  }
  return node;
}

template <typename Tp> _list_node<Tp> *
list<Tp>::append (Tp data)
{ return append(data, size_-1); }

template <typename Tp> void
list<Tp>::remove (long index)
{
  _list_node<Tp> * cursor = get(index);
  if (cursor == NULL) {
    std::cout << "warn: remove nothing from empty list" << std::endl;
    return;
  }
  // surgery and update
  --size_;
  if (cursor->prev() == NULL) head_ = cursor->next();
  if (cursor->next() == NULL) tail_ = cursor->prev();
  if (cursor->prev() != NULL) cursor->prev()->setnext(cursor->next());
  if (cursor->next() != NULL) cursor->next()->setprev(cursor->prev());
  delete cursor;
}

template <typename Tp> void
list<Tp>::purge (void)
{
  _list_node<Tp> * cursor = head_;
  if (cursor == NULL) return;
  while (cursor != NULL) {
    _list_node<Tp> * toberemoved = cursor;
    cursor = cursor->next();
    delete toberemoved;
  }
  size_ = 0;
  head_ = NULL;
  tail_ = NULL;
}

template <typename Tp> _list_node<Tp> *
list<Tp>::head (void) const
{ return head_; }

template <typename Tp> _list_node<Tp> *
list<Tp>::tail (void) const
{ return tail_; }

// --- class stack --- stimulate with list

template <typename Tp>
class stack {
  private:
    list<Tp> s_;
  public:
    stack (void);
    _list_node<Tp> * push (Tp data);
    Tp pop (void);
    size_t size(void) const;
    _list_node<Tp> * top (void) const;
    void dump (bool all = 0) const;
}; // class stack

template <typename Tp>
stack<Tp>::stack (void)
{ list<Tp> s_; }

template <typename Tp> _list_node<Tp> *
stack<Tp>::push (Tp data)
{ return s_.append(data); }

template <typename Tp> Tp
stack<Tp>::pop (void)
{
  if (s_.size() == 0) {
    std::cout << "warn: pop nothing from empty stack" << std::endl;
    return (Tp) NULL;
  }
  Tp ret = s_.tail()->data();
  s_.remove(-1);
  return ret;
}

template <typename Tp> size_t
stack<Tp>::size (void) const
{ return s_.size(); }

template <typename Tp> _list_node<Tp> *
stack<Tp>::top (void) const
{ return s_.tail(); }

template <typename Tp> void
stack<Tp>::dump (bool all) const
{ s_.dump(all); }

// --- class queue

template <typename Tp>
class queue {
  private:
    list<Tp> q_;
  public:
    queue (void);
    _list_node<Tp> * push (Tp data);
    Tp pop (void);
    size_t size (void) const;
    void dump (bool all = 0) const;
}; // class queue

template <typename Tp>
queue<Tp>::queue (void)
{ list<Tp> q_; }

template <typename Tp> _list_node<Tp> *
queue<Tp>::push (Tp data)
{ return q_.append(data); }

template <typename Tp> Tp
queue<Tp>::pop (void)
{
  Tp ret = q_.head()->data();
  q_.remove(0);
  return ret;
}

template <typename Tp> size_t
queue<Tp>::size (void) const
{ return q_.size(); }

template <typename Tp> void
queue<Tp>::dump (bool all) const
{ q_.dump(all); }

// --- class _btree_node ---

template <typename Tp>
class _btree_node {
  private:
    Tp data_;
    _btree_node<Tp> * left_;
    _btree_node<Tp> * right_;
  public:
    _btree_node (Tp data = (Tp) NULL);
    void dump (int indent = 0) const; // dfs, parent left right order
    void pdump (int indent = 0) const; // pretty dump, dfs, parent left right order
    Tp data (void);
    _btree_node<Tp> * left  (void);
    _btree_node<Tp> * right (void);
    void setleft (_btree_node<Tp> * node);
    void setright (_btree_node<Tp> * node);
    bool isleaf (void) {
      return (left_ == NULL && right_ == NULL);
    }
}; // class _btree_node

template <typename Tp>
_btree_node<Tp>::_btree_node (Tp data)
{
  data_ = data;
  left_ = NULL;
  right_= NULL;
}

template <typename Tp> void
_btree_node<Tp>::dump (int indent) const
{
  for (int i = 0; i < indent; i++) std::cout << " ";
  std::cout << this << " _btree_node: data " << data_ << " with left "
    << left_ << " and right " << right_ << std::endl;
  if (left_ != NULL) left_->dump(indent+2);
  if (right_ != NULL) right_->dump(indent+2);
}

template <typename Tp> void
_btree_node<Tp>::pdump (int indent) const
{
  for (int i = 0; i < indent; i++) std::cout << " ";
  if (left_ == NULL && right_ == NULL) {
    std::cout << "(" << this << " " << data_ << ")" << std::endl;
  } else {
    std::cout << "(" << this << " " << data_ << std::endl;
    if (left_  != NULL) left_ ->pdump(indent+2);
    if (right_ != NULL) right_->pdump(indent+2);
    for (int i = 0; i < indent; i++) std::cout << " ";
    std::cout << ")" << std::endl;
  }
}

template <typename Tp> void
_btree_node<Tp>::setleft (_btree_node<Tp> * node)
{ left_ = node; }

template <typename Tp> void
_btree_node<Tp>::setright (_btree_node<Tp> * node)
{ right_ = node; }

// --- class btree ---

template <typename Tp>
class btree {
  private:
    _btree_node<Tp> root_;
  public:
    btree (void) {
      _btree_node<Tp> root_;
    }
    btree (_btree_node<Tp> * node) {
      root_ = node;
    }
    _btree_node<Tp> * root (void) const {
      return root_;
    }
    void dump (void) const {
      
    }
    size_t size (void);
}; // class btree

} // namespace DS

#endif /* _DS_H */
