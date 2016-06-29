#include "ds.hpp"
#include <string>

using namespace std;

string _msg_;

#define TEST(msg) do { \
  cout << endl << msg << endl; \
  _msg_ = msg; \
 } while (0)

#define OK do { \
  cout << _msg_ << " ... OK " << endl; \
 } while (0)

int
main (void)
{

  TEST ("_list_node constructor");
    DS::_list_node<int> a (100);
    DS::_list_node<int> b (200);
  OK;

  TEST ("_list_node setnext and dump");
    a.setnext(&b);
    a.dump();
  OK;

  TEST ("_list_node setprev and dump");
    b.setprev(&a);
    b.dump();
  OK;

  TEST ("list constructor and dump");
    DS::list<int> l;
    l.dump(1);
  OK;

  TEST ("list get");
    assert(l.get(0) == NULL);
    assert(l.get(1) == NULL);
  OK;

  TEST ("list _checklink");
    assert(l._checklink() == 0);
  OK;

  TEST ("list insert");
    l.insert(1);
  OK;
  
  TEST ("list dump and get");
    l.dump(1);
    cout << l.get(0) << endl;;
    cout << l.get(1) << endl;;
  OK;

  TEST ("list insert more and dump");
    l.insert(2);
    l.dump(1);
    l.insert(3);
    l.dump(1);
    l.insert(4, 0);
    l.dump(1);
    l.insert(5, 2);
    l.dump(1);
  OK;

  TEST ("list append");
    l.append(6);
    l.append(7);
    l.append(8, 0);
    l.dump(1);
  OK;

  TEST ("list remove");
    l.remove(1);
    l.dump(1);
  OK;

  TEST ("list purge");
    l.purge();
    l.dump();
  OK;

  return 0;
}
