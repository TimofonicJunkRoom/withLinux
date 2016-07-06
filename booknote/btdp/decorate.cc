
// -std=c++11

#include <iostream>
#include <string>

using namespace std;

class Person {
private:
  string name_;
public:
  Person (string name) { name_ = name; }
  Person () { name_ = ""; }
  void Show (void) {
    cout << "Person name " << name_ << endl;
  }
};

class Finery : public Person {
protected:
  Person person;
public:
  void Decorate (Person p) {
    this->person = p;
  }
  void Show (void) {
    person.Show();
  }
};

class Tshirts : public Finery {
public:
  void Show (void) {
    Finery::Show(); // base.Show()
    cout << "Tshirts" << endl;
  }
};

class Tie : public Finery {
public:
  void Show (void) {
    Finery::Show(); // base.Show()
    cout << "Tie" << endl;
  }
};


int
main (void)
{
  Person * p = new Person("Doe");
  Tshirts * f_ts = new Tshirts();
  Tie * f_ti = new Tie();

  f_ts->Decorate(*p);
  f_ti->Decorate(*f_ts);
  f_ti->Show();

  return 0;
}
// FIXME: there is bug.
