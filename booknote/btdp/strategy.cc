
// -std=c++11

#include <iostream>
#include <string>
#include <cmath>
#include <sstream>

using namespace std;

class CashSuper {
public:
  double acceptCash (double money) { return money; }
};

class CashNormal : public CashSuper {
public:
  double acceptCash (double money) {
    return money;
  }
};

class CashRebate : public CashSuper {
private:
  double rebate_ = 1.0;
public:
  CashRebate (string rebate) {
    istringstream iss (rebate);
    iss >> this->rebate_;
  }
  double acceptCash (double money) {
    return money * rebate_;
  }
};

class CashReturn : public CashSuper {
private:
  double mcondition_ = 0.0;
  double mreturn_    = 0.0;
public:
  CashReturn (string mcondition, string mreturn) {
    istringstream iss1 (mcondition);
    istringstream iss2 (mreturn);
    iss1 >> this->mcondition_;
    iss2 >> this->mreturn_;
  }
  double acceptCash (double money) {
    double ret = money;
    if (ret >= mcondition_)
      ret = ret - floor(ret/mcondition_) * mreturn_;
    return ret;
  }
};

class CashContext {
  // strategy and simple-factory combined
private:
  CashSuper * cs_;
public:
  CashContext (string type) {
    if (type == "Normal") {
      this->cs_ = new CashNormal();
    } else if (type == "300R100") {
      this->cs_ = new CashReturn("300", "100");
    } else if (type == "Discount0.8") {
      this->cs_ = new CashRebate("0.8");
    } else {
      return;
    }
  }
  double GetResult (double money) {
    return cs_->acceptCash (money);
  }
};

int
main (void)
{
  double price = 320.;
  CashContext * cs1 = new CashContext("Normal");
  cout << cs1->GetResult(price) << endl;
  CashContext * cs2 = new CashContext("Discount0.8");
  cout << cs2->GetResult(price) << endl;
  CashContext * cs3 = new CashContext("300R100");
  cout << cs3->GetResult(price) << endl;
  return 0;
}
//FIXME: there is bug but I don't know where it is.
