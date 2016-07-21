Friends, Exceptions
===

* friends
```
classs TV {
private:
  int state, volume, maxchannel, mode, input;
public:
  friend class Remote; // remote can access private part of TV.
  TV(int s = Off, int mc = 125) : state(s), volume(5),
    maxchannel(mc), channel(2), mode(Cable), input(TV) {}
  ...
};

class Remote {
private:
  int mode; // controls TV or DVD
public:
  Remote(int m = TV::Tv) : mode(m) {}
  ...
}; 
```

* nested classes
```
class Team
{
public:
class Coach { ... };
...
};
```

* Exceptions
TODO
