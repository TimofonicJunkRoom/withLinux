
macro_rules! info {
  ($msg:expr) => {{
    println!("\x1b[32;1mI {}:{}] {}\x1b[m", file!(), line!(), $msg);
  }};
}

fn main () {
  println! ("Reference: file:///home/schroot/sid/usr/share/doc/rust-doc/html/book/README.html");

  //guess_game();
  let _guess_game = guess_game; // _XXX avoids "unused_variable" warning
  //panic!(); // crash the program
  syntax();
  effective_rust();
}

fn guess_game () {
  use std::io;
  use std::cmp::Ordering;

  let ans: u32 = 50;
  loop {
    println! ("Guess the number. Input your guess: ");
    let mut guess = String::new();
    //io::stdin().read_line(&mut guess); // this will cause warning.
    io::stdin().read_line(&mut guess).expect("readline failed"); // .expect of io::Result
    //let guess: u32 = guess.trim().parse().expect("Invalid input"); // crash when encountered invalid input
    let guess: u32 = match guess.trim().parse() {
      Ok(num) => num,
      Err(_)  => continue,
    };
    print! ("Your guess is {}:", guess);
    match guess.cmp(&ans) {
      Ordering::Less    => println!("too small"),
      Ordering::Greater => println!("too large"),
      Ordering::Equal   => { println!("You win"); break; },
    }
  }
}

#[allow(unused_variables)]
#[allow(unused_assignments)]
#[allow(dead_code)]
#[allow(unused_mut)]
#[allow(non_shorthand_field_patterns)]
fn syntax () {
  { // 4.1 variable binding
    let x = 5; // general
    let (x, y) = (1, 2); // pattern
    let x: i32 = 5; // type annotation
    let mut x = 5; x = 10; // mutability
    { let y: i32 = 1; } // scope
  }
  { // 4.2 functions
    fn print_number (x: i32) {
      println! ("x = {}", x);
    }
    print_number (30);
    fn print_sum (x: i32, y: i32) {
      println! ("x+y={}", x+y);
    }
    print_sum (2, 3);
    fn plusone (x: i32) -> i32 {
      //return x+1; <- this is considered poor style!
      x+1
    }
    println! ("{}", plusone(1));
    fn diverges() -> ! { // diverging function
      panic!("This function never returns!");
    }
    let f: fn(i32) -> i32 = plusone; // function pointer
    let y = f(5);
  }
  { // 4.3 Primitive types
    let x = true;
    let y: bool = false;
    let x = 'x'; // char
    let alpha = 'α'; // char is 4-Byte unicode, not single byte!
    // let β = 0.1; // Oops, no way.
    // numerical types: i8 i16 i32 i64 u8 u16 u32 u64 isize usize f32 f64
    let a = [ 1, 2, 3 ]; // a: [i32, 3]
    let mut m = [ 1, 2, 3 ]; // m: [i32, 3]
    let a = [0; 20]; // shorthand, [0, 0, ..., 0] size 1x20
    println! ("length of a is {}", a.len());
    let names = ["Graydon", "Brian", "Niko"]; // names: [&str; 3]
    println!("The second name is: {}", names[1]);
    let a = [0, 1, 2, 3, 4, 5];
    let complete = &a[..]; // full slice
    let middle = &a[1..5]; // [1,2,3,4]
    // str
    let x = (1, "hello"); // tuple
    let x: (i32, &str) = (1, "hello");
    let mut x = (1, 2); // x: (i32, i32)
    let y = (2, 3); // y: (i32, i32)
    x = y; // assign one tuple into another
    let (x, y, z) = (1, 2, 3); // accessing fields in tuple
    println!("x is {}", x);
    let x = (0,); // one-element tuple
    let tuple = (1, 2, 3);
    let x = tuple.0; // instead of tuple[0], different from array
    let y = tuple.1;
    let z = tuple.2;
    fn foo(x: i32) -> i32 { x }
    let x: fn(i32) -> i32 = foo; // function has type too
  }
  { // 4.4 comments
    // this is line comment
    // /// this is doc comment for the item following it, markdown is supported
    // //! this is doc comment for the item enclosed, with markdown being supported
    let stub = "stub";
  }
  { // 4.5 if
    let x = 5;
    if x == 5 {
      println!("x is five");
    } else if x == 6 {
      println!("x is six");
    } else {
      println!("x is not five");
    }
    let y = if x == 5 { 10 } else { 15 };
  }
  { // 4.6 loop
    loop { // infinite loop
      println!("entered infinite loop");
      break;
    }
    let mut x: i32 = 1;
    let mut sum: i32 = 0;
    while x <= 100 { // while
      sum += x;
      x +=1 ;
    }
    println!("1+2+...+100={}", sum);
    let mut sum: i32 = 0;
    for i in 0..101 { // 0..101 then be converted into an iterator
      sum += i;
    }
    println!("1+2+...+100={}", sum);
    let mut sum: i32 = 0;
    for (i, j) in (1..11).enumerate() { // enumerate on range
      sum += j;
      println!("{}: 1+...+{}={}", i, j, sum);
    }
    let lines = "hello\nworld".lines();
    for (linenumber, line) in lines.enumerate() { // enumerate on iterators
      println!("{}: {}", linenumber, line);
    }
    // break and continue are available for rust
    'outer: for x in 0..6 { // loop labels
      'inner: for y in 0..6 { // loop labels
        if x % 2 == 0 { continue 'outer; } // continues the loop over x
        if y % 2 == 0 { continue 'inner; } // continues the loop over y
        println!("x: {}, y: {}", x, y);
      }
    }
  }
  { // 4.7 vectors
    let v = vec![1,2,3,4,5]; // v: Vec<i32>
    let v = vec!(1,2,3,4,5); // v: Vec<i32>
    let mut v = vec![0; 10]; // zeros(1, 10);
    println!("v[0]={}", v[0]);
    let idx: usize = 0; // vector index must be usize type, and i32 type won't work
    println!("v[{}]={}", idx, v[idx]);
    for i in 9..12 { // handling errors
      match v.get(i) {
        Some(x) => println!("{}", x),
        None    => println!("idx out of bound"),
      }
    }
    for i in &v { // iterate over a reference to vector
      print!("{} ", i);
    }
    println!("");
    for i in &mut v { // mutable reference
      *i += 1;
      print!("{} ", i);
    }
    println!("");
    for i in v { // take ownership of the vector and its elements
      print!("{} ", i);
    }
    println!("");
  }
  { // 4.8 ownership | * Key feature of Rust
    // FIXME
  }
  { // 4.9 borrowing | * Key feature of Rust
    // FIXME
  }
  { // 4.10 lifetimes | * Key features of Rust
    // FIXME
  }
  { // 4.11 mutability
    let x = 5; // x = 6; // will trigger a failure.
    let mut x = 5; x = 6; // ok
    //let y: &mut i32 = &mut x; // note, y is immutable here
    let mut y = &mut x; // now y is mutable
    let (mut x, y) = (8, 9); // `mut` is a part of pattern
    fn foo(mut x: i32) { ; }
    // * Interior vs. Exterior Mutability
    use std::sync::Arc;
    let x = Arc::new(5);
    let y = x.clone(); // exterior mutability
    use std::cell::RefCell;
    let x = RefCell::new(42);
    let y = x.borrow_mut(); // interior mutability
    // * Field-level mutability
    struct Point {
      x: f32, // can't be mut x: f32
      y: f32,
    }
    let mut a = Point { x: 5., y: 6. };
    a.x = 10.; // this is ok
    use std::cell::Cell;
    struct PointX {
      x: f32,
      y: Cell<f32>, // emulate field-level mutability
    }
    let point = PointX { x: 5., y: Cell::new(0.) };
    point.y.set(7.);
    println!("y: {:?}", point.y); // we changed its value within immutable struct
  }
  { // 4.12 structs
    struct Point { x: i32, y: i32, }
    let mut point = Point { x: 0, y: 0 };
    println!("The origin is at ({}, {})", point.x, point.y);
    struct PointRef<'a> { x: &'a mut i32, y: &'a mut i32 }
    {
      let r = PointRef { x: &mut point.x, y: &mut point.y };
      *r.x = 5; *r.y = 6;
    }
    assert_eq!(5, point.x); assert_eq!(6, point.y);
    // update syntax for structs
    struct Point3d { x: i32, y: i32, z: i32, }
    let mut point = Point3d { x: 0, y: 0, z: 0 };
    point = Point3d { y: 1, .. point }; // update y to 1
    // tuple structs
    struct Color (u8, u8, u8);
    let background = Color(255, 255, 255);
    struct Inches(i32);
    let length = Inches(10);
    let Inches(integer_length) = length;
    println!("length is {} inches", integer_length);
    struct Electron; // actually a empty tuple ()
    let x = Electron;
  }
  { // 4.13 Enums
    enum Message {
      Quit,
      ChangeColor(i32, i32, i32),
      Move { x: i32, y: i32 },
      Write(String),
    }
    let x: Message = Message::Move { x: 3, y: 4 };
    enum BoardGameTurn {
      Move { squares: i32 },
      Pass,
    }
    let y: BoardGameTurn = BoardGameTurn::Move { squares: 1 };
    //fn process_color_change(msg: Message) {
    //  let Message::ChangeColor(r, g, b) = msg; // compile-time error
    //} // use `match` to break this limitation
    let m = Message::Write("hello world".to_string()); // enum constructor
    fn foo (x: String) -> Message { Message::Write(x) }
    let x = foo("hello world".to_string()); // same as the above enum constructor
    let v = vec!["hello".to_string(), "world".to_string()];
    let v1: Vec<Message> = v.into_iter().map(Message::Write).collect();
  }
  { // 4.14 match
    let x = 5;
    match x {
      1 => println!("one"),
      5 => println!("five"),
      _ => println!("unknown"), // compiler fails if this being missing.
    }
    let y = match x {
      1 => "one",
      5 => "five",
      _ => "unknown",
    };
    // matching enums
    enum Message {
      Quit,
      ChangeColor(i32, i32, i32),
      Move { x: i32, y: i32 },
      Write(String),
    }
    fn quit() { /* ... */ }
    fn change_color(r: i32, g: i32, b: i32) { /* ... */ }
    fn move_cursor(x: i32, y: i32) { /* ... */ }
    fn process_message(msg: Message) {
      match msg {
        Message::Quit => quit(),
        Message::ChangeColor(r, g, b) => change_color(r, g, b),
        Message::Move { x: x, y: y } => move_cursor(x, y),
        Message::Write(s) => println!("{}", s),
      };
    }
  }
  { // 4.15 patterns
    let x = 5;
    match x {
      1 | 2 => println!("one or two"), // multiple patterns
      5 => println!("five"),
      10 ... 20 => println!("ten through twenty"), // range
      // 'a' ... 'z' => println!("matched a alphabet?"),
      e @ 30 ... 40 => println!("we can bind the range element like this {}", e),
      e @ 1 ... 5 | e @ 8 ... 10 => println!("got a range element {}", e),
      _ => println!("unknown"), // compiler fails if this being missing.
    }
    // destructuring
    struct Point { x: i32, y: i32, }
    let origin = Point { x: 0, y: 0 };
    match origin {
      Point { x, y } => println!("({},{})", x, y),
    }
    match origin {
      Point { x: x1, y: y1 } => println!("({},{})", x1, y1),
    }
    match origin {
      Point { x, .. } => println!("x is {}", x),
    }
    match origin {
      Point { y, .. } => println!("y is {}", y),
    }
    // let (x, _, z) = coordinate(); // _ is used to discard a part of struct here
    let _ = String::from("  hello  ").trim(); // oh drop the return value immediately
    enum OptTuple {
      Value ( f32, f32, f32 ),
      Missing,
    }
    let x : OptTuple = OptTuple::Value ( 0., 1., 2. );
    match x {
      OptTuple::Value(..) => println!("Got a tuple!"),
      OptTuple::Missing => println!("No such luck."),
    }
    // ref and ref mut
    let x = 5;
    match x {
      ref r => println!("Got a reference to {}", r),
    }
    let mut x = 5;
    match x {
      ref mut mr => println!("Got a mutable reference to {}", mr),
    }
    // complex structure matching
    #[derive(Debug)]
    struct Person {
      name: Option<String>,
    }
    let name = "Steve".to_string();
    let x: Option<Person> = Some(Person { name: Some(name) });
    match x {
      Some(Person { name: ref a @ Some(_), .. }) => println!("{:?}", a),
      _ => {}
    }
    // match guards
    let x = 4;
    let y = false;
    match x {
      //ref r if * r < 0 => println!("got {} < 0", r),
      4 | 5 if y => println!("yes"),
      _ => println!("no"),
    }
  }
  { // 4.16 method syntax
    struct Circle { x: f64, y: f64, radius: f64, }
    impl Circle {
      fn area(&self) -> f64 {
        std::f64::consts::PI * (self.radius * self.radius)
      }
      fn grow(&self, inc: f64) -> Circle {
        Circle { x: self.x, y: self.y, radius: self.radius + inc }
      }
      fn new(x: f64, y: f64, radius: f64) -> Circle { // associated function
        Circle { x: x, y: y, radius: radius, }
      }
    }
    impl Circle { // impl can be split up
      fn reference(&self) {
        println!("taking self by reference!");
      }
      fn mutable_reference(&mut self) {
        println!("taking self by mutable reference!");
      }
      fn takes_ownership(self) {
        println!("taking ownership of self!");
      }
    }
    let c = Circle { x: 0.0, y: 0.0, radius: 2.0 };
    println!("{}", c.area());
    let b = c.grow(2.0);
    println!("{}", b.area());
    let d = Circle::new(0., 1., 2.);
    // there is no method overloading or default argument in rust
    struct CircleBuilder {
      x: f64,
      y: f64,
      radius: f64,
    }
    impl CircleBuilder {
      fn new() -> CircleBuilder {
        CircleBuilder { x: 0.0, y: 0.0, radius: 1.0, }
      }
      fn x(&mut self, coordinate: f64) -> &mut CircleBuilder {
        self.x = coordinate;
        self
      }
      fn y(&mut self, coordinate: f64) -> &mut CircleBuilder {
        self.y = coordinate;
        self
      }
      fn radius(&mut self, radius: f64) -> &mut CircleBuilder {
        self.radius = radius;
        self
      }
      fn finalize(&self) -> Circle {
        Circle { x: self.x, y: self.y, radius: self.radius }
      }
    }
    let c = CircleBuilder::new()
                .x(1.0)
                .y(2.0)
                .radius(2.0)
                .finalize();
    println!("area: {}", c.area());
    println!("x: {}", c.x);
    println!("y: {}", c.y);
  }
  { // 4.17 strings
    // FIXME
  }
  {
    // FIXME
  }
  { // 4.36 unsafe
    unsafe fn dangerous_function () {} // unsafe function
    unsafe { } // unsafe block
    unsafe trait Scary { } // unsafe trait
    unsafe impl Scary for i32 { } // unsafe impl
    // if rust program segfaults, the cause is related to somthing marked "unsafe".
    // Unsafe makes you able to
    // * access or update a static mut
    // * dereference a raw pointer
    // * call unsafe functions
  }
  info!("syntax() done");
} // fn syntax()

extern crate libc;
use libc::pid_t;

#[link(name = "c")] // you can omit this
extern {
  /* pid_t getpid(void); */
  fn getpid() -> pid_t;
}

fn effective_rust () {
  { // 5.9 FFI
    unsafe {
      let pid = getpid();
      println!("my pid is {}", pid);
    }
    // TODO
  }
  info!("effective_rust() done");
} // fn effective_rust()

/* test stuff */
#[test]
fn it_works() {
  assert_eq!(1, 1);
}
#[test]
#[should_panic]
fn it_doesnt_work() {
  assert!(false);
}
#[test]
#[should_panic(expected = "assertion failed")]
fn it_doesnt_work_2() {
    assert_eq!("Hello", "world");
}
