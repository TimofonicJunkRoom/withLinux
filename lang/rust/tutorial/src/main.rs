
fn main () {
  println! ("Reference: file:///home/schroot/sid/usr/share/doc/rust-doc/html/book/README.html");

  //guess_game();
  //panic!(); // crash the program
  syntax();
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
  }
}
