
pub fn kernel (s: &'static str) -> i32 {
  println!("hello rust lib {}", s.to_string());
  42
}
