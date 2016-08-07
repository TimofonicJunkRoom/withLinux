
pub fn kernel (s: &'static str) -> i32 {
//pub fn kernel (s: &'static str, size: &usize) -> i32 {
//pub fn kernel (s: &String) -> i32 {
//pub fn kernel (s: &str) -> i32 {
  println!("arg string [{}] len [{}]", s, s.len());
  println!("hello rust lib: {:?}", s);
  unsafe{
    // s.as_ptr() -> *const u8
    let s_ptr = s.as_ptr();
    //for i in 0..s.len() {
    for i in 0..s.len() {
      let x = s_ptr as usize + i;
      let xp = x as *const u8;
      print!("{}", *xp as char);
    }
    println!("");
    println!("hello rust lib: {:?}", s);
  }
  42
}
