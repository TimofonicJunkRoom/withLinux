
extern {
  fn write(fd: i32, data: *const u8, len: usize) -> i32;
}

fn main() {
  let data = b"Hello, world!\n";
  unsafe {
    write(1, &data[0], data.len());
  }
}
