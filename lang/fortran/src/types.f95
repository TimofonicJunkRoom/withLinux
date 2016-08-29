program types

  implicit none

  integer(kind=1) :: i8
  integer(kind=2) :: i16
  integer(kind=4) :: i32
  integer(kind=8) :: i64
  integer(kind=16) :: i128
  integer :: idefault

  integer :: i, j, k
  real :: p, q, r
  double precision :: x

  logical :: y, n

  character (len=40) :: username

  complex :: c, cx

  real, parameter :: pi = 3.141592653

  print *, huge(i8), huge(i16), huge(i32), huge(i64), huge(i128)
  print *, huge(idefault)

  i = 2
  j = 3
  k = i / j
  p = 2.
  q = 3.
  r = p / q
  print *, k, r

  print *, huge(p), huge(x)

  y = .true.
  n = .false.

  print *, y, n

  username = "Anonymous"
  print *, username, username(1:4)

  c = (3.0, -5.0)
  print *, c
  cx = cmplx(p, q)
  print *, cx

  print *, pi ** p

end program types
