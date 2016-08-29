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
  complex, parameter :: imag = (0, 1) ! sqrt(-1)
  complex :: ca, cb

  character (len=40):: reply

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

  ca = (7, 8)
  cb = (5, -7)
  write (*,*) imag*ca*cb

  print *, kind(ca)
  print *, kind(y), kind(n)

  reply = 'this is reply'
  print *, reply
  print *, username//reply ! string concatenation
  print *, trim(username)//' '//reply

end program types

! len(string)
! index(string,sustring)
! achar(int)
! iachar(c)
! trim(string)
! scan(string, chars)
! verify(string, chars)
! adjustl(string)
! adjustr(string)
! len_trim(string)
! repeat(string,ncopy)
