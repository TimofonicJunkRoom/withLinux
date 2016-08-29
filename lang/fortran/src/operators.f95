program operators

  implicit none

  print *, 1 == 1, 1 .eq. 2
  print *, 1 /= 1, 1 .ne. 2
  print *, 1 >  1, 1 .gt. 2
  print *, 1 <  1, 1 .lt. 2
  print *, 1 >= 1, 1 .ge. 2
  print *, 1 <= 1, 1 .le. 2

  print *, .true. .and. .false., .true. .or. .false., .not. .true., .false. .eqv. .false., .false. .neqv. .true.

end program operators
