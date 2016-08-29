program control

  implicit none

  integer :: a
  integer :: i, j, k

  a = 20

  ! if ... then ... else ...
  ! if ... else if ... else
  if (a > 10) then
    print *, "a > 10"
  else
    print *, 'a < 10'
  end if
  ! select case ...
  select case (a)
  case (10)
    print *, '10'
  case (20)
    print *, '20'
  case default
    print *, 'default'
  end select

  ! do loop
  do a = 1, 10
    print *, a, a**2
  end do
  ! do while loop
  a = 1
  do while (a < 10)
    print *, a, a**2
    a = a + 1
  end do
  ! exit
  iloop: do i = 1, 3
    jloop: do j = 1, 3
      kloop: do k = 1, 3
        if (3 == k) then
          exit kloop
        end if
        if (3 == i) then
          exit iloop
        end if
        print *, i, j, k
      end do kloop
    end do jloop
  end do iloop
  ! cycle
  print *, 'cycle'
  do a = 1, 3
    if (2 == a) then
      cycle
    end if
    print *, a
  end do
  ! stop
  print *, 'stop'
  do a = 1, 100
    if (3 < a) then
      stop
    end if
    print *, a
  end do
    
end program control
