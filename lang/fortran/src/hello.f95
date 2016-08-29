program hello

! we add a and b together in this program
! implicit none means no implicit type declaration
  implicit none

! type declarations
  real :: a, b, result

! executable part
  Print *, "hello fortran"
  a = 12.0
  b = 15.0
  result = a + b
  print *, 'the total is', result
  ! Note, fortran is case-insensitive!
  result = A + B
  print *, 'the total is', result

end program hello

! The non-I/O keywords
! allocatable 	allocate 	assign 	assignment 	block data
! call 	case 	character 	common 	complex
! contains 	continue 	cycle 	data 	deallocate
! default 	do 	double precision 	else 	else if
! elsewhere 	end block data 	end do 	end function 	end if
! end interface 	end module 	end program 	end select 	end subroutine
! end type 	end where 	entry 	equivalence 	exit
! external 	function 	go to 	if 	implicit
! in 	inout 	integer 	intent 	interface
! intrinsic 	kind 	len 	logical 	module
! namelist 	nullify 	only 	operator 	optional
! out 	parameter 	pause 	pointer 	private
! program 	public 	real 	recursive 	result
! return 	save 	select case 	stop 	subroutine
! target 	then 	type 	type() 	use
! Where 	While

! The I/O related keywords
! backspace 	close 	endfile 	format 	inquire
! open 	print 	read 	rewind 	Write
