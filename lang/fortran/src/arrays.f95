program arrays

  implicit none

  integer :: i, j
  real, dimension(5) :: numbers
  integer, dimension(5,5) :: matrix
  real, dimension(5) :: demo
  integer, dimension(4) :: xxx

  interface
    subroutine ArrayFlushZero (x)
      integer, dimension(:), intent (out) :: x
    end subroutine ArrayFlushZero
  end interface

  numbers(1) = 2.0
  demo = (/1.5, 3.2,4.5,0.9,7.2 /) ! short hand array constructor

  do i = 2, 5
    numbers(i) = i
  end do
  do i = 1, 5
    do j = 1, 5
      matrix(i, j) = i*10 + j ! fortran is colum-major indexed
    end do
  end do

  print *, numbers
  print *, demo
  print *, matrix

  call ArrayFlushZero (xxx)
  print *, xxx

  demo(:) = 0
  demo(2:3) = 1
  demo(4:) = 2
  demo(1:5:2) = 10
  print *, demo

  ! multiplication functions
  print *, 'multiplication functions'
  print *, dot_product(demo, demo) ! \vec{a} \cdot \vec{b}
  print *, matmul(matrix, matrix)  ! \mat{A} \cdot \mat{B}

  ! reduction functions
  print *, 'reduction'
  demo = (/1.5, 3.2,4.5,0.9,7.2 /) ! short hand array constructor
  print *, all(demo > 3) ! all(mask, dim)
  print *, all(demo > 0 .and. demo < 10) ! (mask, dim)
  print *, any(demo > 3) ! (mask, dim)
  print *, count(demo > 3) ! (mask, dim)
  print *, maxval(demo) ! maxval(array, dim, mask)
  print *, minval(demo) ! minval(array, dim, mask)
  print *, sum(demo)    ! sum(array, dim, mask)
  print *, product(demo)! product(array, dim, mask)

  ! inquery
  ! allocated(array) logical
  ! lbound(array, dim)
  ! shape(source)
  ! size(array, dim)
  ! ubound(array, dim)

end program arrays

subroutine ArrayFlushZero (x)
  implicit none
  integer, dimension(:), intent (out) :: x

  integer :: i
  do i = 1, size(x)
    x(i) = 0
  end do
end subroutine ArrayFlushZero
