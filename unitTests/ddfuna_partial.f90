
!  DDFUN: A double-double precision computation package with special functions

!  Computational routine module (DDFUNMOD).

!  Revision date:  23 May 2023

!  AUTHOR:
!     David H. Bailey
!     Lawrence Berkeley National Lab (retired) and University of California, Davis
!     Email: dhbailey@lbl.gov

!  COPYRIGHT AND DISCLAIMER:
!    All software in this package (c) 2023 David H. Bailey.
!    By downloading or using this software you agree to the copyright, disclaimer
!    and license agreement in the accompanying file DISCLAIMER.txt.

!  PURPOSE OF PACKAGE:
!    This package permits one to perform floating-point computations (real and complex)
!    to double-double precision (approximately 30 digits), by making only relatively
!    minor changes to existing Fortran programs. It is intended as a substitute for
!    IEEE 128-bit quad precision on systems where quad precision is not available. All
!    basic arithmetic operations and transcendental functions are supported. The
!    package should run correctly on any Unix-based system supporting a Fortran-2008
!    compiler and IEEE 64-bit floating-point arithmetic. Note however that results are
!    NOT guaranteed to the last bit.

!    In addition to fast execution times, one key feature of this package is a
!    100% THREAD-SAFE design, which means that user-level applications can be
!    easily converted for parallel execution, say using a threaded parallel
!    environment such as OpenMP. There are no global shared variables (except
!    static compile-time data), and no initialization is necessary.

!  DOCUMENTATION:
!    See the README-ddfun.txt file in the main DDFUN directory.

!  COMPILER NOTE: The GNU gfortran script to compile this file includes the flag 
!    "-fno-expensive-optimizations". This flag prevents the compiler from optimizing
!    away key operations in several basic arithmetic routines. A similar flag may be
!    required with other compilers.

!  DESCRIPTION OF THIS MODULE (DDFUNMOD):
!    This module contains most lower-level computational routines.

!  The following notational scheme is used to designate datatypes below:

!  DP  double precision [i.e. REAL (KIND (0.d0))] -- approx. 16 digit accuracy
!  DC  double complex [i.e. COMPLEX (KIND (0.d0))] -- approx. 16 digit accuracy
!  DDR Double-double real -- approx. 30 decimal digit accuracy
!  DDC Double-double complex -- approx. 30 decimal digit accuracy 

!  These parameters are set here:

!   Name     Default   Description
!  ddknd     8         Kind parameter for IEEE double floats (usually 8).
!                      This is set automatically by the selected_real_kind function.
!  ddldb     6         Logical device number for output of error messages.
!  ddnbt     53        Number of mantissa bits in DP word.
!  ddnwx     2         Number of words in DDR datum.
!  dddpw     log10(2^53) Approx. number of digits per DP word.
!  ddlogb    log(2^53) Approx. constant needed for zeta routines in ddfune.
!  ddrdfz    1/2^50    "Fuzz" for comparing DP values.
!  ddpicon   Pi        Two-word DDR value of pi.
!  ddegammacon Gamma   Two-word DDR value of Euler's gamma constant.

module ddfuna
   use iso_c_binding, only: c_double, c_int
integer, public, parameter:: ddknd = selected_real_kind (15, 307), ddldb = 6, &
  ddnbt = 53, ddnwx = 2
real(c_double), parameter:: dddpw = 15.9545897701910033d0, &
  ddlogb = 36.7368005696771014d0, ddrdfz = 2.d0**(-50), &
  ddpicon(1:2) = [3.1415926535897931D+00, 1.2246467991473532D-16], &
  ddegammacon(1:2) = [5.772156649015328655D-01, -4.942915152430636394D-18], &
  ddlog2con(1:2) = [6.9314718055994529D-01, 2.3190468138462996D-17]

contains

subroutine ddabrt
implicit none

!   This permits one to insert a call to a vendor-specific traceback routine.

stop
end subroutine ddabrt

subroutine ddabs (a, b) bind(C, name="ddabs_fortran")

!   This sets b = abs (a).

implicit none
real(c_double), intent(in):: a(2)
real(c_double), intent(out):: b(2)

if (a(1) >= 0.d0) then
  b(1) = a(1); b(2) = a(2)
else
  b(1) = - a(1); b(2) = - a(2)
endif

return
end subroutine ddabs

subroutine ddadd (dda, ddb, ddc) bind(C, name="ddadd_fortran")

!   This subroutine computes ddc = dda + ddb, where dda, ddb and ddc are type DDR.

implicit none
real(c_double), intent(in):: dda(2), ddb(2)
real(c_double), intent(out):: ddc(2)
real(c_double) e, t1, t2

!   Compute dda + ddb using Knuth's trick.

t1 = dda(1) + ddb(1)
e = t1 - dda(1)
t2 = ((ddb(1) - e) + (dda(1) - (t1 - e))) + dda(2) + ddb(2)

!   The result is t1 + t2, after normalization.

ddc(1) = t1 + t2
ddc(2) = t2 - (ddc(1) - t1)
return
end subroutine ddadd

subroutine ddacosh (a, b) bind(C, name="ddacosh_fortran")

!   This computes the inverse hyperbolic cosine of A, using the standard formula.

implicit none
real(c_double), intent(in):: a(2)
real(c_double), intent(out):: b(2)
real(c_double) f1(2), t1(2), t2(2)

!   Check if a < 1; error.

if (a(1) < 1.d0) then
  write (6, 1)
1 format ('DDACOSH: Argument is < 1.')
  call ddabrt
endif

f1(1) = 1.d0; f1(2) = 0.d0
call ddmul (a, a, t1)
call ddsub (t1, f1, t2)
call ddsqrt (t2, t1)
call ddadd (a, t1, t2)
call ddlog (t2, b)
end subroutine ddacosh

subroutine ddagmr (a, b, c) bind(C, name="ddagmr_fortran")

!   This performs the arithmetic-geometric mean (AGM) iterations on A and B.
!   The AGM algorithm is as follows: Set a_0 = a and b_0 = b, then iterate

!    a_{k+1} = (a_k + b_k)/2
!    b_{k+1} = sqrt (a_k * b_k)

!   until convergence (i.e., until a_k = b_k to available precision).
!   The result is returned in C.

implicit none
real(c_double), intent(in):: a(1:2), b(1:2)
real(c_double), intent(out):: c(1:2)
integer, parameter:: itrmx = 100
integer j
real(c_double) eps, s0(1:2), s1(1:2), s2(1:2), s3(1:2)

eps = 2.q0 ** (2 - ddnwx*ddnbt)
call ddeq (a, s1)
call ddeq (b, s2)

do j = 1, itrmx
  call ddadd (s1, s2, s0)
  call ddmuld (s0, 0.5d0, s3)
  call ddmul (s1, s2, s0)
  call ddsqrt (s0, s2)
  call ddeq (s3, s1)

!   Check for convergence.

  call ddsub (s1, s2, s0)
  if (s0(1) == 0.q0 .or. s0(1) / s1(1) < eps) goto 100
enddo

write (ddldb, 2)
2 format ('*** DDAGMR: Iteration limit exceeded.')
call ddabrt

100 continue

call ddeq (s1, c)

return
end subroutine ddagmr

subroutine ddang (x, y, a) bind(C, name="ddang_fortran")

!   This computes the DDR angle A subtended by the DDR pair (X, Y) considered as
!   a point in the x-y plane. This is more useful than an arctan or arcsin
!   routine, since it places the result correctly in the full circle, i.e.
!   -Pi < A <= Pi.

!   The Taylor series for Sin converges much more slowly than that of Arcsin.
!   Thus this routine does not employ Taylor series, but instead computes
!   Arccos or Arcsin by solving Cos (a) = x or Sin (a) = y using one of the
!   following Newton iterations, both of which converge to a:

!           z_{k+1} = z_k - [x - Cos (z_k)] / Sin (z_k)
!           z_{k+1} = z_k + [y - Sin (z_k)] / Cos (z_k)

!   The first is selected if Abs (x) <= Abs (y); otherwise the second is used.

implicit none
real(c_double), intent(in):: x(2), y(2)
real(c_double), intent(out):: a(2)
integer i, ix, iy, k, kk, nx, ny
real(c_double) t1, t2, t3
real(c_double) s0(2), s1(2), s2(2), s3(2), s4(2)
real(c_double), parameter:: pi(1:2) = &
  [3.1415926535897931D+00, 1.2246467991473532D-16]

!   Check if both X and Y are zero.

if (x(1) == 0.d0 .and. y(1) == 0.d0) then
  write (ddldb, 1)
1 format ('*** DDANG: Both arguments are zero.')
  call ddabrt
  return
endif

!   Check if one of X or Y is zero.

if (x(1) == 0.d0) then
  if (y(1) > 0.d0) then
    call ddmuld (pi, 0.5d0, a)
  else
    call ddmuld (pi, -0.5d0, a)
  endif
  goto 120
elseif (y(1) == 0.d0) then
  if (x(1) > 0.d0) then
      a(1) = 0.d0
      a(2) = 0.d0
  else
    a(1) = pi(1)
    a(2) = pi(2)
  endif
  goto 120
endif

!   Normalize x and y so that x^2 + y^2 = 1.

call ddmul (x, x, s0)
call ddmul (y, y, s1)
call ddadd (s0, s1, s2)
call ddsqrt (s2, s3)
call dddiv (x, s3, s1)
call dddiv (y, s3, s2)

!   Compute initial approximation of the angle.

call dddddpc (s1, t1)
call dddddpc (s2, t2)
t3 = atan2 (t2, t1)
a(1) = t3
a(2) = 0.d0

!   The smaller of x or y will be used from now on to measure convergence.
!   This selects the Newton iteration (of the two listed above) that has the
!   largest denominator.

if (abs (t1) <= abs (t2)) then
  kk = 1
  s0(1) = s1(1)
  s0(2) = s1(2)
else
  kk = 2
  s0(1) = s2(1)
  s0(2) = s2(2)
endif

!   Perform the Newton-Raphson iteration described.

do k = 1, 3
  call ddcssnr (a, s1, s2)
  if (kk == 1) then
    call ddsub (s0, s1, s3)
    call dddiv (s3, s2, s4)
    call ddsub (a, s4, s1)
  else
    call ddsub (s0, s2, s3)
    call dddiv (s3, s1, s4)
    call ddadd (a, s4, s1)
  endif
  a(1) = s1(1)
  a(2) = s1(2)
enddo

 120  continue

return
end subroutine ddang

subroutine ddasinh (a, b) bind(C, name="ddasinh_fortran")

!   This computes the inverse hyperbolic sine of A, using the standard formula.

implicit none
real(c_double), intent(in):: a(2)
real(c_double), intent(out):: b(2)
real(c_double) f1(2), t1(2), t2(2)

f1(1) = 1.d0; f1(2) = 0.d0
call ddmul (a, a, t1)
call ddadd (t1, f1, t2)
call ddsqrt (t2, t1)
call ddadd (a, t1, t2)
call ddlog (t2, b)
end subroutine ddasinh

subroutine ddatanh (a, b) bind(C, name="ddatanh_fortran")

!   This computes the inverse hyperbolic tangent of A, using the standard formula.

implicit none
real(c_double), intent(in):: a(2)
real(c_double), intent(out):: b(2)
real(c_double) f1(2), t1(2), t2(2), t3(2)

!   Check if a <= -1 or a >= 1; error.

if (abs (a(1)) >= 1.d0) then
  write (6, 1)
1 format ('DDATANH: Argument is <= -1 or >= 1.')
  call ddabrt
endif

f1(1) = 1.d0; f1(2) = 0.d0
call ddadd (f1, a, t1)
call ddsub (f1, a, t2)
call dddiv (t1, t2, t3)
call ddlog (t3, t1)
call ddmuld (t1, 0.5d0, b)
end subroutine ddatanh

subroutine ddcadd (a, b, c) bind(C, name="ddcadd_fortran")

!   This computes the sum of the DDC numbers A and B and returns the DDC
!   result in C.

implicit none
real(c_double), intent(in):: a(4), b(4)
real(c_double), intent(out):: c(4)

call ddadd (a, b, c)
call ddadd (a(3), b(3), c(3))

return
end subroutine ddcadd

subroutine ddcdiv (a, b, c) bind(C, name="ddcdiv_fortran")

!   This routine divides the DDC numbers A and B to yield the DDC quotient C.
!   This routine employs the formula described in ddCMUL to save multiprecision
!   multiplications.

implicit none
real(c_double), intent(in):: a(4), b(4)
real(c_double), intent(out):: c(4)
real(c_double) f(2), s0(2), s1(2), s2(2), s3(2), s4(2)

if (b(1) == 0.d0 .and. b(3) == 0.d0) then
  write (ddldb, 1)
1 format ('*** DDCDIV: Divisor is zero.')
  call ddabrt
  return
endif

f(1) = 1.d0
f(2) = 0.d0
call ddmul (a, b, s0)
call ddmul (a(3), b(3), s1)
call ddadd (s0, s1, s2)
call ddsub (s0, s1, s3)
call ddadd (a, a(3), s0)
call ddsub (b, b(3), s1)
call ddmul (s0, s1, s4)
call ddsub (s4, s3, s1)
call ddmul (b, b, s0)
call ddmul (b(3), b(3), s3)
call ddadd (s0, s3, s4)
call dddiv (f, s4, s0)
call ddmul (s2, s0, c)
call ddmul (s1, s0, c(3))

return
end subroutine ddcdiv

subroutine ddceq (a, b) bind(C, name="ddceq_fortran")

!   This sets the DDC number B equal to the DDC number A.

implicit none
real(c_double), intent(in):: a(4)
real(c_double), intent(out):: b(4)

b(1) = a(1)
b(2) = a(2)
b(3) = a(3)
b(4) = a(4)

return
end subroutine ddceq

subroutine ddcmul (a, b, c) bind(C, name="ddcmul_fortran")

!   This routine multiplies the DDC numbers A and B to yield the DDC result.

implicit none
real(c_double), intent(in):: a(4), b(4)
real(c_double), intent(out):: c(4)
real(c_double) s0(2), s1(2), s2(2), s3(2)

call ddmul (a, b, s0)
call ddmul (a(3), b(3), s1)
call ddmul (a, b(3), s2)
call ddmul (a(3), b, s3)
call ddsub (s0, s1, c)
call ddadd (s2, s3, c(3))

return
end subroutine ddcmul

subroutine ddcpr (a, b, ic) bind(C, name="ddcpr_fortran")

!   This routine compares the DDR numbers A and B and returns in IC the value
!   -1, 0, or 1 depending on whether A < B, A = B, or A > B. It is faster
!   than merely subtracting A and B and looking at the sign of the result.

implicit none
real(c_double), intent(in):: a(2), b(2)
integer, intent(out):: ic

if (a(1) < b(1)) then
  ic = -1
elseif (a(1) == b(1)) then
  if (a(2) < b(2)) then
    ic = -1
  elseif (a(2) == b(2)) then
    ic = 0
  else
    ic = 1
  endif
else
  ic = 1
endif

return
end subroutine ddcpr

subroutine ddcpwr (a, n, b) bind(C, name="ddcpwr_fortran")

!   This computes the N-th power of the ddC number A and returns the DDC
!   result C in B. When N is zero, 1 is returned. When N is negative, the
!   reciprocal of A ^ |N| is returned.

!   This routine employs the binary method for exponentiation.

implicit none
real(c_double), intent(in):: a(4)
integer(c_int), intent(in):: n
real(c_double), intent(out):: b(4)
real(c_double), parameter:: cl2 = 1.4426950408889633d0
integer j, kk, kn, l1, mn, na1, na2, nn
real(c_double) t1
real(c_double) s0(4), s1(4), s2(4), s3(4)

if (a(1) == 0.d0 .and. a(3) == 0.d0) then
  if (n >= 0) then
    b(1) = 0.d0
    b(2) = 0.d0
    b(3) = 0.d0
    b(4) = 0.d0
    goto 120
  else
    write (ddldb, 1)
1   format ('*** DDCPWR: Argument is zero and N is negative or zero.')
    call ddabrt
    return
  endif
endif

nn = abs (n)
if (nn == 0) then
  b(1) = 1.d0
  b(2) = 0.d0
  b(3) = 0.d0
  b(4) = 0.d0
  goto 120
elseif (nn == 1) then
  s2(1) = a(1)
  s2(2) = a(2)
  s2(3) = a(3)
  s2(4) = a(4)
  goto 110
elseif (nn == 2) then
  call ddcmul (a, a, s2)
  goto 110
endif

!   Determine the least integer MN such that 2 ^ MN > NN.

t1 = nn
mn = cl2 * log (t1) + 1.d0 + 1.d-14

s0(1) = a(1)
s0(2) = a(2)
s0(3) = a(3)
s0(4) = a(4)
s2(1) = 1.d0
s2(2) = 0.d0
s2(3) = 0.d0
s2(4) = 0.d0
kn = nn

!   Compute B ^ N using the binary rule for exponentiation.

do j = 1, mn
  kk = kn / 2
  if (kn /= 2 * kk) then
    call ddcmul (s2, s0, s1)
    s2(1) = s1(1)
    s2(2) = s1(2)
    s2(3) = s1(3)
    s2(4) = s1(4)
  endif
    kn = kk
  if (j < mn) then
    call ddcmul (s0, s0, s1)
    s0(1) = s1(1)
    s0(2) = s1(2)
    s0(3) = s1(3)
    s0(4) = s1(4)
  endif
enddo

!   Compute reciprocal if N is negative.

110  continue

if (n < 0) then
  s1(1) = 1.d0
  s1(2) = 0.d0
  s1(3) = 0.d0
  s1(4) = 0.d0
  call ddcdiv (s1, s2, s0)
  s2(1) = s0(1)
  s2(2) = s0(2)
  s2(3) = s0(3)
  s2(4) = s0(4)
endif

b(1) = s2(1)
b(2) = s2(2)
b(3) = s2(3)
b(4) = s2(4)

120  continue
return
end subroutine ddcpwr

subroutine ddcsqrt (a, b) bind(C, name="ddcsqrt_fortran")

!   This routine computes the complex square root of the DDC number C.
!   This routine uses the following formula, where A1 and A2 are the real and
!   imaginary parts of A, and where R = Sqrt [A1 ^ 2 + A2 ^2]:

!      B = Sqrt [(R + A1) / 2] + I Sqrt [(R - A1) / 2]

!   If the imaginary part of A is < 0, then the imaginary part of B is also
!   set to be < 0.

implicit none
real(c_double), intent(in):: a(4)
real(c_double), intent(out):: b(4)
real(c_double) s0(2), s1(2), s2(2)

if (a(1) == 0.d0 .and. a(3) == 0.d0) then
  b(1) = 0.d0
  b(2) = 0.d0
  b(3) = 0.d0
  b(4) = 0.d0
  goto 100
endif

call ddmul (a, a, s0)
call ddmul (a(3), a(3), s1)
call ddadd (s0, s1, s2)
call ddsqrt (s2, s0)

s1(1) = a(1)
s1(2) = a(2)
if (s1(1) < 0.d0) then
  s1(1) = - s1(1)
  s1(2) = - s1(2)
endif
call ddadd (s0, s1, s2)
call ddmuld (s2, 0.5d0, s1)
call ddsqrt (s1, s0)
call ddmuld (s0, 2.d0, s1)
if (a(1) >= 0.d0) then
  b(1) = s0(1)
  b(2) = s0(2)
  call dddiv (a(3), s1, b(3))
else
  call dddiv (a(3), s1, b)
  if (b(1) < 0.d0) then
    b(1) = - b(1)
    b(2) = - b(2)
  endif
  b(3) = s0(1)
  b(4) = s0(2)
  if (a(3) < 0.d0) then
    b(3) = - b(3)
    b(4) = - b(4)
  endif
endif

 100  continue
return
end subroutine ddcsqrt

subroutine ddcsshr (a, x, y) bind(C, name="ddcsshr_fortran")

!   This computes the hyperbolic cosine and sine of the DDR number A and
!   returns the two DDR results in X and Y, respectively. 

implicit none
real(c_double), intent(in):: a(2)
real(c_double), intent(out):: x(2), y(2)
real(c_double) f(2), s0(2), s1(2), s2(2)

f(1) = 1.d0
f(2) = 0.d0
call ddexp (a, s0)
call dddiv (f, s0, s1)
call ddadd (s0, s1, s2)
call ddmuld (s2, 0.5d0, x)
call ddsub (s0, s1, s2)
call ddmuld (s2, 0.5d0, y)

return
end subroutine ddcsshr

subroutine ddcssnr (a, x, y) bind(C, name="ddcssnr_fortran")

!   This computes the cosine and sine of the DDR number A and returns the
!   two DDR results in X and Y, respectively.

!   This routine uses the conventional Taylor series for Sin (s):

!   Sin (s) =  s - s^3 / 3! + s^5 / 5! - s^7 / 7! ...

!   where the argument S has been reduced to (-pi, pi). To further accelerate
!   convergence of the series, the reduced argument is divided by 2^NQ. After
!   convergence, the double-angle formulas for cos are applied NQ times.

implicit none
real(c_double), intent(in):: a(2)
real(c_double), intent(out):: x(2), y(2)
integer, parameter:: itrmx = 1000, nq = 5
real(c_double), parameter:: eps = 1.d-32
integer is, i1, j, na, n1
real(c_double) d1, t1, t2
real(c_double) f1(2), f2(2), s0(2), s1(2), s2(2), s3(2), s4(2), s5(2)
real(c_double), parameter:: pi(1:2) = &
  [3.1415926535897931D+00, 1.2246467991473532D-16]

! End of declaration

na = 2
if (a(1) == 0.d0) na = 0
if (na == 0) then
  x(1) = 1.d0
  x(2) = 0.d0
  y(1) = 0.d0
  y(2) = 0.d0
  goto 120
endif

!   Set f1 = 1 and f2 = 1/2.

f1(1) = 1.d0
f1(2) = 0.d0
f2(1) = 0.5d0
f2(2) = 0.d0

!   Check if argument is too large to compute meaningful cos/sin values.

if (a(1) >= 1.q60) then
  write (ddldb, 3)
3 format ('*** DDCSSNR: argument is too large to compute cos or sin.')
  call ddabrt
endif

!   Reduce to between - Pi and Pi.

call ddmuld (pi, 2.d0, s0)
call dddiv (a, s0, s1)
call ddnint (s1, s2)
call ddmul (s0, s2, s4)
call ddsub (a, s4, s3)

!   Check if reduced argument is zero. If so then cos = 1 and sin = 0.

if (s3(1) == 0.d0) then
  x(1) = 1.d0
  x(2) = 0.d0
  y(1) = 0.d0
  y(2) = 0.d0
  goto 120
endif

call dddivd (s3, 2.d0**nq, s0)
s1(1) = s0(1); s1(2) = s0(2)

!   Compute the sin of the reduced argument of s1 using a Taylor series.

call ddmul (s0, s0, s2)
if (s0(1) < 0.d0) then
  is = -1
else
  is = 1
endif

do i1 = 1, itrmx
  t2 = - (2.d0 * i1) * (2.d0 * i1 + 1.d0)
  call ddmul (s2, s1, s3)
  call dddivd (s3, t2, s1)
  call ddadd (s1, s0, s3)
!  call mpeq (s3, s0, mpnw1)
  s0(1) = s3(1); s0(2) = s3(2)

!   Check for convergence of the series, and adjust working precision
!   for the next term.

  if (abs (s1(1)) < eps) goto 110
enddo

write (ddldb, 4)
4 format ('*** DDCSSNR: Iteration limit exceeded.')
call ddabrt

110 continue

!   Apply the formula cos(2*x) = 2*cos^2(x) - 1 NQ times to produce
!   the cosine of the reduced argument, except that the first iteration is
!   cos(2*x) = 1 - 2*sin^2(x), since we have computed sin(x) above.
!   Note that these calculations are performed as 2 * (cos^2(x) - 1/2) and
!   2 * (1/2 - sin^2(x)), respectively, to avoid loss of precision.

call ddmul (s0, s0, s4)
call ddsub (f2, s4, s5)
call ddmuld (s5, 2.d0, s0)

do j = 2, nq
  call ddmul (s0, s0, s4)
  call ddsub (s4, f2, s5)
  call ddmuld (s5, 2.d0, s0)
enddo

!   Compute sin of result and correct sign.

call ddmul (s0, s0, s4)
call ddsub (f1, s4, s5)
call ddsqrt (s5, s1)

if (is < 1) then
  s1(1) = - s1(1); s1(2) = - s1(2)
endif

115 continue

x(1) = s0(1); x(2) = s0(2)
y(1) = s1(1); y(2) = s1(2)

120 continue

return
end subroutine ddcssnr

subroutine ddcsub (a, b, c) bind(C, name="ddcsub_fortran")

!   This subracts the DDC numbers A and B and returns the DDC difference in C.

implicit none
real(c_double) a(4), b(4), c(4)

call ddsub (a, b, c)
call ddsub (a(3), b(3), c(3))

return
end subroutine ddcsub

character(32) function dddigout (a, n)

!   This converts the double precision input A to a character(32) string of
!   nonblank length N. A must be a whole number, and N must be sufficient
!   to hold it. This is intended for internal use only.

  implicit none
  integer, intent(in):: n
  real(c_double), intent(in):: a
  character(10), parameter:: digits = '0123456789'
  real(c_double) d1, d2
  character(32) ca
  integer i, k

! End of declaration

  ca = ' '
  d1 = abs (a)

  do i = n, 1, -1
    d2 = aint (d1 / 10.d0)
    k = 1.d0 + (d1 - 10.d0 * d2)
    d1 = d2
    ca(i:i) = digits(k:k)
  enddo

  dddigout = ca
  return
end function dddigout

subroutine dddiv (dda, ddb, ddc) bind(C, name="dddiv_fortran")

!   This divides the DDR number DDA by the DDR number DDB to yield the DD
!   quotient DDC.

implicit none
real(c_double), intent(in):: dda(2), ddb(2)
real(c_double), intent(out):: ddc(2)
real(c_double), parameter:: split = 134217729.d0
real(c_double) a1, a2, b1, b2, cona, conb, c11, c2, c21, e, s1, s2, &
  t1, t2, t11, t12, t21, t22

!   Compute a DDR approximation to the quotient.

s1 = dda(1) / ddb(1)

!   This splits s1 and ddb(1) into high-order and low-order words.

cona = s1 * split
conb = ddb(1) * split
a1 = cona - (cona - s1)
b1 = conb - (conb - ddb(1))
a2 = s1 - a1
b2 = ddb(1) - b1

!   Multiply s1 * ddb(1) using Dekker's method.

c11 = s1 * ddb(1)
c21 = (((a1 * b1 - c11) + a1 * b2) + a2 * b1) + a2 * b2
!>
!   Compute s1 * ddb(2) (only high-order word is needed).

c2 = s1 * ddb(2)

!   Compute (c11, c21) + c2 using Knuth's trick.

t1 = c11 + c2
e = t1 - c11
t2 = ((c2 - e) + (c11 - (t1 - e))) + c21

!   The result is t1 + t2, after normalization.

t12 = t1 + t2
t22 = t2 - (t12 - t1)

!   Compute dda - (t12, t22) using Knuth's trick.

t11 = dda(1) - t12
e = t11 - dda(1)
t21 = ((-t12 - e) + (dda(1) - (t11 - e))) + dda(2) - t22

!   Compute high-order word of (t11, t21) and divide by ddb(1).

s2 = (t11 + t21) / ddb(1)

!   The result is s1 + s2, after normalization.

ddc(1) = s1 + s2
ddc(2) = s2 - (ddc(1) - s1)

return
end subroutine dddiv

subroutine dddivd (dda, db, ddc) bind(C, name="dddivd_fortran")

!   This routine divides the DDR number A by the DP number B to yield
!   the DDR quotient C. DB must be an integer or exact binary fraction,
!   such as 3., 0.25 or -0.375.

implicit none
real(c_double), intent(in):: dda(2), db
real(c_double), intent(out):: ddc(2)
real(c_double), parameter:: split = 134217729.d0
real(c_double) a1, a2, b1, b2, cona, conb, e, t1, t2, t11, t12, t21, t22

!   Compute a DP approximation to the quotient.

t1 = dda(1) / db
!>
!   On systems with a fused multiply add, such as IBM systems, it is faster to
!   uncomment the next two lines and comment out the following lines until !>.
!   On other systems, do the opposite.

! t12 = t1 * db
! t22 = t1 * db - t12

!   This splits t1 and db into high-order and low-order words.

cona = t1 * split
conb = db * split
a1 = cona - (cona - t1)
b1 = conb - (conb - db)
a2 = t1 - a1
b2 = db - b1

!   Multiply t1 * db using Dekker's method.

t12 = t1 * db
t22 = (((a1 * b1 - t12) + a1 * b2) + a2 * b1) + a2 * b2
!>
!   Compute dda - (t12, t22) using Knuth's trick.

t11 = dda(1) - t12
e = t11 - dda(1)
t21 = ((-t12 - e) + (dda(1) - (t11 - e))) + dda(2) - t22

!   Compute high-order word of (t11, t21) and divide by db.

t2 = (t11 + t21) / db

!   The result is t1 + t2, after normalization.

ddc(1) = t1 + t2
ddc(2) = t2 - (ddc(1) - t1)
return
end subroutine dddivd

subroutine dddpddc (a, b) bind(C, name="dddpddc_fortran")

!   This routine converts the DP number A to DDR form in B. A must
!   be an integer or exact binary fraction, such as 3., 0.25 or -0.375.

implicit none
real(c_double), intent(in):: a
real(c_double), intent(out):: b(2)

b(1) = a
b(2) = 0.d0
return
end subroutine dddpddc

subroutine dddddpc (a, b) bind(C, name="dddddpc_fortran")

!   This converts the DDR number A to DP.

implicit none
real(c_double), intent(in):: a(2)
real(c_double), intent(out):: b

b = a(1)
return
end subroutine dddddpc

subroutine ddeq (a, b) bind(C, name="ddeq_fortran")

!   This routine sets the DDR number B equal to the DDR number A. 

implicit none
real(c_double), intent(in):: a(2)
real(c_double), intent(out):: b(2)

b(1) = a(1)
b(2) = a(2)
end subroutine ddeq

real(c_double) function dddigin (ca, n)
implicit none
character(*), intent(in):: ca
integer, intent(in):: n
character(10), parameter:: digits = '0123456789'
real(c_double) d1
integer i, k

d1 = 0.d0

do i = 1, n
  k = index (digits, ca(i:i)) - 1
  if (k < 0) then
    write (ddldb, *) 'DDDIGIN: non-digit in character string'
  elseif (k <= 9) then
    d1 = 10.d0 * d1 + k
  endif
enddo

dddigin = d1
end function dddigin

subroutine dddmc (a, n, b) bind(C, name="dddmc_fortran")

!   This converts the DP number A * 2^N to DDR form in B.

!   NOTE however that the product is not fully accurate unless A is an exact
!   binary value.
!   Examples of exact binary values (good): 123456789.d0, 0.25d0, -5.3125d0.
!   Examples of inexact binary values (bad): 0.1d0, 123467.8d0, -3333.3d0.

implicit none
integer, intent(in):: n
real(c_double), intent(in):: a
real(c_double), intent(out):: b(2)
integer i, k, n1, n2
real(c_double) aa

b(1) = a * 2.d0 ** n
b(2) = 0.d0
return
end subroutine dddmc

subroutine ddeformat (a, nb, nd, b) bind(C, name="ddeformat_fortran")

!   Converts the DDR number A into character form in the character(1) array B.
!   NB (input) is the length of the output string, and ND (input) is the
!   number of digits after the decimal point. The format is analogous to
!   Fortran E format. The result is left-justified among the NB cells of B.
!   The condition NB >= ND + 8 must hold or an error message will result.
!   NB cells must be available in array B.

implicit none
real(c_double), intent(in):: a(2)
integer, intent(in):: nb, nd
character(1), intent(out):: b(nb)
character(10), parameter:: digits = '0123456789'
integer, parameter:: ndpw = 15
real(c_double), parameter:: d10w = 10.d0**ndpw
integer i, ia, ix, ixp, i1, i2, j, k, na, nexp, nl
character(1) b2(nb+50)
character(32) ca
real(c_double) aa, an, f(2), s0(2), s1(2), s2(2), s3(2), t1, t2

! End of declaration

if (nb < nd + 8) then
  write (ddldb, 1)
1 format ('*** DDEFORMAT: uninitialized or inadequately sized arrays')
  call ddabrt
endif

ia = sign (1.d0, a(1))
if (a(1) == 0.d0) ia = 0
na = 2
if (ia == 0) na = 0

!   Set f = 10.

f(1) = 10.d0; f(2) = 0.d0

!   Determine power of ten for exponent, and scale input to within 1 and 10.

if (ia < 0) then
  s1(1) = - a(1); s1(2) = - a(2)
else
  s1(1) = a(1); s1(2) = a(2)
endif

if (na > 0) then
  aa = s1(1)
  t1 = log10 (aa)

  if (t1 >= 0.d0) then
    nexp = t1
  else
    nexp = t1 - 1.d0
  endif

  if (nexp == 0) then
  elseif (nexp > 0) then
    call ddnpwr (f, nexp, s0)
    call dddiv (s1, s0, s2)
    s1(1) = s2(1); s1(2) = s2(2)
  elseif (nexp < 0) then
    call ddnpwr (f, -nexp, s0)
    call ddmul (s1, s0, s2)
    s1(1) = s2(1); s1(2) = s2(2)
  endif

!   If we didn't quite get it exactly right, multiply or divide by 10 to fix.

100 continue

  if (s1(1) < 1.d0) then
    nexp = nexp - 1
    call ddmuld (s1, 10.d0, s0)
    s1(1) = s0(1); s1(2) = s0(2)
    goto 100
  elseif (s1(1) >= 10.d0) then
    nexp = nexp + 1
    call dddivd (s1, 10.d0, s0)
    s1(1) = s0(1); s1(2) = s0(2)
    goto 100
  endif
else
  nexp = 0
endif

!   Insert sign and first digit.

ix = 0
if (ia == -1) then
  ix = ix + 1
  b2(ix) = '-'
endif
if (na > 0) then
  call ddinfr (s1, s2, s3)
  an = s2(1)
else
  an = 0.d0
endif
ca = dddigout (an, 1)
ix = ix + 1
b2(ix) = ca(1:1)
ix = ix + 1
b2(ix) = '.'
ixp = ix

!   Set f = an.

f(1) = an
f(2) = 0.d0
call ddsub (s1, f, s0)
call ddmuld (s0, d10w, s1)

!   Calculate the number of remaining chunks.

nl = nd / ndpw + 1

!   Insert the digits of the remaining words.

do j = 1, nl
  if (s1(1) /= 0.d0) then
    call ddinfr (s1, s2, s3)
    an = s2(1)
    f(1) = an
    f(2) = 0.d0
  else
    an = 0.d0
    f(1) = 0.d0
    f(2) = 0.d0
  endif

  ca = dddigout (an, ndpw)

  do i = 1, ndpw
    ix = ix + 1
    if (ix > nb + 50) then
      write (ddldb, 2)
2     format ('DDEFORMAT: Insufficient space in B2 array.')
      call ddabrt
    endif
    b2(ix) = ca(i:i)
  enddo
  
  call ddsub (s1, f, s0)
  call ddmuld (s0, d10w, s1)
enddo

!   Round the result.

if (ix >= nd + 1) then
  i1 = index (digits, b2(nd+1)) - 1
  if (i1 >= 5) then

!   Perform rounding, beginning at the last digit (position IX). If the rounded
!   digit is 9, set to 0, then repeat at position one digit to left. Continue
!   rounding if necessary until the decimal point is reached.

    do i = ix, ixp + 1, -1
      i2 = index (digits, b2(i)) - 1
      if (i2 <= 8) then
        b2(i) = digits(i2+2:i2+2)
        goto 180
      else
        b2(i) = '0'
      endif
    enddo

!   We have rounded up all digits to the right of the decimal point. If the
!   digit to the left of the decimal point is a 9, then set that digit to 1
!   and increase the exponent by one; otherwise increase that digit by one.

    if (b2(ixp-1) == '9') then
      b2(ixp-1) = '1'
      nexp = nexp + 1
    else
      i1 = index (digits, b2(ixp-1)) - 1
      b2(ixp-1) = digits(i1+2:i1+2)
    endif
  endif
endif

180 continue

!   Done with mantissa. Insert exponent.

ix = nd + 2
if (ia < 0) ix = ix + 1
b2(ix) = 'e'
if (nexp < 0) then
  ix = ix + 1
  b2(ix) = '-'
endif
ca = dddigout (real (abs (nexp), ddknd), 10)

do k = 1, 10
  if (ca(k:k) /= '0') goto 190
enddo

k = 10

190 continue

do i = k, 10
  ix = ix + 1
  b2(ix) = ca(i:i)
enddo

do i = ix + 1, nb
  b2(i) = ' '
enddo

!   Copy entire b2 array to B.

do i = 1, nb
  b(i) = b2(i)
enddo

return
end subroutine ddeformat

subroutine ddegamc (egam) bind(C, name="ddegamc_fortran")

!   This returns EGAMMA to DDR precision.

implicit none
real(c_double), intent(out):: egam(2)

egam(1) = ddegammacon(1)
egam(2) = ddegammacon(2)

return
end subroutine ddegamc

subroutine ddexp (a, b) bind(C, name="ddexp_fortran")

!   This computes the exponential function of the DDR number A and returns the
!   DDR result in B.

!   This routine uses a modification of the Taylor's series for Exp (t):

!   Exp (t) =  (1 + r + r^2 / 2! + r^3 / 3! + r^4 / 4! ...) ^ q * 2 ^ n

!   where q = 64, r = t' / q, t' = t - n Log(2) and where n is chosen so
!   that -0.5 Log(2) < t' <= 0.5 Log(2). Reducing t mod Log(2) and
!   dividing by 64 insures that -0.004 < r <= 0.004, which accelerates
!   convergence in the above series.

implicit none
real(c_double), intent(in):: a(2)
real(c_double), intent(out):: b(2)
integer, parameter:: nq = 6
integer i, ia, l1, na, nz, n1
real(c_double) t1, t2
real(c_double) eps, f(2), s0(2), s1(2), s2(2), s3(2), tl
real(c_double), parameter:: al2(1:2) = &
  [6.9314718055994529D-01,  2.3190468138462996D-17]

!   Check for overflows and underflows.

eps = 10.d0 ** (-32)
if (abs (a(1)) >= 300.d0) then
  if (a(1) > 0.d0) then
    write (ddldb, 1) a(1)
1   format ('*** DDEXP: Argument is too large',f12.6)
    call ddabrt
    return
  else
    call dddpddc (0.d0, b)
    goto 130
  endif
endif

f(1) = 1.d0
f(2) = 0.d0

!   Compute the reduced argument A' = A - Log(2) * Nint [A / Log(2)]. Save
!   NZ = Nint [A / Log(2)] for correcting the exponent of the final result.

call dddiv (a, al2, s0)
call ddnint (s0, s1)
t1 = s1(1)
nz = t1 + sign (1.d-14, t1)
call ddmul (al2, s1, s2)
call ddsub (a, s2, s0)

!   Check if the reduced argument is zero.

if (s0(1) == 0.d0) then
  s0(1) = 1.d0
  s0(2) = 0.d0
  l1 = 0
  goto 120
endif

!   Divide the reduced argument by 2 ^ NQ.

call dddivd (s0, 2.d0 ** nq, s1)

!   Compute Exp using the usual Taylor series.

s2(1) = 1.d0
s2(2) = 0.d0
s3(1) = 1.d0
s3(2) = 0.d0
l1 = 0

100  l1 = l1 + 1
if (l1 == 100) then
  write (ddldb, 2)
2 format ('*** DDEXP: Iteration limit exceeded.')
  call ddabrt
  return
endif

t2 = l1
call ddmul (s2, s1, s0)
call dddivd (s0, t2, s2)
call ddadd (s3, s2, s0)
call ddeq (s0, s3)

!   Check for convergence of the series.

if (abs (s2(1)) > eps * abs (s3(1))) goto 100

!   Raise to the (2 ^ NQ)-th power.

do i = 1, nq
  call ddmul (s0, s0, s1)
  s0(1) = s1(1)
  s0(2) = s1(2)
enddo

!  Multiply by 2 ^ NZ.

120  call ddmuld (s0, 2.d0 ** nz, b)

!   Restore original precision level.

130  continue
return
end subroutine ddexp

subroutine ddfformat (a, nb, nd, b) bind(C, name="ddfformat_fortran")

!   Converts the DDR number A into character form in the character(1) array B.
!   NB (input) is the length of the output string, and ND (input) is the
!   number of digits after the decimal point. The format is analogous to
!   Fortran F format; the result is right-justified among the NB cells of B.
!   The condition NB >= ND + 8 must hold or an error message will result.
!   However, if it is found during execution that there is not sufficient space,
!   to hold all digits, the entire output field will be filled with asterisks.
!   NB cells of type character(1) must be available in B.

implicit none
real(c_double), intent(in):: a(2)
integer, intent(in):: nb, nd
character(1), intent(out):: b(nb)
integer i, ixp, i1, i2, i3, j, k, nb2, nb3, nexp
character(1) b2(nb+20)
character(32) ca
real(c_double) t1

! End of declaration

if (nb < nd + 8) then
  write (ddldb, 1)
1 format ('*** DDFFORMAT: uninitialized or inadequately sized arrays')
  call ddabrt
endif

!   Call ddeformat with sufficient precision.

if (a(1) == 0.d0) then
  nb2 = nd + 11
else
  nb2 = max (log10 (abs (a(1))), 0.d0) + nd + 11
endif
nb3 = nb2 - 8
call ddeformat (a, nb2, nb3, b2)

!   Trim off trailing blanks.

do i = nb2, 1, -1
  if (b2(i) /= ' ') goto 90
enddo

90 continue

nb2 = i

!   Look for the 'e' in B2.

do k = 1, nb2
  if (b2(k) == 'e') goto 100
enddo

write (ddldb, 2)
2 format ('*** DDFFORMAT: Syntax error in output of ddeformat')
call ddabrt

100 continue

!   Check the sign of the exponent.

k = k + 1
if (b2(k) == '-') then
  ixp = -1
  k = k + 1
else
  ixp = 1
endif
j = 0
ca = ' '

!   Copy the exponent into CA.

do i = k, nb2
  j = j + 1
  if (j <= 16) ca(j:j) = b2(i)
enddo

t1 = dddigin (ca, j)

!   Check if there is enough space in the output array for all digits.

if (ixp == 1 .and. t1 + nd + 3 > nb) then
  do i = 1, nb
    b(i) = '*'
  enddo

  goto 210
endif
nexp = ixp * t1

!   Insert the sign of the number, if any.

i1 = 0
i2 = 0
if (b2(1) == '-') then
  i1 = i1 + 1
  b(i1) = '-'
  i2 = i2 + 1
endif

if (nexp == 0) then

!   Exponent is zero. Copy first digit, period and ND more digits.

  do i = 1, nd + 2
    i1 = i1 + 1
    i2 = i2 + 1
    b(i1) = b2(i2)
  enddo

  goto 200
elseif (nexp > 0) then

!   Exponent is positive. Copy first digit, skip the period, then copy
!   nexp digits.

  i1 = i1 + 1
  i2 = i2 + 1
  b(i1) = b2(i2)
  i2 = i2 + 1

  do i = 1, nexp
    i1 = i1 + 1
    i2 = i2 + 1
    b(i1) = b2(i2)
  enddo

!   Insert the period.

  i1 = i1 + 1
  b(i1) = '.'

!   Copy nd more digits.

  do i = 1, nd
    i1 = i1 + 1
    i2 = i2 + 1
    b(i1) = b2(i2)
  enddo

  goto 200
else

!   Exponent is negative. Insert a zero, then a period, then -1 - nexp
!   zeroes, then the first digit, then the remaining digits up to ND total
!   fractional digits.

  i1 = i1 + 1
  b(i1) = '0'
  i1 = i1 + 1
  b(i1) = '.'
  i3 = min (- nexp - 1, nd - 1)

  do i = 1, i3
    i1 = i1 + 1
    b(i1) = '0'
  enddo

  if (- nexp - 1 < nd) then
    i1 = i1 + 1
    i2 = i2 + 1
    b(i1) = b2(i2)
    i2 = i2 + 1
  endif

  do i = i3 + 2, nd
    i1 = i1 + 1
    i2 = i2 + 1
    b(i1) = b2(i2)
  enddo
endif

200 continue

!   Right-justify in field.

k = nb - i1

do i = 1, i1
  b(nb-i+1) = b(nb-i-k+1)
enddo

do i = 1, k
  b(i) = ' '
enddo

210 continue

return
end subroutine ddfformat

subroutine ddinfr (a, b, c) bind(C, name="ddinfr_fortran")

!   Sets B to the integer part of the DDR number A and sets C equal to the
!   fractional part of A. Note that if A = -3.3, then B = -3 and C = -0.3.

implicit none
real(c_double), intent(in):: a(2)
real(c_double), intent(out):: b(2), c(2)
real(c_double), parameter:: t105 = 2.d0 ** 105, t52 = 2.d0 ** 52
integer ic
real(c_double) con(2), f(2), s0(2), s1(2)
save con
data con / t105, t52/

!   Check if  A  is zero.

if (a(1) == 0.d0)  then
  b(1) = 0.d0
  b(2) = 0.d0
  c(1) = 0.d0
  c(2) = 0.d0
  goto 120
endif

if (a(1) >= t105) then
  write (ddldb, 1)
1 format ('*** DDINFR: Argument is too large.')
  call ddabrt
  return
endif

f(1) = 1.d0
f(2) = 0.d0
if (a(1) > 0.d0) then
  call ddadd (a, con, s0)
  call ddsub (s0, con, b)
  call ddcpr (a, b, ic)
  if (ic >= 0) then
    call ddsub (a, b, c)
  else
    call ddsub (b, f, s1)
    b(1) = s1(1)
    b(2) = s1(2)
    call ddsub (a, b, c)
  endif
else
  call ddsub (a, con, s0)
  call ddadd (s0, con, b)
  call ddcpr (a, b, ic)
  if (ic <= 0) then
    call ddsub (a, b, c)
  else
    call ddadd (b, f, s1)
    b(1) = s1(1)
    b(2) = s1(2)
    call ddsub (a, b, c)
  endif
endif

120  continue
return
end subroutine ddinfr

! subroutine ddinp (iu, a) bind(C, name="ddinp_fortran")

! !   This routine reads the DDR number A from logical unit IU. The input
! !   value must be placed on a single line of not more than 120 characters.

! implicit none
! integer, intent(in):: iu
! real(c_double), intent(out):: a(2)
! integer, parameter:: ln = 120
! character(120) cs

! read (iu, '(a)', end = 100) cs
! call ddinpc (cs, a)
! goto 110

! 100 write (ddldb, 1)
! 1  format ('*** DDINP: End-of-file encountered.')
! call ddabrt

! 110 return
! end subroutine ddinp

! subroutine ddinpc (a, n_a, b) bind(C, name="ddinpc_fortran")

! !   Converts the CHARACTER(*) array A into the DDR number B.
! use iso_c_binding, only: c_char
! implicit none
! character(kind=c_char), dimension(*), intent(in) :: a
! integer, intent(in):: n_a
! real(c_double), intent(out):: b(2)
! character(10), parameter:: dig = '0123456789'
! integer i, ib, id, ie, inz, ip, is, ix, k, ln, lnn
! real(c_double) bi
! character(1) ai
! character(10) ca
! real(c_double) f(2), s0(2), s1(2), s2(2)

! id = 0
! ip = -1
! is = 0
! inz = 0
! s1(1) = 0.d0
! s1(2) = 0.d0
! ln = n_a

! do i = ln, 1, -1
!   if (a(i:i) /= ' ') goto 90
! enddo

! 90 continue

! lnn = i

! ! write (6, '(a)') 'x'//a(1:lnn)//'x'

! !   Scan for digits, looking for the period also.

! do i = 1, lnn
!   ai = a(i:i)
!   if (ai == ' ' .and. id == 0) then
!   elseif (ai == '.') then
!     if (ip >= 0) goto 210
!     ip = id
!     inz = 1
!   elseif (ai == '+') then
!     if (id /= 0 .or. ip >= 0 .or. is /= 0) goto 210
!     is = 1
!   elseif (ai == '-') then
!     if (id /= 0 .or. ip >= 0 .or. is /= 0) goto 210
!     is = -1
!   elseif (ai == 'e' .or. ai == 'E' .or. ai == 'd' .or. ai == 'D') then
!     goto 100
!   elseif (index (dig, ai) == 0) then
!     goto 210
!   else
! !    read (ai, '(f1.0)') bi
!     bi = index (dig, ai) - 1
!     if (inz > 0 .or. bi > 0.d0) then
!       inz = 1
!       id = id + 1
!       call ddmuld (s1, 10.d0, s0)
!       f(1) = bi
!       f(2) = 0.d0
!       call dddpddc (bi, f)
!       call ddadd (s0, f, s1)
!     endif
!   endif
! enddo

! 100   continue
! if (is == -1) then
!   s1(1) = - s1(1)
!   s1(2) = - s1(2)
! endif
! k = i
! if (ip == -1) ip = id
! ie = 0
! is = 0
! ca = ' '

! do i = k + 1, lnn
!   ai = a(i:i)
!   if (ai == ' ') then
!   elseif (ai == '+') then
!     if (ie /= 0 .or. is /= 0) goto 210
!     is = 1
!   elseif (ai == '-') then
!     if (ie /= 0 .or. is /= 0) goto 210
!     is = -1
!   elseif (index (dig, ai) == 0) then
!     goto 210
!   else
!     ie = ie + 1
!     if (ie > 3) goto 210
!     ca(ie:ie) = ai
!   endif
! enddo

! ! read (ca, '(i4)') ie

! if (ca == ' ') then
!   ie = 0
! else
!   ie = dddigin (ca, ie)
! endif

! if (is == -1) ie = - ie
! ie = ie + ip - id
! s0(1) = 10.d0
! s0(2) = 0.d0
! call ddnpwr (s0, ie, s2)
! call ddmul (s1, s2, b)
! goto 220

! 210  write (ddldb, 1)
! 1 format ('*** DDINPC: Syntax error in literal string.')
! call ddabrt

! 220  continue

! return
! end subroutine ddinpc

subroutine ddlog (a, b) bind(C, name="ddlog_fortran")

!   This computes the natural logarithm of the DDR number A and returns the DD
!   result in B.

!   The Taylor series for Log converges much more slowly than that of Exp.
!   Thus this routine does not employ Taylor series, but instead computes
!   logarithms by solving Exp (b) = a using the following Newton iteration,
!   which converges to b:

!           x_{k+1} = x_k + [a - Exp (x_k)] / Exp (x_k)

implicit none
integer k
real(c_double), intent(in):: a(2)
real(c_double), intent(out):: b(2)
real(c_double) t1, t2
real(c_double) s0(2), s1(2), s2(2)

if (a(1) <= 0.d0) then
  write (ddldb, 1)
1 format ('*** DDLOG: Argument is less than or equal to zero.')
  call ddabrt
  return
endif

!   Compute initial approximation of Log (A).

t1 = a(1)
t2 = log (t1)
b(1) = t2
b(2) = 0.d0

!   Perform the Newton-Raphson iteration described above.

do k = 1, 3
  call ddexp (b, s0)
  call ddsub (a, s0, s1)
  call dddiv (s1, s0, s2)
  call ddadd (b, s2, s1)
  b(1) = s1(1)
  b(2) = s1(2)
enddo

120  continue

return
end subroutine ddlog

subroutine ddlog2c (alog2d) bind(C, name="ddlog2c_fortran")

!   This returns log(2) to DDR precision.

implicit none
real(c_double), intent(out):: alog2d(2)

alog2d(1) = ddlog2con(1)
alog2d(2) = ddlog2con(2)
return
end subroutine ddlog2c

subroutine ddmdc (a, b, n) bind(C, name="ddmdc_fortran")

!   This returns a DP approximation the DDR number A in the form B * 2^n.

implicit none
real(c_double), intent(in):: a(2)
real(c_double), intent(out):: b
integer, intent(out):: n

b = a(1)
n = 0
return
end subroutine ddmdc

subroutine ddneg (a, b) bind(C, name="ddneg_fortran")

!   This sets b = -a.

implicit none
real(c_double), intent(in):: a(2)
real(c_double), intent(out):: b(2)

b(1) = -a(1); b(2) = -a(2)
return
end subroutine ddneg

subroutine ddpower (a, b, c) bind(C, name="ddpower_fortran")

!   This computes C = A^B.

implicit none
real(c_double), intent(in):: a(2), b(2)
real(c_double), intent(out):: c(2)
real(c_double) t1(2), t2(2)
if (a(1) <= 0.d0) then
  write (6, 1)
1 format ('DDPOWER: A <= 0')
  call ddabrt
endif

call ddlog (a, t1)
! print at each step
call ddmul (t1, b, t2)
call ddexp (t2, c)
return
end subroutine ddpower

subroutine ddqqc (a, b, c) bind(C, name="ddqqc_fortran")

!   This converts DDR numbers A and B to DDC form in C, i.e. C = A + B i.

implicit none
real(c_double), intent(in):: a(2), b(2)
real(c_double), intent(out):: c(4)

c(1) = a(1)
c(2) = a(2)
c(3) = b(1)
c(4) = b(2)
return
end subroutine ddqqc

subroutine ddmul (dda, ddb, ddc) bind(C, name="ddmul_fortran")

!   This routine multiplies DDR numbers DDA and DDB to yield the DDR product DDC.

implicit none
real(c_double), intent(in):: dda(2), ddb(2)
real(c_double), intent(out):: ddc(2)
real(c_double), parameter:: split = 134217729.d0
real(c_double) a1, a2, b1, b2, cona, conb, c11, c21, c2, e, t1, t2

!   This splits dda(1) and ddb(1) into high-order and low-order words.

cona = dda(1) * split
conb = ddb(1) * split
a1 = cona - (cona - dda(1))
b1 = conb - (conb - ddb(1))
a2 = dda(1) - a1
b2 = ddb(1) - b1

!   Multilply dda(1) * ddb(1) using Dekker's method.

c11 = dda(1) * ddb(1)
c21 = (((a1 * b1 - c11) + a1 * b2) + a2 * b1) + a2 * b2
!>
!   Compute dda(1) * ddb(2) + dda(2) * ddb(1) (only high-order word is needed).

c2 = dda(1) * ddb(2) + dda(2) * ddb(1)

!   Compute (c11, c21) + c2 using Knuth's trick, also adding low-order product.

t1 = c11 + c2
e = t1 - c11
t2 = ((c2 - e) + (c11 - (t1 - e))) + c21 + dda(2) * ddb(2)

!   The result is t1 + t2, after normalization.

ddc(1) = t1 + t2
ddc(2) = t2 - (ddc(1) - t1)

return
end subroutine ddmul

subroutine ddmuld (dda, db, ddc) bind(C, name="ddmuld_fortran")

!   This routine multiplies the DDR number DDA by the DP number DB to yield
!   the DDR product DDC. DB must be an integer or exact binary fraction,
!   such as 3., 0.25 or -0.375.

implicit none
real(c_double), intent(in):: dda(2), db
real(c_double), intent(out):: ddc(2)
real(c_double), parameter:: split = 134217729.d0
real(c_double) a1, a2, b1, b2, cona, conb, c11, c21, c2, e, t1, t2

!   This splits dda(1) and db into high-order and low-order words.

cona = dda(1) * split
conb = db * split
a1 = cona - (cona - dda(1))
b1 = conb - (conb - db)
a2 = dda(1) - a1
b2 = db - b1

!   Multilply dda(1) * db using Dekker's method.

c11 = dda(1) * db
c21 = (((a1 * b1 - c11) + a1 * b2) + a2 * b1) + a2 * b2
!>
!   Compute dda(2) * db (only high-order word is needed).

c2 = dda(2) * db

!   Compute (c11, c21) + c2 using Knuth's trick.

t1 = c11 + c2
e = t1 - c11
t2 = ((c2 - e) + (c11 - (t1 - e))) + c21

!   The result is t1 + t2, after normalization.

ddc(1) = t1 + t2
ddc(2) = t2 - (ddc(1) - t1)

return
end subroutine ddmuld

subroutine ddmuldd (da, db, ddc) bind(C, name="ddmuldd_fortran")

!   This subroutine computes DDC = DA x DB, where DA and DB are quad and DDC is
!   douuble-double.

implicit none
real(c_double), intent(in):: da, db
real(c_double), intent(out):: ddc(2)
real(c_double), parameter:: split = 134217729.d0
real(c_double) a1, a2, b1, b2, cona, conb, s1, s2

!>
!   On systems with a fused multiply add, such as IBM systems, it is faster to
!   uncomment the next two lines and comment out the following lines until !>.
!   On other systems, do the opposite.

! s1 = da * db
! s2 = da * db - s1

!   This splits da and db into high-order and low-order words.

cona = da * split
conb = db * split
a1 = cona - (cona - da)
b1 = conb - (conb - db)
a2 = da - a1
b2 = db - b1

!   Multiply da * db using Dekker's method.

s1 = da * db
s2 = (((a1 * b1 - s1) + a1 * b2) + a2 * b1) + a2 * b2
!>
ddc(1) = s1
ddc(2) = s2

return
end subroutine ddmuldd

subroutine ddmzc (a, b) bind(C, name="ddmzc_fortran")

!  This converts the DDR real variable A to the DDC variable B.
!  This routine is not intended to be called directly by the user.

implicit none
real(c_double), intent(in):: a(2)
real(c_double), intent(out):: b(4)

b(1) = a(1)
b(2) = a(2)
b(3) = 0.d0
b(4) = 0.d0
return
end subroutine ddmzc

subroutine ddnint (a, b) bind(C, name="ddnint_fortran")

!   This sets B equal to the integer (type DDR) nearest to the DDR number A.

implicit none
real(c_double), intent(in):: a(2)
real(c_double), intent(out):: b(2)
real(c_double), parameter:: t105 = 2.d0 ** 105, t52 = 2.d0 ** 52
real(c_double) con(2), s0(2)
save con
data con / t105, t52/

!   Check if  A  is zero.

if (a(1) == 0.d0)  then
  b(1) = 0.d0
  b(2) = 0.d0
  goto 120
endif

if (a(1) >= t105) then
  write (ddldb, 1)
1 format ('*** DDNINT: Argument is too large.')
  call ddabrt
  return
endif

if (a(1) > 0.d0) then
  call ddadd (a, con, s0)
  call ddsub (s0, con, b)
else
  call ddsub (a, con, s0)
  call ddadd (s0, con, b)
endif

120  continue
return
end subroutine ddnint

subroutine ddnpwr (a, n, b) bind(C, name="ddnpwr_fortran")

!   This computes the N-th power of the DDR number A and returns the DDR result
!   in B. When N is zero, 1 is returned. When N is negative, the reciprocal
!   of A ^ |N| is returned. 

!   This routine employs the binary method for exponentiation.

implicit none
real(c_double), intent(in):: a(2)
integer(c_int), intent(in):: n
real(c_double), intent(out):: b(2)
real(c_double), parameter:: cl2 = 1.4426950408889633d0
integer j, kk, kn, l1, mn, na1, na2, nn
real(c_double) t1
real(c_double) s0(2), s1(2), s2(2), s3(2)

if (a(1) == 0.d0) then
  if (n >= 0) then
    s2(1) = 0.d0
    s2(2) = 0.d0
    goto 120
  else
    write (ddldb, 1)
1   format ('*** DDNPWR: Argument is zero and N is negative or zero.')
    call ddabrt
    return
  endif
endif

nn = abs (n)
if (nn == 0) then
  s2(1) = 1.d0
  s2(2) = 0.d0
  goto 120
elseif (nn == 1) then
  s2(1) = a(1)
  s2(2) = a(2)
  goto 110
elseif (nn == 2) then
  call ddmul (a, a, s2)
  goto 110
endif

!   Determine the least integer MN such that 2 ^ MN > NN.

t1 = nn
mn = cl2 * log (t1) + 1.d0 + 1.d-14
s0(1) = a(1)
s0(2) = a(2)
s2(1) = 1.d0
s2(2) = 0.d0
kn = nn

!   Compute B ^ N using the binary rule for exponentiation.

do j = 1, mn
  kk = kn / 2
  if (kn /= 2 * kk) then
    call ddmul (s2, s0, s1)
    s2(1) = s1(1)
    s2(2) = s1(2)
  endif
  kn = kk
  if (j < mn) then
    call ddmul (s0, s0, s1)
    s0(1) = s1(1)
    s0(2) = s1(2)
  endif
enddo

!   Compute reciprocal if N is negative.

110  continue

if (n < 0) then
  s1(1) = 1.d0
  s1(2) = 0.d0
  call dddiv (s1, s2, s0)
  s2(1) = s0(1)
  s2(2) = s0(2)
endif

120  continue

b(1) = s2(1)
b(2) = s2(2)
  
return
end subroutine ddnpwr

subroutine ddnrtf (a, n, b) bind(C, name="ddnrtf_fortran")

!   This computes the N-th root of the DDR number A and returns the DDR result
!   in B. N must be at least one.

!   This subroutine employs the following Newton-Raphson iteration, which
!   converges to A ^ (-1/N):

!    X_{k+1} = X_k + (X_k / N) * (1 - A * X_k^N)

!   The reciprocal of the final approximation to A ^ (-1/N) is the N-th root.

implicit none
real(c_double), intent(in):: a(2)
integer, intent(in):: n
real(c_double), intent(out):: b(2)
integer i, k
real(c_double) t1, t2, tn
real(c_double) f1(2), s0(2), s1(2)

if (a(1) == 0.d0) then
  b(1) = 0.d0
  b(2) = 0.d0
  goto 140
elseif (a(1) < 0.d0) then
  write (ddldb, 1)
1 format ('*** DDNRT: Argument is negative.')
  call ddabrt
  return
endif
if (n <= 0) then
  write (ddldb, 2) n
2 format ('*** DDNRT: Improper value of N',i10)
  call ddabrt
  return
endif

!   Handle cases N = 1 and 2.

if (n == 1) then
  b(1) = a(1)
  b(2) = a(1)
  goto 140
elseif (n == 2) then
  call ddsqrt (a, b)
  goto 140
endif

f1(1) = 1.d0
f1(2) = 0.d0

!   Compute the initial approximation of A ^ (-1/N).

tn = n
t1 = a(1)
t2 = exp (- log (t1) / tn)
b(1) = t2
b(2) = 0.d0

!   Perform the Newton-Raphson iteration described above.

do k = 1, 3
  call ddnpwr (b, n, s0)
  call ddmul (a, s0, s1)
  call ddsub (f1, s1, s0)
  call ddmul (b, s0, s1)
  call dddivd (s1, tn, s0)
  call ddadd (b, s0, s1)
  b(1) = s1(1)
  b(2) = s1(2)
enddo

!   Take the reciprocal to give final result.

call dddiv (f1, b, s1)
b(1) = s1(1)
b(2) = s1(2)

140  continue
return
end subroutine ddnrtf

subroutine ddout (iu, nb, nd, a) bind(C, name="ddout_fortran")

!   This routine writes the DDR number A on logical unit IU using a standard
!   Fortran Enb.nd format.

implicit none
integer, intent(in):: iu, nb, nd
real(c_double), intent(in):: a(2)
integer i
character(1) cs(nb)

call ddeformat (a, nb, nd, cs)
write (iu, '(120a1)') (cs(i), i = 1, nb)

return
end subroutine ddout

subroutine ddpic (pi) bind(C, name="ddpic_fortran")

!   This returns Pi to DDR precision.

implicit none
real(c_double), intent(out):: pi(2)

pi(1) = ddpicon(1)
pi(2) = ddpicon(2)

return
end subroutine ddpic

subroutine ddpolyr (n, a, x0, x) bind(C, name="ddpolyr_fortran")

!   This finds the root X, near X0 (input) for the nth-degree DDR polynomial
!   whose coefficients are given in the (n+1)-long vector A. It may be
!   necessary to adjust eps -- default value is 1.d-29.

implicit none
integer, intent(in):: n
real(c_double), intent(in):: a(2,0:n), x0(2)
real(c_double), intent(out):: x(2)
real(c_double), parameter:: eps = 1.d-29
integer i, it
real(c_double) ad(2,0:n), t1(2), t2(2), t3(2), t4(2), t5(2), dt1

do i = 0, n - 1
  dt1 = i + 1
  call ddmuld (a(1,i+1), dt1, ad(1,i))
enddo

ad(1,n) = 0.d0
ad(2,n) = 0.d0
x(1) = x0(1)
x(2) = x0(2)

do it = 1, 20
  t1(1) = 0.d0
  t1(2) = 0.d0
  t2(1) = 0.d0
  t2(2) = 0.d0
  t3(1) = 1.d0
  t3(2) = 0.d0

  do i = 0, n
    call ddmul (a(1,i), t3, t4)
    call ddadd (t1, t4, t5)
    t1(1) = t5(1)
    t1(2) = t5(2)
    call ddmul (ad(1,i), t3, t4)
    call ddadd (t2, t4, t5)
    t2(1) = t5(1)
    t2(2) = t5(2)
    call ddmul (t3, x, t4)
    t3(1) = t4(1)
    t3(2) = t4(2)
  enddo

  call dddiv (t1, t2, t3)
  call ddsub (x, t3, t4)
  x(1) = t4(1)
  x(2) = t4(2)
  if (abs (t3(1)) <= eps) goto 110
enddo

write (ddldb, 1)
1 format ('DDROOT: failed to converge.')


110 continue

return
end subroutine ddpolyr

integer function ddsgn (ra)

!   This function returns 1, 0 or -1, depending on whether ra > 0, ra = 0 or ra < 0.

implicit none
real(c_double), intent(in):: ra(2)
integer ia

if (ra(1) == 0.q0) then
  ddsgn = 0
elseif (ra(1) > 0.q0) then
  ddsgn = 1
else
  ddsgn = -1
endif
return
end function ddsgn

subroutine ddsqrt (a, b) bind(C, name="ddsqrt_fortran")

!   This computes the square root of the DDR number A and returns the DDR result
!   in B.

!   This subroutine employs the following formula (due to Alan Karp):

!          Sqrt(A) = (A * X) + 0.5 * [A - (A * X)^2] * X  (approx.)

!   where X is a double precision approximation to the reciprocal square root,
!   and where the multiplications A * X and [] * X are performed with only
!   double precision.

implicit none
real(c_double), intent(in):: a(2)
real(c_double), intent(out):: b(2)
real(c_double) t1, t2, t3
real(c_double) f(2), s0(2), s1(2)

if (a(1) == 0.d0) then
  b(1) = 0.d0
  b(2) = 0.d0
  goto 100
endif
t1 = 1.d0 / sqrt (a(1))
t2 = a(1) * t1
call ddmuldd (t2, t2, s0)
call ddsub (a, s0, s1)
t3 = 0.5d0 * s1(1) * t1
s0(1) = t2
s0(2) = 0.d0
s1(1) = t3
s1(2) = 0.d0
call ddadd (s0, s1, b)

100 continue

return
end subroutine ddsqrt

subroutine ddsub (dda, ddb, ddc) bind(C, name="ddsub_fortran")

!   This subroutine computes DDC = DDA - DDB, where all args are DDR.

implicit none 
real(c_double), intent(in):: dda(2), ddb(2)
real(c_double), intent(out):: ddc(2)
real(c_double) e, t1, t2

!   Compute dda + ddb using Knuth's trick.
t1 = dda(1) - ddb(1)
e = t1 - dda(1)
t2 = ((-ddb(1) - e) + (dda(1) - (t1 - e))) + dda(2) - ddb(2)

!   The result is t1 + t2, after normalization.

ddc(1) = t1 + t2
ddc(2) = t2 - (ddc(1) - t1)
return
end subroutine ddsub

subroutine ddxzc (a, b) bind(C, name="ddxzc_fortran")

!  This converts the DC variable A to the DDC variable B.
!  This routine is not intended to be called directly by the user.

implicit none
complex (ddknd), intent(in):: a
real(c_double), intent(out):: b(4)

b(1) = a
b(2) = 0.d0
b(3) = aimag (a)
b(4) = 0.d0
return
end subroutine ddxzc

subroutine dd_calc_scale_diff(result, expected, ires) bind(C, name="dd_calc_scale_diff_fortran")
  !-------------------------------------------------------------------
  ! dd_calc_scale_diff:
  !   Emulates the logic of the C++ "calculate_scale_difference" function.
  !   Takes two double-double variables, RES and EXP, each in the form
  !   of a 2-element array (res(1) = hi, res(2) = lo). Returns an integer
  !   measure of scale difference in the exponent.
  !
  !   If difference in the hi parts > 0, we compute:
  !      scale_difference = int( abs( log10( abs( res_hi - exp_hi ) )
  !                                  - log10( abs( exp_hi ) ) ) )
  !   If hi parts differ by exactly 0, we check the lo parts in a similar way,
  !   using exp_hi to gauge magnitude. If still zero, return 0.
  !
  !   In your unit test, compare this "ires" to a tolerance (like 30).
  !
  !    integer ires = dd_calc_scale_diff(result, expected)
  !    if ( ires >= 30 ) then
  !       ! Test passes
  !    else
  !       ! Test fails
  !    endif
  !
  !-------------------------------------------------------------------
    use, intrinsic :: iso_fortran_env, only: real64  ! or your ddknd parameter
    implicit none
    real(real64), intent(in) :: result(2), expected(2)   ! "double-double" arrays
    integer :: ires
    real(real64) :: error_hi_abs, error_lo_abs
    real(real64) :: error_hi_exp, error_lo_exp, exp_hi_exp
    real(real64) :: tiny = 0.0_real64
  !-------------------------------------------------------------------
  
    ! Default result
    ires = 0
  
    ! 1) Check difference in hi parts
    error_hi_abs = abs( result(1) - expected(1) )

    if (error_hi_abs > tiny) then
      error_hi_exp = log10(error_hi_abs)
      exp_hi_exp   = log10(abs( expected(1) ))
      ires = int( abs( error_hi_exp - exp_hi_exp ) )
      return
    end if
  
    ! 2) If hi difference is zero, check difference in lo parts
    error_lo_abs = abs( result(2) - expected(2) )
    if (error_lo_abs > tiny) then
       ! We'll still gauge the scale relative to exp(1) if that's nonzero
       if (abs( expected(1) ) > tiny) then
          error_lo_exp = log10(error_lo_abs)
          exp_hi_exp   = log10(abs( expected(1) ))
          ires = int( abs( error_lo_exp - exp_hi_exp ) )
          return
       else
          ! If expected(1)=0 but expected(2) is nonzero, you could do something else
          ! e.g. compare log10( error_lo_abs ) to log10( abs( exp(2) ) ).
          ! For brevity we do:
          if (abs(expected(2)) > tiny) then
             error_lo_exp = log10(error_lo_abs)
             exp_hi_exp   = log10(abs(expected(2)))
             ires = int( abs(error_lo_exp - exp_hi_exp) )
             return
          end if
          ! else if everything is zero except we have difference in lo => ires=999 or so
          ires = 999
          return
       end if
    end if
  
    ! 3) If both hi and lo differences are zero, return zero scale difference
    ires = 0
    return
  end subroutine dd_calc_scale_diff
  

end module ddfuna
