#pragma once
#include <Kokkos_Core.hpp>
#include "ddouble.hpp"  // Make sure this header is in your include path

namespace ddfun {

struct ddcomplex {
   ddouble real;
   ddouble imag;

   // Constructors
   KOKKOS_INLINE_FUNCTION
   ddcomplex() : real(0.0), imag(0.0) {}

   KOKKOS_INLINE_FUNCTION
   ddcomplex(const ddouble &r) : real(r), imag(0.0) {}

   KOKKOS_INLINE_FUNCTION
   ddcomplex(const double &r) : real(r), imag(0.0) {}

   KOKKOS_INLINE_FUNCTION
   ddcomplex(const ddouble &r, const ddouble &i) : real(r), imag(i) {}

   KOKKOS_INLINE_FUNCTION
   ddcomplex(const double &r, const double &i) : real(r), imag(i) {}

   // Assignment operators
   KOKKOS_INLINE_FUNCTION
   ddcomplex& operator=(const ddcomplex &other) {
      real = other.real;
      imag = other.imag;
      return *this;
   }

   KOKKOS_INLINE_FUNCTION
   ddcomplex& operator=(const ddouble &other) {
      real = other;
      imag = ddouble(0.0);
      return *this;
   }

   // Arithmetic operators
   KOKKOS_INLINE_FUNCTION
   ddcomplex operator+(const ddcomplex &b) const {
      return ddcomplex(real + b.real, imag + b.imag);
   }

   KOKKOS_INLINE_FUNCTION
   ddcomplex operator-(const ddcomplex &b) const {
      return ddcomplex(real - b.real, imag - b.imag);
   }

   KOKKOS_INLINE_FUNCTION
   ddcomplex operator*(const ddcomplex &b) const {
      // (a+bi)*(c+di) = (a*c - b*d) + (a*d + b*c)i
      ddouble s0 = real * b.real;
      ddouble s1 = imag * b.imag;
      ddouble s2 = real * b.imag;
      ddouble s3 = imag * b.real;
      return ddcomplex(s0 - s1, s2 + s3);
   }

   KOKKOS_INLINE_FUNCTION
   ddcomplex operator/(const ddcomplex &b) const {
      // Check for division by zero.
      if (b.real.hi == 0.0 && b.imag.hi == 0.0) {
         Kokkos::printf("DDCOMPLEX: Division by zero.\n");
         return ddcomplex();
      }
      // Following the algorithm from the Fortran code.
      ddouble f = {1.0, 0.0};
      ddouble s0, s1, s2, s3, s4;
      s0 = real * b.real;
      s1 = imag * b.imag;
      s2 = s0 + s1;           // sum of products
      s3 = s0 - s1;           // difference of products
      s0 = real + imag;
      s1 = b.real - b.imag;
      s4 = s0 * s1;
      s1 = s4 - s3;
      // Compute denominator: b.real^2 + b.imag^2.
      s0 = b.real * b.real;
      s3 = b.imag * b.imag;
      s4 = s0 + s3;
      // Use division: f / s4.
      s0 = f / s4;
      return ddcomplex(s2 * s0, s1 * s0);
   }

   KOKKOS_INLINE_FUNCTION
   ddcomplex pwr(const int& n) const{
      /* 
!   This computes the N-th power of the ddC number A and returns the DDC
!   result C in B. When N is zero, 1 is returned. When N is negative, the
!   reciprocal of A ^ |N| is returned.

!   This routine employs the binary method for exponentiation.

      */
      double cl2 = 1.4426950408889633;
      ddcomplex s0,s1,s2,s3;

      if (real.hi == 0.0 && imag.hi == 0.0) {
         if (n >= 0)
            return ddcomplex();
         else{
            printf("DDCPWR: Argument is zero and N is negative or zero.\n");
            return ddcomplex();
         }
      }
      int nn = abs(n);
      if (nn == 0) {
         return ddcomplex(1.0);
      } else if (nn == 1) {
         s2 = *this;
         if (n < 0){ //   Compute reciprocal if N is negative.
            s1.real.hi = 1.0;
            s0 = s1 / s2;
            s2 = s0;
         }
         return s2;
      } else if (nn == 2) {
         s2 = *this * *this;
         if (n < 0){ //   Compute reciprocal if N is negative.
            s1.real.hi = 1.0;
            s0 = s1 / s2;
            s2 = s0;
         }
         return s2;
      }

      //   Determine the least integer MN such that 2 ^ MN > NN.

      int t1 = nn;
      int mn = cl2 * std::log(t1) + 1.0e0 + 1.0e-14;

      s0 = *this;
      s2.real.hi = 1.0;
      int kn = nn;


      //   Compute B ^ N using the binary rule for exponentiation.
      int kk = 0;
      for (int i = 1; i <= mn; i++) {
         kk = kn / 2;
         if (kn % 2 == 1) {
            s2 = s0 * s2;
         }
         kn = kk;
         if (i < mn) {
            s0 = s0 * s0;
         }
      }

      if (n < 0){ //   Compute reciprocal if N is negative.
         s1.real.hi = 1.0;
         s0 = s1 / s2;
         s2 = s0;
      }

      return s2;
   }

   KOKKOS_INLINE_FUNCTION
   ddcomplex sqrt() const {
      /*
   !   This routine computes the complex square root of the DDC number C.
   !   This routine uses the following formula, where A1 and A2 are the real and
   !   imaginary parts of A, and where R = Sqrt [A1 ^ 2 + A2 ^2]:

   !      B = Sqrt [(R + A1) / 2] + I Sqrt [(R - A1) / 2]

   !   If the imaginary part of A is < 0, then the imaginary part of B is also
   !   set to be < 0.
      */

      ddouble s0,s1,s2;
      ddcomplex b;

      if (real.hi == 0.0 && imag.hi == 0.0)
            return ddcomplex();

      s0 = real * real;
      s1 = imag * imag;
      s2 = s0 + s1;
      s0 = ddsqrt(s2);

      s1 = real;
      if (s1.hi < 0.0)
            s1 = -s1;
      s2 = s0 + s1;
      s1 = ddmuld(s2, 0.5);
      s0 = ddsqrt(s1);
      s1 = ddmuld(s0, 2.0);
      if (real.hi >= 0.0){
            b.real = s0;
            b.imag = imag / s1;
      } else {
            b.real = imag / s1;
            if ( b.real.hi < 0.0)
               b.real = - b.real;
            b.imag = s0;
            if (imag.hi < 0.0)
               b.imag = -b.imag;
      }

      return b;
   }
};

// Host-only print operator.
inline std::ostream& operator<<(std::ostream &os, const ddcomplex &c) {
   os << "(" << c.real << ") + (" << c.imag << ")i";
   return os;
}

} // end namespace ddfun
