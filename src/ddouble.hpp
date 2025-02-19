#pragma once
#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <inttypes.h>
#include <vector>

namespace ddfun {

///////////////////////////////////////////////////////////////////////////////
// Forward declarations of ddmath functions operating on ddouble.
///////////////////////////////////////////////////////////////////////////////
struct ddouble;  // forward declaration

KOKKOS_INLINE_FUNCTION ddouble ddneg(const ddouble &a);
KOKKOS_INLINE_FUNCTION ddouble ddadd(const ddouble &a, const ddouble &b);
KOKKOS_INLINE_FUNCTION ddouble ddsub(const ddouble &a, const ddouble &b);
KOKKOS_INLINE_FUNCTION ddouble ddmul(const ddouble &a, const ddouble &b);
KOKKOS_INLINE_FUNCTION ddouble dddiv(const ddouble &a, const ddouble &b);
KOKKOS_INLINE_FUNCTION ddouble ddmuld(const ddouble &a, const double &b);
KOKKOS_INLINE_FUNCTION ddouble dddivd(const ddouble &a, const double &b);
KOKKOS_INLINE_FUNCTION ddouble ddexp(const ddouble &a);
KOKKOS_INLINE_FUNCTION ddouble ddlog(const ddouble &a);
KOKKOS_INLINE_FUNCTION ddouble ddnint(const ddouble &a);
KOKKOS_INLINE_FUNCTION ddouble ddabs(const ddouble &a);
KOKKOS_INLINE_FUNCTION ddouble ddsqrt(const ddouble &a);
KOKKOS_INLINE_FUNCTION ddouble ddnpwr(const ddouble &a, int n);
KOKKOS_INLINE_FUNCTION ddouble ddpower(const ddouble &a, const ddouble &b);
KOKKOS_INLINE_FUNCTION ddouble ddacosh(const ddouble &a);
KOKKOS_INLINE_FUNCTION ddouble ddasinh(const ddouble &a);
KOKKOS_INLINE_FUNCTION ddouble ddatanh(const ddouble &a);
KOKKOS_INLINE_FUNCTION ddouble ddmuldd(const double da, const double db);

///////////////////////////////////////////////////////////////////////////////
// The ddouble type.
///////////////////////////////////////////////////////////////////////////////
struct ddouble {
   double hi;
   double lo;

   KOKKOS_INLINE_FUNCTION
   ddouble() : hi(0.0), lo(0.0) {}

   KOKKOS_INLINE_FUNCTION
   ddouble(double h) : hi(h), lo(0.0) {}

   KOKKOS_INLINE_FUNCTION
   ddouble(double h, double l) : hi(h), lo(l) {}

   KOKKOS_INLINE_FUNCTION
   ddouble& operator=(const ddouble &other) {
      hi = other.hi;
      lo = other.lo;
      return *this;
   }

   KOKKOS_INLINE_FUNCTION
   ddouble operator-() const { return ddneg(*this); }

   KOKKOS_INLINE_FUNCTION
   ddouble operator+(const ddouble &b) const { return ddadd(*this, b); }

   KOKKOS_INLINE_FUNCTION
   ddouble operator-(const ddouble &b) const { return ddsub(*this, b); }

   KOKKOS_INLINE_FUNCTION
   ddouble operator*(const ddouble &b) const { return ddmul(*this, b); }

   KOKKOS_INLINE_FUNCTION
   ddouble operator/(const ddouble &b) const { return dddiv(*this, b); }

   KOKKOS_INLINE_FUNCTION
   ddouble operator*(const double &b) const { return ddmuld(*this, b); }

   KOKKOS_INLINE_FUNCTION
   ddouble operator/(const double &b) const { return dddivd(*this, b); }

   KOKKOS_INLINE_FUNCTION
   ddouble exp() const { return ddexp(*this); }

   KOKKOS_INLINE_FUNCTION
   ddouble log() const { return ddlog(*this); }

   KOKKOS_INLINE_FUNCTION
   ddouble nint() const { return ddnint(*this); }

   KOKKOS_INLINE_FUNCTION
   ddouble abs() const { return ddabs(*this); }

   KOKKOS_INLINE_FUNCTION
   ddouble sqrt() const { return ddsqrt(*this); }

   KOKKOS_INLINE_FUNCTION
   ddouble npwr(int n) const { return ddnpwr(*this, n); }

   KOKKOS_INLINE_FUNCTION
   ddouble power(const ddouble &b) const { return ddpower(*this, b); }

   KOKKOS_INLINE_FUNCTION
   ddouble acosh() const { return ddacosh(*this); }

   KOKKOS_INLINE_FUNCTION
   ddouble asinh() const { return ddasinh(*this); }

   KOKKOS_INLINE_FUNCTION
   ddouble atanh() const { return ddatanh(*this); }
};

// Host-only print operator.
inline std::ostream& operator<<(std::ostream &os, const ddouble &d) {
   os << "[ " << d.hi << ", " << d.lo << " ]";
   return os;
}

///////////////////////////////////////////////////////////////////////////////
// Inline definitions of ddmath functions.
///////////////////////////////////////////////////////////////////////////////

// ddneg
KOKKOS_INLINE_FUNCTION
ddouble ddneg(const ddouble &a) {
   return ddouble(-a.hi, -a.lo);
}

// ddadd: Knuth's trick.
KOKKOS_INLINE_FUNCTION
ddouble ddadd(const ddouble &a, const ddouble &b) {
   double t1 = a.hi + b.hi;
   double e = t1 - a.hi;
   double t2 = ((b.hi - e) + (a.hi - (t1 - e))) + a.lo + b.lo;
   double hi = t1 + t2;
   double lo = t2 - (hi - t1);
   return ddouble(hi, lo);
}

// ddsub.
KOKKOS_INLINE_FUNCTION
ddouble ddsub(const ddouble &a, const ddouble &b) {
   double t1 = a.hi - b.hi;
   double e = t1 - a.hi;
   double t2 = ((-b.hi - e) + (a.hi - (t1 - e))) + a.lo - b.lo;
   double hi = t1 + t2;
   double lo = t2 - (hi - t1);
   return ddouble(hi, lo);
}

// ddmul: using Dekker's splitting.
KOKKOS_INLINE_FUNCTION
ddouble ddmul(const ddouble &a, const ddouble &b) {
   const double split = 134217729.0;
   double cona = a.hi * split;
   double conb = b.hi * split;
   double a1 = cona - (cona - a.hi);
   double b1 = conb - (conb - b.hi);
   double a2 = a.hi - a1;
   double b2 = b.hi - b1;
   double c11 = a.hi * b.hi;
   double c21 = (((a1 * b1 - c11) + a1 * b2) + a2 * b1) + a2 * b2;
   double c2 = a.hi * b.lo + a.lo * b.hi;
   double t1 = c11 + c2;
   double e = t1 - c11;
   double t2 = ((c2 - e) + (c11 - (t1 - e))) + c21 + a.lo * b.lo;
   double hi = t1 + t2;
   double lo = t2 - (hi - t1);
   return ddouble(hi, lo);
}

// dddiv.
KOKKOS_INLINE_FUNCTION
ddouble dddiv(const ddouble &a, const ddouble &b) {
   const double split = 134217729.0;
   double s1 = a.hi / b.hi;
   double cona = s1 * split;
   double conb = b.hi * split;
   double a1 = cona - (cona - s1);
   double b1 = conb - (conb - b.hi);
   double a2 = s1 - a1;
   double b2 = b.hi - b1;
   double c11 = s1 * b.hi;
   double c21 = (((a1 * b1 - c11) + a1 * b2) + a2 * b1) + a2 * b2;
   double c2 = s1 * b.lo;
   double t1 = c11 + c2;
   double e = t1 - c11;
   double t2 = ((c2 - e) + (c11 - (t1 - e))) + c21;
   double t12 = t1 + t2;
   double t22 = t2 - (t12 - t1);
   double t11 = a.hi - t12;
   e = t11 - a.hi;
   double t21 = ((-t12 - e) + (a.hi - (t11 - e))) + a.lo - t22;
   double s2 = (t11 + t21) / b.hi;
   double hi = s1 + s2;
   double lo = s2 - (hi - s1);
   return ddouble(hi, lo);
}

// ddmuld.
KOKKOS_INLINE_FUNCTION
ddouble ddmuld(const ddouble &a, const double &b) {
   const double split = 134217729.0;
   double cona = a.hi * split;
   double conb = b * split;
   double a1 = cona - (cona - a.hi);
   double b1 = conb - (conb - b);
   double a2 = a.hi - a1;
   double b2 = b - b1;
   double c11 = a.hi * b;
   double c21 = (((a1 * b1 - c11) + a1 * b2) + a2 * b1) + a2 * b2;
   double c2 = a.lo * b;
   double t1 = c11 + c2;
   double e = t1 - c11;
   double t2 = ((c2 - e) + (c11 - (t1 - e))) + c21;
   double hi = t1 + t2;
   double lo = t2 - (hi - t1);
   return ddouble(hi, lo);
}

// dddivd.
KOKKOS_INLINE_FUNCTION
ddouble dddivd(const ddouble &a, const double &b) {
   const double split = 134217729.0;
   double t1 = a.hi / b;
   double cona = t1 * split;
   double conb = b * split;
   double a1 = cona - (cona - t1);
   double b1 = conb - (conb - b);
   double a2 = t1 - a1;
   double b2 = b - b1;
   double t12 = t1 * b;
   double t22 = (((a1 * b1 - t12) + a1 * b2) + a2 * b1) + a2 * b2;
   double t11 = a.hi - t12;
   double e = t11 - a.hi;
   double t21 = ((-t12 - e) + (a.hi - (t11 - e))) + a.lo - t22;
   double t2 = (t11 + t21) / b;
   double hi = t1 + t2;
   double lo = t2 - (hi - t1);
   return ddouble(hi, lo);
}

// ddexp.
KOKKOS_INLINE_FUNCTION
ddouble ddexp(const ddouble &a) {
   const int nq = 6;
   int i, l1, nz;
   double eps = 1.0e-32;
   ddouble al2 = {0.69314718055994529, 2.3190468138462996e-17};
   ddouble f = {1.0, 0.0};
   ddouble s0, s1, s2, s3, result;
   if (Kokkos::fabs(a.hi) >= 300.0) {
      if (a.hi > 0.0) {
         Kokkos::printf("DDEXP: Argument is too large\n");
         return ddouble();
      } else {
         return ddouble();
      }
   }
   s0 = dddiv(a, al2);
   s1 = ddnint(s0);
   double t1_val = s1.hi;
   nz = static_cast<int>(t1_val + copysign(1e-14, t1_val));
   s2 = ddmul(al2, s1);
   s0 = ddsub(a, s2);
   if (s0.hi == 0.0) {
      s0 = {1.0, 0.0};
      l1 = 0;
   } else {
      s1 = dddivd(s0, ldexp(1.0, nq)); // ldexp(1.0, nq) == 2^nq
      s2 = {1.0, 0.0};
      s3 = {1.0, 0.0};
      l1 = 0;
      do {
         l1 = l1 + 1;
         if (l1 == 100) {
            Kokkos::printf("DDEXP: Iteration limit exceeded\n");
            return ddouble(0.0);
         }
         double t2 = static_cast<double>(l1);
         s0 = ddmul(s2, s1);
         s2 = dddivd(s0, t2);
         s0 = ddadd(s3, s2);
         s3 = s0;
      } while (Kokkos::fabs(s2.hi) > eps * Kokkos::fabs(s3.hi));
      for (i = 0; i < nq; i++) {
         s1 = ddmul(s0, s0);
         s0 = s1;
      }
   }
   double pow_nz = ldexp(1.0, nz); // ldexp(1.0, nz) == 2^nz
   result = ddmuld(s0, pow_nz);
   return result;
}

// ddlog.
KOKKOS_INLINE_FUNCTION
ddouble ddlog(const ddouble &a) {
   if (a.hi <= 0.0) {
      Kokkos::printf("*** DDLOG: Argument is less than or equal to zero.\n");
      return ddouble();
   }
   ddouble b;
   double t1 = a.hi;
   double t2 = Kokkos::log(t1);
   b.hi = t2;
   b.lo = 0.0;
   for (int k = 1; k <= 3; ++k) {
      ddouble s0 = ddexp(b);
      ddouble s1 = ddsub(a, s0);
      ddouble s2 = dddiv(s1, s0);
      ddouble s1_new = ddadd(b, s2);
      b.hi = s1_new.hi;
      b.lo = s1_new.lo;
   }
   return b;
}

// ddmuldd.
KOKKOS_INLINE_FUNCTION
ddouble ddmuldd(const double da, const double db) {
   const double split = 134217729.0;
   double cona = da * split;
   double conb = db * split;
   double a1 = cona - (cona - da);
   double b1 = conb - (conb - db);
   double a2 = da - a1;
   double b2 = db - b1;
   double s1 = da * db;
   double s2 = (((a1 * b1 - s1) + a1 * b2) + a2 * b1) + a2 * b2;
   ddouble ddc;
   ddc.hi = s1;
   ddc.lo = s2;
   return ddc;
}

// ddacosh.
KOKKOS_INLINE_FUNCTION
ddouble ddacosh(const ddouble& a) {
   if (a.hi < 1.0) {
     Kokkos::printf("DDACOSH: Argument is < 1.\n");
     return ddouble();
   }
 
   ddouble f1{1.0, 0.0};
   ddouble t1, t2, b;
 
   t1 = ddmul(a, a);
   t2 = ddsub(t1, f1);
   t1 = ddsqrt(t2);
   t2 = ddadd(a, t1);
   b = ddlog(t2);
 
   return b;
 }

// ddasinh.
KOKKOS_INLINE_FUNCTION
ddouble ddasinh(const ddouble &a) {
   ddouble f1, t1, t2, b;
   f1.hi = 1.0; f1.lo = 0.0;
   t1 = ddmul(a, a);
   t2 = ddadd(t1, f1);
   t1 = ddsqrt(t2);
   t2 = ddadd(a, t1);
   b = ddlog(t2);
   return b;
}

// ddatanh.
KOKKOS_INLINE_FUNCTION
ddouble ddatanh(const ddouble &a) {
   if (Kokkos::fabs(a.hi) >= 1.0) {
      Kokkos::printf("DDATANH: Argument is <= -1 or >= 1.\n");
      return ddouble();
   }
   ddouble f1 = {1.0, 0.0};
   ddouble t1, t2, t3;
   t1 = ddadd(f1, a);
   t2 = ddsub(f1, a);
   t3 = dddiv(t1, t2);
   t1 = ddlog(t3);
   return ddmuld(t1, 0.5);
}

///////////////////////////////////////////////////////////////////////////////
// ddnint.
///////////////////////////////////////////////////////////////////////////////
KOKKOS_INLINE_FUNCTION
ddouble ddnint(const ddouble &a) {
   ddouble b = {0.0, 0.0};
   ddouble s0;
   if (a.hi == 0.0) {
      return b;
   }
   double T105 = ldexp(1.0, 105); // 2^105
   double T52  = ldexp(1.0, 52);  // 2^52
   ddouble CON = {T105, T52};
   if (a.hi >= T105) {
      Kokkos::printf("*** DDNINT: Argument is too large.\n");
      return b;
   }
   if (a.hi > 0.0) {
      s0 = ddadd(a, CON);
      b = ddsub(s0, CON);
   } else {
      s0 = ddsub(a, CON);
      b = ddadd(s0, CON);
   }
   return b;
}

// ddabs.
KOKKOS_INLINE_FUNCTION
ddouble ddabs(const ddouble& a) {
   ddouble b;
   if (a.hi >= 0.0) {
      b.hi = a.hi;
      b.lo = a.lo;
   } else {
      b.hi = -a.hi;
      b.lo = -a.lo;
   }
   return b;
}

// sqrt.
KOKKOS_INLINE_FUNCTION
ddouble ddsqrt(const ddouble& a) {

   if (a.hi == 0.0) {
      return ddouble();
   }

   double t1 = 1.0 / sqrt(a.hi);
   double t2 = a.hi * t1;
   ddouble s0 = ddmuldd(t2, t2);
   ddouble s1 = ddsub(a, s0);
   double t3 = 0.5 * s1.hi * t1;
   s0.hi = t2;
   s0.lo = 0.0;
   s1.hi = t3;
   s1.lo = 0.0;
   ddouble b = ddadd(s0, s1);

   return b;
}

// ddnpwr.
KOKKOS_INLINE_FUNCTION
ddouble ddnpwr(const ddouble &a, int n) {
   const double cl2 = 1.4426950408889633;
   ddouble s0, s1, s2 = {1.0, 0.0};
   double t1;
   int nn, mn, kn, kk;
   if (a.hi == 0.0) {
      if (n >= 0) {
         return {0.0, 0.0};
      } else {
         Kokkos::printf("*** DDNPWR: Argument is zero and N is negative or zero.\n");
         return ddouble();
      }
   }
   nn = (n < 0) ? -n : n;
   if (nn == 0) {
      return {1.0, 0.0};
   } else if (nn == 1) {
      if(n > 0)
         return a;
      else
         return dddiv(ddouble(1.0, 0.0), a);
   } else if (nn == 2) {
      if(n > 0)
         return ddmul(a, a);
      else
         return dddiv(ddouble(1.0, 0.0), ddmul(a, a));
   }
   t1 = static_cast<double>(nn);
   mn = static_cast<int>(cl2 * log(t1) + 1.0 + 1.0e-14);
   s0 = a;
   kn = nn;
   for (int j = 1; j <= mn; ++j) {
      kk = kn / 2;
      if (kn != 2 * kk) {
         s1 = ddmul(s2, s0);
         s2 = s1;
      }
      kn = kk;
      if (j < mn) {
         s1 = ddmul(s0, s0);
         s0 = s1;
      }
   }
   if (n < 0) {
      s1 = {1.0, 0.0};
      s0 = dddiv(s1, s2);
      s2 = s0;
   }
   return s2;
}

// ddpower.
KOKKOS_INLINE_FUNCTION
ddouble ddpower(const ddouble &a, const ddouble &b) {
   if (a.hi <= 0.0) {
      Kokkos::printf("DDPOWER: A <= 0\n");
      return ddouble();
   }
   ddouble t1 = ddlog(a);
   ddouble t2 = ddmul(t1, b);
   ddouble c = ddexp(t2);
   return c;
}


///////////////////////////////////////////////////////////////////////////////
// Optionally, functions for hyperbolic cosine/sine.
///////////////////////////////////////////////////////////////////////////////
KOKKOS_INLINE_FUNCTION
void ddcsshr(const ddouble &a, ddouble &x, ddouble &y) {
   ddouble f(1.0, 0.0);
   ddouble s0, s1, s2;
   s0 = ddexp(a);
   s1 = dddiv(f, s0);
   s2 = ddadd(s0, s1);
   x = ddmuld(s2, 0.5);
   s2 = ddsub(s0, s1);
   y = ddmuld(s2, 0.5);
}

KOKKOS_INLINE_FUNCTION
void ddcssnr(const ddouble &a, ddouble &x, ddouble &y) {
   const int itrmx = 1000, nq = 5;
   const double eps = 1.0e-32;
   const ddouble pi = {3.1415926535897931, 1.2246467991473532e-16};
   int na = 2;
   if(a.hi == 0.0) na = 0;
   if(na == 0) {
      x = {1.0, 0.0};
      y = {0.0, 0.0};
      return;
   }
   ddouble f1 = {1.0, 0.0}, f2 = {0.5, 0.0};
   if(a.hi >= 1.0e60) {
      Kokkos::printf("*** DDCSSNR: argument is too large to compute cos or sin.\n");
      return;
   }
   ddouble s0 = ddmuld(pi, 2.0);
   ddouble s1 = dddiv(a, s0);
   ddouble s2 = ddnint(s1);
   ddouble s3 = ddsub(a, ddmul(s0, s2));
   if(s3.hi == 0.0) {
      x = {1.0, 0.0};
      y = {0.0, 0.0};
      return;
   }
   s0 = dddivd(s3, ldexp(1.0, nq));  // ldexp(1.0, nq) == 2^nq
   s1 = s0;
   ddouble s2_squared = ddmul(s0, s0);
   int is = (s0.hi < 0.0) ? -1 : 1;
   for (int i1 = 1; i1 <= itrmx; ++i1) {
      double t2 = -(2.0 * i1) * (2.0 * i1 + 1.0);
      ddouble s3_temp = ddmul(s2_squared, s1);
      s1 = dddivd(s3_temp, t2);
      s3_temp = ddadd(s1, s0);
      s0 = s3_temp;
      if (Kokkos::fabs(s1.hi) < eps) break;
      if (i1 == itrmx) {
         Kokkos::printf("*** DDCSSNR: Iteration limit exceeded.\n");
         return;
      }
   }
   ddouble s4 = ddmul(s0, s0);
   ddouble s5 = ddsub(f2, s4);
   s0 = ddmuld(s5, 2.0);
   for (int j = 2; j <= nq; ++j) {
      s4 = ddmul(s0, s0);
      s5 = ddsub(s4, f2);
      s0 = ddmuld(s5, 2.0);
   }
   s4 = ddmul(s0, s0);
   s5 = ddsub(f1, s4);
   s1 = ddsqrt(s5);
   if (is < 1) {
      s1.hi = -s1.hi;
      s1.lo = -s1.lo;
   }
   x = s0;
   y = s1;
}


} // end namespace ddfun
