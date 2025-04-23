#!/usr/bin/env python3

# generate a few standard constants in double-double precision and save to binary format
# usage: python3 generateConstants.py

#  list of constants:
#     pi
#     e
#     sqrt(2)


import random
import struct
from mpmath import mp
import sys
import os
import argparse

mp.dps = 50   # set higher than double-double precision

def generateConstants(output_path):
   # PI
   pi = mp.pi
   pi_hi,pi_lo = get_double_double(pi)
   print("KOKKOS_INLINE_FUNCTION ddouble get_dd_pi()    { return",string_double_double(pi_hi, pi_lo),"}")
   # E
   e = mp.e
   e_hi, e_lo = get_double_double(e)
   print("KOKKOS_INLINE_FUNCTION ddouble get_dd_e()     { return",string_double_double(e_hi, e_lo),"}")
   # sqrt(2)
   sqrt2 = mp.sqrt(2)
   sqrt2_hi, sqrt2_lo = get_double_double(sqrt2)
   print("KOKKOS_INLINE_FUNCTION ddouble get_dd_sqrt2() { return",string_double_double(sqrt2_hi, sqrt2_lo),"}")
   # log(2)
   log2 = mp.log(2)
   log2_hi, log2_lo = get_double_double(log2)
   print("KOKKOS_INLINE_FUNCTION ddouble get_dd_log2()  { return",string_double_double(log2_hi, log2_lo),"}")
   # log(10)
   log10 = mp.log(10)
   log10_hi, log10_lo = get_double_double(log10)
   print("KOKKOS_INLINE_FUNCTION ddouble get_dd_log10() { return",string_double_double(log10_hi, log10_lo),"}")
   # log(e)
   loge = mp.log(e)
   loge_hi, loge_lo = get_double_double(loge)
   print("KOKKOS_INLINE_FUNCTION ddouble get_dd_loge()  { return",string_double_double(loge_hi, loge_lo),"}")
   # Euler's constant
   euler = mp.euler
   euler_hi, euler_lo = get_double_double(euler)
   print("KOKKOS_INLINE_FUNCTION ddouble get_dd_euler() { return",string_double_double(euler_hi, euler_lo),"}")


   with open(os.path.join(output_path, "constants.bin"), "wb") as f:
      f.write(struct.pack("dd", pi_hi, pi_lo))
      f.write(struct.pack("dd", e_hi, e_lo))
      f.write(struct.pack("dd", sqrt2_hi, sqrt2_lo))
      f.write(struct.pack("dd", log2_hi, log2_lo))
      f.write(struct.pack("dd", log10_hi, log10_lo))
      f.write(struct.pack("dd", loge_hi, loge_lo))
      f.write(struct.pack("dd", euler_hi, euler_lo))


def string_double_double(hi,lo):
   # Convert each double to its 64-bit integer representation and then to hex.
   # then print them in a way to be cut & pasted into the C++ code following this format:
   # KOKKOS_INLINE_FUNCTION ddouble get_dd_pi()    { return make_ddouble_from_bits(0x400921fb54442d18ULL, 0x3ca1a62633145c07ULL); }
   hi_bits = hex(struct.unpack("<Q", struct.pack("<d", hi))[0])
   lo_bits = hex(struct.unpack("<Q", struct.pack("<d", lo))[0])
   return "make_ddouble_from_bits(" + hi_bits + "ULL, " + lo_bits + "ULL);"

def get_double_double(x):
   """
   Convert an mpmath mpf x to a double-double representation.
   hi is the standard double approximation; lo is the residual.
   """
   try:
      hi = float(x)
      lo = float(x - mp.mpf(hi))
   except:
      print(x)
      print(type(x))
      raise
   return hi, lo


if __name__ == "__main__":
   OUTPUT_PATH_DEFAULT = os.path.join(os.path.dirname(os.path.realpath(__file__)),"data")
   
   args = argparse.ArgumentParser()
   args.add_argument("-o","--output-path",help=f"output path for binary file, DEFAULT={OUTPUT_PATH_DEFAULT}", default=OUTPUT_PATH_DEFAULT)
   args = args.parse_args()
   generateConstants(args.output_path)