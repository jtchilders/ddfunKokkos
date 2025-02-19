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
   pi = mp.pi
   pi_hi,pi_lo = get_double_double(pi)
   print("PI:        ",string_double_double(pi_hi, pi_lo))
   e = mp.e
   e_hi, e_lo = get_double_double(e)
   print("e:         ",string_double_double(e_hi, e_lo))
   sqrt2 = mp.sqrt(2)
   sqrt2_hi, sqrt2_lo = get_double_double(sqrt2)
   print("sqrt(2):   ",string_double_double(sqrt2_hi, sqrt2_lo))

   with open(os.path.join(output_path, "constants.bin"), "wb") as f:
      f.write(struct.pack("dd", pi_hi, pi_lo))
      f.write(struct.pack("dd", e_hi, e_lo))
      f.write(struct.pack("dd", sqrt2_hi, sqrt2_lo))


def string_double_double(hi,lo):
   # Convert each double to its 64-bit integer representation and then to hex.
   hi_bits = hex(struct.unpack("<Q", struct.pack("<d", hi))[0])
   lo_bits = hex(struct.unpack("<Q", struct.pack("<d", lo))[0])
   return "[" + hi_bits + ", " + lo_bits + "]"

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