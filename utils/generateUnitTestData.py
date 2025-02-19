#!/usr/bin/env python3
import random
import struct
from mpmath import mp
import sys
import os
import argparse

mp.dps = 50   # set high precision

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

def generate_operator_test_data(filename, op, input_specs, num_cases=100):
   """
   Generate test data for an operator/function.
   Parameters:
      filename     : output binary file name.
      op           : a Python callable that takes inputs (from input_specs) and returns
                     a single mp number (or a tuple of mp numbers) as output.
      input_specs  : a list of dictionaries, one per input.
                     Each dict should have:
                        "type": one of the following: "mpf","mpc","double","int"
                        "min": minimum allowed value
                        "max": maximum allowed value
      num_cases    : number of test cases to generate.
   The output record contains:
      For each input of type "mpf": hi and lo (2 doubles),
      For each input of type "mpc": hi and lo for real and imaginary respectively (4 doubles),
      For each input of type "double": one double,
      For each input of type "int": one int,
      For each output (assumed to be mpf unless "mpc" appears in inputs): 
         if mpf: hi and lo (2 doubles)
         if mpc: real hi/lo and imag hi/lo (4 doubles).
   The fields are packed in the order: all inputs (in order) then all outputs.
   """
   # assume output is mpf unless any input is mpc
   output_type = "mpf"
   fmt = ""
   for spec in input_specs:
      if spec["type"] == "mpf":
         fmt += "dd"   # two doubles
      elif spec["type"] == "mpc":
         fmt += "dddd" # four doubles
         output_type = "mpc"
      elif spec["type"] == "double":
         fmt += "d"    # one double
      elif spec["type"] == "int":
         fmt += "i"    # one int
      else:
         sys.exit("Unknown input type in input_specs")
   
   # Determine output format from sample result.
   sample_inputs = []
   for spec in input_specs:
      if spec["type"] == "mpf":
         # Use the midpoint of the range.
         val = mp.mpf(spec["min"] + (spec["max"] - spec["min"]) / 2)
         sample_inputs.append(val)
      elif spec["type"] == "mpc":
         real = mp.mpf(spec["min"] + (spec["max"] - spec["min"]) / 2)
         imag = mp.mpf(spec["min"] + (spec["max"] - spec["min"]) / 2)
         sample_inputs.append(mp.mpc(real, imag))
      elif spec["type"] == "double":
         sample_inputs.append(mp.mpf(spec["min"] + (spec["max"] - spec["min"]) / 2))
      elif spec["type"] == "int":
         sample_inputs.append(int((spec["min"] + spec["max"]) // 2))
   sample_result = op(*sample_inputs)
   if not isinstance(sample_result, tuple):
      sample_result = (sample_result,)
   num_outputs = len(sample_result)
   for _ in range(num_outputs):
      if output_type == "mpc":
         fmt += "dddd"
      elif output_type == "mpf":
         fmt += "dd"
   
   # Ensure output directory exists.
   os.makedirs(os.path.dirname(filename), exist_ok=True)
   
   with open(filename, "wb") as f:
      for _ in range(num_cases):
         inputs = []
         # Generate each input according to its spec.
         for spec in input_specs:
            if spec["type"] == "mpf":
               r = mp.rand()
               val = mp.mpf(spec["min"]) + (mp.mpf(spec["max"]) - mp.mpf(spec["min"])) * r
               inputs.append(val)
            elif spec["type"] == "mpc":
               r = mp.rand()
               real = mp.mpf(spec["min"]) + (mp.mpf(spec["max"]) - mp.mpf(spec["min"])) * r
               imag = mp.mpf(spec["min"]) + (mp.mpf(spec["max"]) - mp.mpf(spec["min"])) * r
               val = mp.mpc(real, imag)
               inputs.append(val)
            elif spec["type"] == "double":
               val = mp.mpf(random.uniform(spec["min"], spec["max"]))
               inputs.append(val)
            elif spec["type"] == "int":
               val = random.randint(spec["min"], spec["max"])
               inputs.append(val)
         
         result = op(*inputs)
         if not isinstance(result, tuple):
            result = (result,)
         
         data = []
         for spec, inp in zip(input_specs, inputs):
            if spec["type"] == "mpf":
               hi, lo = get_double_double(inp)
               data.extend([hi, lo])
            elif spec["type"] == "mpc":
               real_hi, real_lo = get_double_double(inp.real)
               imag_hi, imag_lo = get_double_double(inp.imag)
               data.extend([real_hi, real_lo, imag_hi, imag_lo])
            elif spec["type"] == "double":
               data.append(float(inp))
            elif spec["type"] == "int":
               data.append(inp)
         for out in result:
            if output_type == "mpf":
               hi, lo = get_double_double(out)
               data.extend([hi, lo])
            elif output_type == "mpc":
               real_hi, real_lo = get_double_double(out.real)
               imag_hi, imag_lo = get_double_double(out.imag)
               data.extend([real_hi, real_lo, imag_hi, imag_lo])
         
         packed = struct.pack(fmt, *data)
         f.write(packed)
   print(f"Generated {num_cases} cases in {filename}")

def generate_all_test_data(N, output_path):
   # Two mpf inputs for basic arithmetic.
   two_mp_inputs = [
      {"type": "mpf", "min": -1, "max": 1},
      {"type": "mpf", "min": -1, "max": 1}
   ]
   # ddadd: addition operator.
   generate_operator_test_data(os.path.join(output_path, "ddadd.bin"),
      lambda a, b: a + b, two_mp_inputs, N)
   
   # ddsub: subtraction operator.
   generate_operator_test_data(os.path.join(output_path, "ddsub.bin"),
      lambda a, b: a - b, two_mp_inputs, N)
   
   # ddmul: multiplication operator.
   generate_operator_test_data(os.path.join(output_path, "ddmul.bin"), lambda a, b: a * b, two_mp_inputs, N)

   # ddmuld: multiplication operator double-double * double.
   generate_operator_test_data(os.path.join(output_path, "ddmuld.bin"), lambda a, b: a * b,
                               [{"type": "mpf", "min": -1, "max": 1}, {"type": "double", "min": -1, "max": 1}], N)
   
   # ddmuldd: multiplication operator double * double resul is in double-double.
   generate_operator_test_data(os.path.join(output_path, "ddmuldd.bin"), lambda a, b: a * b,
                               [{"type": "double", "min": -1, "max": 1}, {"type": "double", "min": -1, "max": 1}], N)
   
   # dddiv: division operator.
   generate_operator_test_data(os.path.join(output_path, "dddiv.bin"), lambda a, b: a / b if b != 0 else mp.mpf(0.), two_mp_inputs, N)

   # ddivd: division operator double-double / double.
   generate_operator_test_data(os.path.join(output_path, "dddivd.bin"), lambda a, b: a / b if b != 0 else mp.mpf(0.),
                               [{"type": "mpf", "min": -1, "max": 1}, {"type": "double", "min": -1, "max": 1}], N)
   
   # ddsqrt: one input; require nonnegative.
   generate_operator_test_data(os.path.join(output_path, "ddsqrt.bin"),
      lambda a: mp.sqrt(a), [{"type": "mpf", "min": 0, "max": 10}], N)
   
   # ddlog: one input; require > 0.
   generate_operator_test_data(os.path.join(output_path, "ddlog.bin"),
      lambda a: mp.log(a), [{"type": "mpf", "min": 0.1, "max": 1}], N)
   
   # ddexp: one input; safe range.
   generate_operator_test_data(os.path.join(output_path, "ddexp.bin"),
      lambda a: mp.exp(a), [{"type": "mpf", "min": -100, "max": 100}], N)
   
   # ddnint: one input.
   generate_operator_test_data(os.path.join(output_path, "ddnint.bin"),
      lambda a: mp.nint(a), [{"type": "mpf", "min": -100, "max": 100}], N)
   
   # ddnpwr: two inputs: one mpf and one int.
   npwr_inputs = [
      {"type": "mpf", "min": -1, "max": 1},
      {"type": "int", "min": -10, "max": 10}
   ]
   generate_operator_test_data(os.path.join(output_path, "ddnpwr.bin"),
      lambda a, n: a ** n, npwr_inputs, N)
   
   # ddpower: two inputs: two mpf values.
   power_inputs = [
      {"type": "mpf", "min": 0, "max": 1},
      {"type": "mpf", "min": -10, "max": 10}
   ]
   generate_operator_test_data(os.path.join(output_path, "ddpower.bin"),
      lambda a, n: a ** n, power_inputs, N)
   
   # ddabs: one input.
   generate_operator_test_data(os.path.join(output_path, "ddabs.bin"),
      lambda a: mp.fabs(a), [{"type": "mpf", "min": -10, "max": 10}], N)
   
   # ddacosh: one input.
   generate_operator_test_data(os.path.join(output_path, "ddacosh.bin"),
      lambda a: mp.acosh(a), [{"type": "mpf", "min": 1, "max": 10}], N)
   
   # ddasinh: one input.
   generate_operator_test_data(os.path.join(output_path, "ddasinh.bin"),
      lambda a: mp.asinh(a), [{"type": "mpf", "min": -10, "max": 10}], N)
   
   # ddatanh: one input.
   generate_operator_test_data(os.path.join(output_path, "ddatanh.bin"),
      lambda a: mp.atanh(a), [{"type": "mpf", "min": -0.999, "max": 0.999}], N)
   
   # ddcsshr: one input; returns two mpf outputs.
   # (For example, use mp.cosh and mp.sinh as the gold standard.)
   generate_operator_test_data(os.path.join(output_path, "ddcsshr.bin"),
      lambda a: (mp.cosh(a), mp.sinh(a)), [{"type": "mpf", "min": 0, "max": 10}], N)
   
   # ddcssnr: one input; returns two mpf outputs.
   # (For example, use mp.cos and mp.sin as the gold standard.)
   generate_operator_test_data(os.path.join(output_path, "ddcssnr.bin"),
      lambda a: (mp.cos(a), mp.sin(a)), [{"type": "mpf", "min": -mp.pi, "max": mp.pi}], N)
   
   # Complex math functions.
   # ddcadd: two mpc inputs.
   generate_operator_test_data(os.path.join(output_path, "ddcadd.bin"),
      lambda a, b: a + b, [{"type": "mpc", "min": -1, "max": 1},
                            {"type": "mpc", "min": -1, "max": 1}], N)
   
   # ddcsub: two mpc inputs.
   generate_operator_test_data(os.path.join(output_path, "ddcsub.bin"),
      lambda a, b: a - b, [{"type": "mpc", "min": -1, "max": 1},
                            {"type": "mpc", "min": -1, "max": 1}], N)
   
   # ddcmul: two mpc inputs.
   generate_operator_test_data(os.path.join(output_path, "ddcmul.bin"),
      lambda a, b: a * b, [{"type": "mpc", "min": -1, "max": 1},
                            {"type": "mpc", "min": -1, "max": 1}], N)
   
   # ddcdiv: two mpc inputs.
   generate_operator_test_data(os.path.join(output_path, "ddcdiv.bin"),
      lambda a, b: a / b if abs(b) != 0 else mp.mpc(0,0), [{"type": "mpc", "min": -1, "max": 1},
                                                         {"type": "mpc", "min": 0.1, "max": 1}], N)
   
   # ddcsqrt: one mpc input.
   generate_operator_test_data(os.path.join(output_path, "ddcsqrt.bin"),
      lambda a: mp.sqrt(a), [{"type": "mpc", "min": 0, "max": 10}], N)
   
   # ddcpwr: one mpc, one int input.
   generate_operator_test_data(os.path.join(output_path, "ddcpwr.bin"),
      lambda a, n: a ** n, [{"type": "mpc", "min": -1, "max": 1},
                            {"type": "int", "min": -10, "max": 10}], N)


if __name__ == "__main__":
   
   OUTPUT_PATH_DEFAULT = os.path.join(os.path.dirname(os.path.realpath(__file__)),"data")
   N_DEFAULT = 100
   
   args = argparse.ArgumentParser()
   args.add_argument("-o","--output-path",help=f"output path for binary file, DEFAULT={OUTPUT_PATH_DEFAULT}", default=OUTPUT_PATH_DEFAULT)
   args.add_argument("-n","--num-cases",help=f"number of cases to generate, DEFAULT={N_DEFAULT}", default=N_DEFAULT)
   args = args.parse_args()

   generate_all_test_data(args.num_cases, args.output_path)
