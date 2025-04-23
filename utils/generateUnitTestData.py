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
   
   # --------------------------------------------------------------------------
   # ddlog: one input; require > 0.
   log_edge_cases = [
      mp.mpf("1.0e-100"),        # Very small positive value
      mp.mpf("1.0") + mp.mpf("1.0e-17"),  # Value close to 1
      mp.mpf("1.0"),             # Exact value 1 (log should be 0)
      mp.mpf("1.23456789") + mp.mpf("1.23456789e-16"),  # Value with significant low-order bits
      mp.mpf("2.0"),             # log(2) edge case
      hex_to_ddouble("0x4149000020000000"), # 3276800.25 -> log(x) ≈ 15.002377
   ]
   
   # Generate regular random test cases
   generate_operator_test_data(os.path.join(output_path, "ddlog.bin"),
      lambda a: mp.log(a), [{"type": "mpf", "min": 0.1, "max": 1}], N)
   
   # Append edge cases to the log binary file
   with open(os.path.join(output_path, "ddlog.bin"), "ab") as f:
      for a in log_edge_cases:
         result = mp.log(a)
         print(f"   input: {a}   result: {result}")
         hi_a, lo_a = get_double_double(a)
         hi_r, lo_r = get_double_double(result)
         print(f"   HEX input: {hex(struct.unpack('>Q', struct.pack('>d', hi_a))[0])} {hex(struct.unpack('>Q', struct.pack('>d', lo_a))[0])}   HEX output: {hex(struct.unpack('>Q', struct.pack('>d', hi_r))[0])} {hex(struct.unpack('>Q', struct.pack('>d', lo_r))[0])}")
         packed = struct.pack("dddd", hi_a, lo_a, hi_r, lo_r)
         f.write(packed)
   print(f"   Appended {len(log_edge_cases)} edge cases to ddlog.bin")

   
   # --------------------------------------------------------------------------
   # ddexp: one input; safe range.
   exp_edge_cases = [
      mp.mpf("-100.0"),          # Large negative value (exp should be close to 0)
      mp.mpf("20.0"),            # Large positive value (exp should be very large)
      mp.mpf("0.0"),             # Exact value 0 (exp should be 1)
      mp.mpf("-1.0e-10"),        # Small negative value
      mp.mpf("-0.0001") - mp.mpf("1.234e-20"),  # Small negative with significant low bits
      mp.log(mp.mpf("2.0")),     # ln(2) - exp(ln(2)) should be 2
      mp.mpf("50.0") + mp.mpf("1.23e-14"),  # Large value with significant low bits
      hex_to_ddouble("0x3FEFFFFFA0000000"),
      hex_to_ddouble("0xC013000000000000"), # exp(x) ≈ 0.00865169520312
   ]
   
   # Generate regular random test cases
   generate_operator_test_data(os.path.join(output_path, "ddexp.bin"),
      lambda a: mp.exp(a), [{"type": "mpf", "min": -100, "max": 100}], N)
   
   # Append edge cases to the exp binary file
   with open(os.path.join(output_path, "ddexp.bin"), "ab") as f:
      for a in exp_edge_cases:
         result = mp.exp(a)
         hi_a, lo_a = get_double_double(a)
         hi_r, lo_r = get_double_double(result)
         packed = struct.pack("dddd", hi_a, lo_a, hi_r, lo_r)
         f.write(packed)
   print(f"    Appended {len(exp_edge_cases)} edge cases to ddexp.bin")
   
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

   # ddagmr: two mpf input, one mpf output.
   generate_operator_test_data(os.path.join(output_path, "ddagmr.bin"),
      lambda a, b: agmr(a, b), [{"type": "mpf", "min": 0, "max": 10},
                                {"type": "mpf", "min": 0, "max": 10}], N)
   
   # ddang: two mpf inputs (x,y coordinates), one mpf output (angle in radians).
   generate_operator_test_data(os.path.join(output_path, "ddang.bin"),
      lambda x, y: mp.atan2(y, x), [{"type": "mpf", "min": -10, "max": 10},
                                    {"type": "mpf", "min": -10, "max": 10}], N)
   
   # ddnrtf: one mpf input, one int input.
   generate_operator_test_data(os.path.join(output_path, "ddnrtf.bin"),
      lambda a, n: ddnrtf(a, n), [{"type": "mpf", "min": -10, "max": 10},
                                  {"type": "int", "min": -10, "max": 10}], N)

def agmr(a, b, max_iter=100, tol=mp.mpf('1e-32')):
   a = mp.mpf(a)
   b = mp.mpf(b)
   for i in range(max_iter):
      a_next = (a + b) / 2
      b_next = mp.sqrt(a * b)
      # Check convergence: if relative difference is small enough, return a_next.
      if mp.fabs(a_next - b_next) < tol * mp.fabs(a_next):
         return a_next
      a, b = a_next, b_next
   # If convergence is not reached, print a message and return current value.
   print("AGM did not converge within", max_iter, "iterations.")
   return a

def ddnrtf(a, n):
   # Implementation of the nth root algorithm from the MPFR manual:
   # https://www.mpfr.org/mpfr-current/mpfr.html#Nth-Root
   # The error is at most 1ulp.
   # The algorithm is:
   #   - k = 1
   #   - x = a
   #   - While k < n:
   #     - x = (x + a/x^(k-1))/k
   #     - k = k + 1
   #   - Return x
   a = mp.mpf(a)
   n = mp.mpf(n)
   x = a
   k = 1
   while k < n:
      x = (x + a / (x ** (k - 1))) / k
      k += 1
   return x

def ddpolyr(a, x0):
   """
   Solve a polynomial equation using the Newton-Raphson method.
   Intputs are mpf numbers:
      a: list of coefficients
      x0: initial guess
   Returns a tuple of two mpf numbers:
      x: root
   """
   mp0 = mp.mpf(0)
   mp1 = mp.mpf(1)
   ad = [mp0] * (len(a))
   for i in range(len(a)-1):
      ad[i] = a[i+1] * mp.mpf(i + 1)
   ad[len(a)-1] = mp0

   x = x0
   found = False
   for it in range(40):
      t1 = mp0
      t2 = mp0
      t3 = mp1

      for i in range(len(a)):
         t4 = a[i] * t3
         t5 = t1 + t4
         t1 = t5
         t4 = ad[i] * t3
         t5 = t2 + t4
         t2 = t5
         t4 = t3 * x
         t3 = t4
      
      t3 = t1 / t2
      t4 = x - t3
      x = t4
      if mp.fabs(t3) <= mp.mpf('1e-29'):
         return x
   print("DDPOLYR: failed to converge.")
   return None


def generate_ddpolyr_data(output_path):
    """Generate test data for ddpolyr (polynomial root finder)
    
    Args:
        output_path (str): Directory where the output file will be written
    """
    # Test cases:
    # 1. Simple quadratic: x^2 - 2 = 0 (root at sqrt(2))
    # 2. Cubic: x^3 - 2x^2 - 5x + 6 = 0 (roots at -2, 1, 3)
    # 3. Quartic: x^4 - 10x^3 + 35x^2 - 50x + 24 = 0 (roots at 1, 2, 3, 4)
    
    test_cases = [
        {
            'n': 2,
            'a': [mp.mpf('-2'), mp.mpf('0'), mp.mpf('1')],  # x^2 - 2
            'x0': mp.mpf('1.0'),  # Initial guess near sqrt(2)
        },
        {
            'n': 3,
            'a': [mp.mpf('6'), mp.mpf('-5'), mp.mpf('-2'), mp.mpf('1')],  # x^3 - 2x^2 - 5x + 6
            'x0': mp.mpf('2.5'),  # Initial guess near 3
        },
        {
            'n': 4,
            'a': [mp.mpf('24'), mp.mpf('-50'), mp.mpf('35'), mp.mpf('-10'), mp.mpf('1')],  # x^4 - 10x^3 + 35x^2 - 50x + 24
            'x0': mp.mpf('3.7'),
        },
        {
            'n': 4,
            'a': [mp.mpf('5.4'), mp.mpf('-333'), mp.mpf('2.3'), mp.mpf('-0.5'), mp.mpf('333')],  # 333x^4 - 0.5x^3 + 2.3x^2 - 333x + 5.4
            'x0': mp.mpf('3'),
        }
    ]
    
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Write test data to binary file
    output_file = os.path.join(output_path, 'ddpolyr.bin')
    with open(output_file, 'wb') as f:
        for case in test_cases:
            
            exp = ddpolyr(case['a'], case['x0'])
            # print(f"ddpolyr({case['a']}, {case['x0']}) = {exp}")
            if exp is None:
                continue
            
            n = case['n']
            a = case['a']
            x0 = case['x0']
            exp = ddpolyr(a, x0)

            output = [n,0]
            for x in a:
               hi,lo = get_double_double(x)
               output.append(hi)
               output.append(lo)
            hi,lo = get_double_double(x0)
            output.append(hi)
            output.append(lo)
            hi,lo = get_double_double(exp)
            output.append(hi)
            output.append(lo)
            
            fmt = 'ii' + ('d'*len(a)*2) + 'd'*2 + 'd'*2

            # print(f" case: {case}")
            # print(f" output: {output}")
            # print(f" fmt: {fmt}")
            
            f.write(struct.pack(
               fmt,
               *output
            ))

    
    print(f"Generated ddpolyr test data in {output_file}")

def hex_to_double(hex_str):
   """Convert a hexadecimal string representation of a double to a Python float."""
   # Remove '0x' prefix if present
   if hex_str.startswith('0x') or hex_str.startswith('0X'):
      hex_str = hex_str[2:]
   
   # Ensure the string is 16 characters (64 bits) by padding with zeros
   hex_str = hex_str.zfill(16)
   
   # Convert to an integer and then unpack as a double
   int_val = int(hex_str, 16)
   
   return struct.unpack('d', struct.pack('Q', int_val))[0]

def hex_to_ddouble(hi_hex, lo_hex=None):
   """
   Convert hexadecimal representations to a mpmath value.
   
   Parameters:
      hi_hex: Hexadecimal string for the high part
      lo_hex: Optional hexadecimal string for the low part (default: 0.0)
   
   Returns:
      mpmath.mpf value representing the double-double number
   """
   # Convert high part
   hi_val = hex_to_double(hi_hex)
   
   # Convert low part if provided
   if lo_hex:
      lo_val = hex_to_double(lo_hex)
   else:
      lo_val = 0.0
   
   # Create mpmath value with full precision
   return mp.mpf(hi_val) + mp.mpf(lo_val)

if __name__ == "__main__":
   
   OUTPUT_PATH_DEFAULT = os.path.join(os.path.dirname(os.path.realpath(__file__)),"data")
   N_DEFAULT = 100
   
   args = argparse.ArgumentParser()
   args.add_argument("-o","--output-path",help=f"output path for binary file, DEFAULT={OUTPUT_PATH_DEFAULT}", default=OUTPUT_PATH_DEFAULT)
   args.add_argument("-n","--num-cases",help=f"number of cases to generate, DEFAULT={N_DEFAULT}", default=N_DEFAULT)
   args = args.parse_args()

   generate_all_test_data(args.num_cases, args.output_path)
   generate_ddpolyr_data(args.output_path)
