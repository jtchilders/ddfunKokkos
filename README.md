# ddfunKokkos
A partial [Kokkos](https://github.com/kokkos/kokkos) port of the DDFUN library written by David H. Bailey (Berkeley Lab, UC-San Diego).
Written by J. Taylor Childers (Argonne National Laboratory)

# Overview

This port creates two C++ structs:
```C++
#include "ddouble.hpp"
#include "ddcomplex.hpp"

ddouble x, y(1), z(1,0);
ddouble x = y * z;
ddcomplex a,  b(ddouble(1,0),ddouble(1,0)),  c(ddouble(5,0),ddouble(10,0));
a = b / c;
std::cout << x << a << std::endl;
std::cout << x.sqrt() << std::endl;
```
These are intended to be used as effective datatypes.

# Dependencies

The header files only depend on a properly compiled Kokkos setup and included via `CMAKE_PREFIX_PATH`.

The Unit Tests depend on [Catch2](https://github.com/catchorg/Catch2) which will be downloaded and installed via the `ddfunKokkos` `cmake` process described below.

# Usage

Everything is defined in header files and therefore simply can be included without library compilation into one's own code. 

You can install the package by doing:
``` bash
cd <path/to/ddfunKokkos>
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$PWD/install
make -C build -j install
```

Once installed, you can add this to your environment:
```bash
export CMAKE_PREFIX_PATH=/path/to/ddfunKokkos/install/lib/cmake/ddfunKokkos
```
Then you can add this to your own `CMakeLists.txt`:
```cmake
find_package(ddfunKokkos REQUIRED)
target_link_libraries(MyTarget PRIVATE ddfunKokkos::ddfunKokkos)
```

And it should provide the proper include paths in your build commands.

# Pieces of DDFUN Implemented thus far:

## In [`src/ddouble.hpp`](./src/ddouble.hpp):
* Arithmetic: =, +, -, /, *, negation
* Math Functions: abs, sqrt, exp, log, power (exponent is a float), npwr (exponent is an integer)
* Trigonometric: acosh, asinh, atanh, csshr, cssnr
* Helper Functions:
   * ddmuld (`ddouble * double`, returns `ddouble`)
   * ddmuldd (`double * double`, returns `ddouble`)
   * dddivd (`ddouble / double`, returns `ddouble`)

## In [`src/ddcomplex.hpp`](./src/ddcomplex.hpp):
* Arithmetic: =, +, -, /, *, negation
* Functions: sqrt, pwr (exponent is an integer)

## In [`src/ddconstants.hpp`](./src/ddconstants.hpp):
* `PI`
* `e`
* `sqrt(2)`

# Unit Tests

Unit tests were included to validate the ported algorithms produce the same outputs as the fortran originals.

These unit tests use the [Catch2](https://github.com/catchorg/Catch2) framework which will be installed automatically by `cmake`.

These can be build using `cmake`:
``` bash
cd <path/to/ddfunKokkos>
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$PWD/install
make -C build -j install
```

They rely on binary inputs generated in full precision by some Python scripts. The [`generateUnitTestData.py](./utils/generateUnitTestData.py) script will produce individual binary files in the `utils/data` folder each containing inputs and expected outputs for each ddfun function. The unit tests will read these binary inputs, use the original Fortran and Kokkkos implementations to calculate the outputs for the given inputs, then compare these to the expected ouputs. Unit tests pass if the scale variation is larger than 20 digits of precision, which exceeds the double precision is 15-16 digits of precision.

Run the unit tests with:
```bash
./build/unitTests/unit_tests
```

Tested on JLSE V100 GPUs using Kokkos v4.5.01 and CUDA 12.3 and GCC 12.2.0

# DDFUN Lincensing with Update

An updated version of David's original license is [HERE](./DHB-License.txt)

