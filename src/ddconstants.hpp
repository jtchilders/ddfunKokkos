#include <cstdint>
#include <cstring>
#include "ddouble.hpp"

namespace ddfun {


ddouble make_ddouble(uint64_t hi_bits, uint64_t lo_bits) {
   double hi, lo;
   std::memcpy(&hi, &hi_bits, sizeof(double));
   std::memcpy(&lo, &lo_bits, sizeof(double));
   return ddfun::ddouble(hi, lo);
}

const ddouble PI     = make_ddouble(0x400921fb54442d18ULL, 0x3ca1a62633145c07ULL);
const ddouble E      = make_ddouble(0x4005bf0a8b145769ULL, 0x3ca4d57ee2b1013aULL);
const ddouble SQRT2  = make_ddouble(0x3ff6a09e667f3bcdULL, 0xbc9bdd3413b26456ULL);

}