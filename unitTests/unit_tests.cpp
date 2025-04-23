#include <Kokkos_Core.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_session.hpp>
#include "ddouble.hpp"
#include "ddcomplex.hpp"
#include "ddrand.hpp"
#include <fstream>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>


#define CATCH_CONFIG_RUNNER
int main(int argc, char* argv[]) {
   // Initialize Kokkos using ScopeGuard so that Finalize is called automatically.
   Kokkos::initialize();
   // Run Catch2 tests.
   int result = Catch::Session().run(argc, argv);
   Kokkos::finalize();
   return result;
}


// Fortran routines must be exported with ISO_C_BINDING.
// Uncomment and update the prototypes as needed.
extern "C" {
   // math operators
   void ddadd_fortran(const double a[2], const double b[2], double c[2]);
   void ddsub_fortran(const double a[2], const double b[2], double c[2]);
   void ddmul_fortran(const double a[2], const double b[2], double c[2]);
   void ddmuld_fortran(const double a[2], const double& b, double c[2]);
   void ddmuldd_fortran(const double& a, const double& b, double c[2]);
   void dddiv_fortran(const double a[2], const double b[2], double c[2]);
   void dddivd_fortran(const double a[2], const double& b, double c[2]);

   // For math functions:
   void ddexp_fortran(const double a[2], double b[2]);
   void ddlog_fortran(const double a[2], double b[2]);
   void ddnint_fortran(const double a[2], double b[2]);
   void ddabs_fortran(const double a[2], double b[2]);
   void ddsqrt_fortran(const double a[2], double b[2]);
   void ddnpwr_fortran(const double a[2], const int& b, double c[2]);
   void ddpower_fortran(const double a[2], const double b[2], double c[2]);
   void ddagmr_fortran(const double a[2], const double b[2], double c[2]);
   void ddang_fortran(const double a[2], const double b[2], double c[2]);

   // For trig functions
   void ddacosh_fortran(const double a[2], double b[2]);
   void ddasinh_fortran(const double a[2], double b[2]);
   void ddatanh_fortran(const double a[2], double b[2]);
   void ddcsshr_fortran(const double a[2], double b[2], double c[2]);
   void ddcssnr_fortran(const double a[2], double b[2], double c[2]);

   // complex math operators
   void ddcadd_fortran(const double a[4], const double b[4], double c[4]);
   void ddcsub_fortran(const double a[4], const double b[4], double c[4]);
   void ddcmul_fortran(const double a[4], const double b[4], double c[4]);
   void ddcdiv_fortran(const double a[4], const double b[4], double c[4]);

   // complex math functions
   void ddcpwr_fortran(const double a[4], const int& b, double c[4]);
   void ddcsqrt_fortran(const double a[4], double b[4]);
   void ddpolyr_fortran(const int& n, const double* a, const double x0[2], double x[2]);
}


std::string double_to_hex(double x) {
   union {
      double d;
      uint64_t u;
   } conv;
   conv.d = x;
   std::stringstream ss;
   ss << "0x" << std::hex << std::setw(16) << std::setfill('0') << conv.u;
   return ss.str();
}

std::string ddouble_to_hex(ddfun::ddouble x) {
   std::stringstream ss;
   ss << "[" << double_to_hex(x.hi) << ", " << double_to_hex(x.lo) << "]";
   return ss.str();
}

const std::string INPUT_FILES_DIR("/home/jchilders/git/ddfunKokkos/utils/data/");
const int REQUIRED_SCALE_PRECISION = 20;

// Helper function to compute scale difference.
static int calculate_scale_difference(const ddfun::ddouble &result, const ddfun::ddouble &expected) {
   double error_hi = std::fabs(result.hi - expected.hi);
   if (error_hi > 0.0) {
      double error_hi_exp = std::log10(error_hi);
      double expected_hi_exp = std::log10(std::fabs(expected.hi));
      return static_cast<int>(std::fabs(error_hi_exp - expected_hi_exp));
   }
   double error_lo = std::fabs(result.lo - expected.lo);
   if (error_lo > 0.0) {
      double error_lo_exp = std::log10(error_lo);
      double expected_hi_exp = std::log10(std::fabs(expected.hi));
      return static_cast<int>(std::fabs(error_lo_exp - expected_hi_exp));
   }
   return 0;
}



TEST_CASE("ddadd on device using mirror views", "[kokkos][ddouble]") {

   // Structure to hold ddadd test data. 
   // Since ddfun::ddouble is just two contiguous doubles, this struct is tightly packed.
   struct DataRecord {
      ddfun::ddouble a;
      ddfun::ddouble b;
      ddfun::ddouble exp;
   };

   // Read test data from file into a vector of DataRecord.
   std::string filename = INPUT_FILES_DIR + "ddadd.bin";
   std::ifstream infile(filename, std::ios::binary);
   INFO("Reading " << filename);
   REQUIRE(infile.good());
   std::vector<DataRecord> hostRecords;
   std::vector<ddfun::ddouble> fortranResults;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      // add data to inputs
      hostRecords.push_back(rec);
      // run Fortran to validate the results
      double a[2] = {rec.a.hi, rec.a.lo};
      double b[2] = {rec.b.hi, rec.b.lo};
      double c[2] = {0.0, 0.0};
      ddadd_fortran(a, b, c);
      fortranResults.push_back(ddfun::ddouble(c[0], c[1]));
   }
   int N = hostRecords.size();
   INFO("Read " << N << " records.");
   REQUIRE(N > 0);

   // Create a host-side Kokkos::View from the vector.
   Kokkos::View<ddfun::ddouble*, Kokkos::HostSpace> hA("hA", N);
   Kokkos::View<ddfun::ddouble*, Kokkos::HostSpace> hB("hB", N);

   for (int i = 0; i < N; i++) {
      hA(i) = hostRecords[i].a;
      hB(i) = hostRecords[i].b;
   }

   // Create a device view by deep copying the host view.
   auto dA = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hA);
   auto dB = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hB);

   // Create a device view to hold the computed results.
   Kokkos::View<ddfun::ddouble*, Kokkos::DefaultExecutionSpace> dResults("dResults", N);

   // Launch a kernel to compute ddadd on each input.
   Kokkos::parallel_for("compute_ddadd", N, KOKKOS_LAMBDA(const int i) {
      dResults(i) = dA(i) + dB(i);
   });
   Kokkos::fence();

   // Create a host mirror of the results.
   auto hResults = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dResults);

   // Validate the results.
   for (int i = 0; i < N; i++) {
      ddfun::ddouble a = hostRecords[i].a;
      ddfun::ddouble b = hostRecords[i].b;
      ddfun::ddouble computed = hResults(i);
      ddfun::ddouble expected = hostRecords[i].exp;
      ddfun::ddouble fortranComputed = fortranResults[i];
      int scaleDiff = calculate_scale_difference(computed, expected);
      int scaleDiffFortran = calculate_scale_difference(fortranComputed, expected);
      INFO("Record " << i << " scale differences (C++, Fortran): (" << scaleDiff << ", " << scaleDiffFortran << ");\n"
         << " computed: " << computed << ";\n"
         << " fortran:  " << fortranComputed << ";\n"
         << " expected: " << expected);
      // We require at least REQUIRED_SCALE_PRECISION digits of precision.
      REQUIRE( (scaleDiff >= REQUIRED_SCALE_PRECISION || scaleDiff == 0) );
      REQUIRE( (scaleDiffFortran >= REQUIRED_SCALE_PRECISION || scaleDiffFortran == 0) );
   }
}


TEST_CASE("ddsub on device using mirror views", "[kokkos][ddouble]") {

   // Structure to hold ddsub test data. 
   // Since ddfun::ddouble is just two contiguous doubles, this struct is tightly packed.
   struct DataRecord {
      ddfun::ddouble a;
      ddfun::ddouble b;
      ddfun::ddouble exp;
   };

   // Read test data from file into a vector of DataRecord.
   std::string filename = INPUT_FILES_DIR + "ddsub.bin";
   std::ifstream infile(filename, std::ios::binary);
   INFO("Reading " << filename);
   REQUIRE(infile.good());
   std::vector<DataRecord> hostRecords;
   std::vector<ddfun::ddouble> fortranResults;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      // add data to inputs
      hostRecords.push_back(rec);
      // run Fortran to validate the results
      double a[2] = {rec.a.hi, rec.a.lo};
      double b[2] = {rec.b.hi, rec.b.lo};
      double c[2] = {0.0, 0.0};
      ddsub_fortran(a, b, c);
      fortranResults.push_back(ddfun::ddouble(c[0], c[1]));
   }
   int N = hostRecords.size();
   INFO("Read " << N << " records.");
   REQUIRE(N > 0);

   // Create a host-side Kokkos::View from the vector.
   Kokkos::View<ddfun::ddouble*, Kokkos::HostSpace> hA("hA", N);
   Kokkos::View<ddfun::ddouble*, Kokkos::HostSpace> hB("hB", N);

   for (int i = 0; i < N; i++) {
      hA(i) = hostRecords[i].a;
      hB(i) = hostRecords[i].b;
   }

   // Create a device view by deep copying the host view.
   auto dA = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hA);
   auto dB = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hB);

   // Create a device view to hold the computed results.
   Kokkos::View<ddfun::ddouble*, Kokkos::DefaultExecutionSpace> dResults("dResults", N);

   // Launch a kernel to compute ddsub on each input.
   Kokkos::parallel_for("compute_ddsub", N, KOKKOS_LAMBDA(const int i) {
      dResults(i) = dA(i) - dB(i);
   });
   Kokkos::fence();

   // Create a host mirror of the results.
   auto hResults = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dResults);

   // Validate the results.
   for (int i = 0; i < N; i++) {
      ddfun::ddouble a = hostRecords[i].a;
      ddfun::ddouble b = hostRecords[i].b;
      ddfun::ddouble computed = hResults(i);
      ddfun::ddouble expected = hostRecords[i].exp;
      ddfun::ddouble fortranComputed = fortranResults[i];
      int scaleDiff = calculate_scale_difference(computed, expected);
      int scaleDiffFortran = calculate_scale_difference(fortranComputed, expected);
      INFO("Record " << i << " scale differences (C++, Fortran): (" << scaleDiff << ", " << scaleDiffFortran << ");\n"
         << " computed: " << computed << ";\n"
         << " fortran:  " << fortranComputed << ";\n"
         << " expected: " << expected);
      // We require at least REQUIRED_SCALE_PRECISION digits of precision.
      REQUIRE( (scaleDiff >= REQUIRED_SCALE_PRECISION || scaleDiff == 0) );
      REQUIRE( (scaleDiffFortran >= REQUIRED_SCALE_PRECISION || scaleDiffFortran == 0) );
   }
}


TEST_CASE("ddmul on device using mirror views", "[kokkos][ddouble]") {

   // Structure to hold ddmul test data. 
   // Since ddfun::ddouble is just two contiguous doubles, this struct is tightly packed.
   struct DataRecord {
      ddfun::ddouble a;
      ddfun::ddouble b;
      ddfun::ddouble exp;
   };

   // Read test data from file into a vector of DataRecord.
   std::string filename = INPUT_FILES_DIR + "ddmul.bin";
   std::ifstream infile(filename, std::ios::binary);
   INFO("Reading " << filename);
   REQUIRE(infile.good());
   std::vector<DataRecord> hostRecords;
   std::vector<ddfun::ddouble> fortranResults;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      // add data to inputs
      hostRecords.push_back(rec);
      // run Fortran to validate the results
      double a[2] = {rec.a.hi, rec.a.lo};
      double b[2] = {rec.b.hi, rec.b.lo};
      double c[2] = {0.0, 0.0};
      ddmul_fortran(a, b, c);
      fortranResults.push_back(ddfun::ddouble(c[0], c[1]));
   }
   int N = hostRecords.size();
   INFO("Read " << N << " records.");
   REQUIRE(N > 0);

   // Create a host-side Kokkos::View from the vector.
   Kokkos::View<ddfun::ddouble*, Kokkos::HostSpace> hA("hA", N);
   Kokkos::View<ddfun::ddouble*, Kokkos::HostSpace> hB("hB", N);

   for (int i = 0; i < N; i++) {
      hA(i) = hostRecords[i].a;
      hB(i) = hostRecords[i].b;
   }

   // Create a device view by deep copying the host view.
   auto dA = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hA);
   auto dB = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hB);

   // Create a device view to hold the computed results.
   Kokkos::View<ddfun::ddouble*, Kokkos::DefaultExecutionSpace> dResults("dResults", N);

   // Launch a kernel to compute ddmul on each input.
   Kokkos::parallel_for("compute_ddmul", N, KOKKOS_LAMBDA(const int i) {
      dResults(i) = dA(i) * dB(i);
   });
   Kokkos::fence();

   // Create a host mirror of the results.
   auto hResults = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dResults);

   // Validate the results.
   for (int i = 0; i < N; i++) {
      ddfun::ddouble a = hostRecords[i].a;
      ddfun::ddouble b = hostRecords[i].b;
      ddfun::ddouble computed = hResults(i);
      ddfun::ddouble expected = hostRecords[i].exp;
      ddfun::ddouble fortranComputed = fortranResults[i];
      int scaleDiff = calculate_scale_difference(computed, expected);
      int scaleDiffFortran = calculate_scale_difference(fortranComputed, expected);
      INFO("Record " << i << " scale differences (C++, Fortran): (" << scaleDiff << ", " << scaleDiffFortran << ");\n"
         << " computed: " << computed << ";\n"
         << " fortran:  " << fortranComputed << ";\n"
         << " expected: " << expected);
      // We require at least REQUIRED_SCALE_PRECISION digits of precision.
      REQUIRE( (scaleDiff >= REQUIRED_SCALE_PRECISION || scaleDiff == 0) );
      REQUIRE( (scaleDiffFortran >= REQUIRED_SCALE_PRECISION || scaleDiffFortran == 0) );
   }
}


TEST_CASE("ddmuld on device using mirror views", "[kokkos][ddouble]") {

   // Structure to hold ddmuld test data. 
   // Since ddfun::ddouble is just two contiguous doubles, this struct is tightly packed.
   struct DataRecord {
      ddfun::ddouble a;
      double b;
      ddfun::ddouble exp;
   };

   // Read test data from file into a vector of DataRecord.
   std::string filename = INPUT_FILES_DIR + "ddmuld.bin";
   std::ifstream infile(filename, std::ios::binary);
   INFO("Reading " << filename);
   REQUIRE(infile.good());
   std::vector<DataRecord> hostRecords;
   std::vector<ddfun::ddouble> fortranResults;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      // add data to inputs
      hostRecords.push_back(rec);

      // run Fortran version
      double fA[2] = {rec.a.hi, rec.a.lo};
      double fC[2] = {0.0, 0.0};
      ddmuld_fortran(fA,rec.b, fC);
      fortranResults.push_back(ddfun::ddouble(fC[0], fC[1]));
   }
   int N = hostRecords.size();
   INFO("Read " << N << " records.");
   REQUIRE(N > 0);

   // Create a host-side Kokkos::View from the vector.
   Kokkos::View<ddfun::ddouble*, Kokkos::HostSpace> hA("hA", N);
   Kokkos::View<double*, Kokkos::HostSpace> hB("hB", N);

   for (int i = 0; i < N; i++) {
      hA(i) = hostRecords[i].a;
      hB(i) = hostRecords[i].b;
   }

   // Create a device view by deep copying the host view.
   auto dA = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hA);
   auto dB = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hB);

   // Create a device view to hold the computed results.
   Kokkos::View<ddfun::ddouble*, Kokkos::DefaultExecutionSpace> dResults("dResults", N);

   // Launch a kernel to compute ddmuld on each input.
   Kokkos::parallel_for("compute_ddmuld", N, KOKKOS_LAMBDA(const int i) {
      dResults(i) = ddfun::ddmuld(dA(i),dB(i));
   });
   Kokkos::fence();

   // Create a host mirror of the results.
   auto hResults = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dResults);

   // Validate the results.
   for (int i = 0; i < N; i++) {
      ddfun::ddouble computed = hResults(i);
      ddfun::ddouble expected = hostRecords[i].exp;
      ddfun::ddouble fortranComputed = fortranResults[i];
      int scaleDiff = calculate_scale_difference(computed, expected);
      int scaleDiffFortran = calculate_scale_difference(fortranComputed, expected);
      INFO("Record " << i << " scale differences (C++, Fortran): (" << scaleDiff << ", " << scaleDiffFortran << ");\n"
         << " computed: " << computed << ";\n"
         << " fortran:  " << fortranComputed << ";\n"
         << " expected: " << expected);
      // We require at least REQUIRED_SCALE_PRECISION digits of precision.
      REQUIRE( (scaleDiff >= REQUIRED_SCALE_PRECISION || scaleDiff == 0) );
      REQUIRE( (scaleDiffFortran >= REQUIRED_SCALE_PRECISION || scaleDiffFortran == 0) );
   }
}


TEST_CASE("ddmuldd on device using mirror views", "[kokkos][ddouble]") {

   // Structure to hold ddmuldd test data. 
   // Since ddfun::ddouble is just two contiguous doubles, this struct is tightly packed.
   struct DataRecord {
      double a;
      double b;
      ddfun::ddouble exp;
   };

   // Read test data from file into a vector of DataRecord.
   std::string filename = INPUT_FILES_DIR + "ddmuldd.bin";
   std::ifstream infile(filename, std::ios::binary);
   INFO("Reading " << filename);
   REQUIRE(infile.good());
   std::vector<DataRecord> hostRecords;
   std::vector<ddfun::ddouble> fortranResults;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      // add data to inputs
      hostRecords.push_back(rec);
      // run Fortran to validate the results
      double c[2] = {0.0, 0.0};
      ddmuldd_fortran(rec.a, rec.b, c);
      fortranResults.push_back(ddfun::ddouble(c[0], c[1]));
   }
   int N = hostRecords.size();
   INFO("Read " << N << " records.");
   REQUIRE(N > 0);

   // Create a host-side Kokkos::View from the vector.
   Kokkos::View<ddfun::ddouble*, Kokkos::HostSpace> hA("hA", N);
   Kokkos::View<double*, Kokkos::HostSpace> hB("hB", N);

   for (int i = 0; i < N; i++) {
      hA(i) = hostRecords[i].a;
      hB(i) = hostRecords[i].b;
   }

   // Create a device view by deep copying the host view.
   auto dA = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hA);
   auto dB = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hB);

   // Create a device view to hold the computed results.
   Kokkos::View<ddfun::ddouble*, Kokkos::DefaultExecutionSpace> dResults("dResults", N);

   // Launch a kernel to compute ddmuldd on each input.
   Kokkos::parallel_for("compute_ddmuldd", N, KOKKOS_LAMBDA(const int i) {
      dResults(i) = dA(i) * dB(i);
   });
   Kokkos::fence();

   // Create a host mirror of the results.
   auto hResults = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dResults);

   // Validate the results.
   for (int i = 0; i < N; i++) {
      ddfun::ddouble a = hostRecords[i].a;
      double b = hostRecords[i].b;
      ddfun::ddouble computed = hResults(i);
      ddfun::ddouble expected = hostRecords[i].exp;
      ddfun::ddouble fortranComputed = fortranResults[i];
      int scaleDiff = calculate_scale_difference(computed, expected);
      int scaleDiffFortran = calculate_scale_difference(fortranComputed, expected);
      INFO("Record " << i << " scale differences (C++, Fortran): (" << scaleDiff << ", " << scaleDiffFortran << ");\n"
         << " computed: " << computed << ";\n"
         << " fortran:  " << fortranComputed << ";\n"
         << " expected: " << expected);
      // We require at least REQUIRED_SCALE_PRECISION digits of precision.
      REQUIRE( (scaleDiff >= REQUIRED_SCALE_PRECISION || scaleDiff == 0) );
      REQUIRE( (scaleDiffFortran >= REQUIRED_SCALE_PRECISION || scaleDiffFortran == 0) );
   }
}


TEST_CASE("dddiv on device using mirror views", "[kokkos][ddouble]") {

   // Structure to hold dddiv test data. 
   // Since ddfun::ddouble is just two contiguous doubles, this struct is tightly packed.
   struct DataRecord {
      ddfun::ddouble a;
      ddfun::ddouble b;
      ddfun::ddouble exp;
   };

   // Read test data from file into a vector of DataRecord.
   std::string filename = INPUT_FILES_DIR + "dddiv.bin";
   std::ifstream infile(filename, std::ios::binary);
   INFO("Reading " << filename);
   REQUIRE(infile.good());
   std::vector<DataRecord> hostRecords;
   std::vector<ddfun::ddouble> fortranResults;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      // add data to inputs
      hostRecords.push_back(rec);
      // run Fortran to validate the results
      double a[2] = {rec.a.hi, rec.a.lo};
      double b[2] = {rec.b.hi, rec.b.lo};
      double c[2] = {0.0, 0.0};
      dddiv_fortran(a, b, c);
      fortranResults.push_back(ddfun::ddouble(c[0], c[1]));
   }
   int N = hostRecords.size();
   INFO("Read " << N << " records.");
   REQUIRE(N > 0);

   // Create a host-side Kokkos::View from the vector.
   Kokkos::View<ddfun::ddouble*, Kokkos::HostSpace> hA("hA", N);
   Kokkos::View<ddfun::ddouble*, Kokkos::HostSpace> hB("hB", N);

   for (int i = 0; i < N; i++) {
      hA(i) = hostRecords[i].a;
      hB(i) = hostRecords[i].b;
   }

   // Create a device view by deep copying the host view.
   auto dA = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hA);
   auto dB = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hB);

   // Create a device view to hold the computed results.
   Kokkos::View<ddfun::ddouble*, Kokkos::DefaultExecutionSpace> dResults("dResults", N);

   // Launch a kernel to compute dddiv on each input.
   Kokkos::parallel_for("compute_dddiv", N, KOKKOS_LAMBDA(const int i) {
      dResults(i) = dA(i) / dB(i);
   });
   Kokkos::fence();

   // Create a host mirror of the results.
   auto hResults = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dResults);

   // Validate the results.
   for (int i = 0; i < N; i++) {
      ddfun::ddouble a = hostRecords[i].a;
      ddfun::ddouble b = hostRecords[i].b;
      ddfun::ddouble computed = hResults(i);
      ddfun::ddouble expected = hostRecords[i].exp;
      ddfun::ddouble fortranComputed = fortranResults[i];
      int scaleDiff = calculate_scale_difference(computed, expected);
      int scaleDiffFortran = calculate_scale_difference(fortranComputed, expected);
      INFO("Record " << i << " scale differences (C++, Fortran): (" << scaleDiff << ", " << scaleDiffFortran << ");\n"
         << " computed: " << computed << ";\n"
         << " fortran:  " << fortranComputed << ";\n"
         << " expected: " << expected);
      // We require at least REQUIRED_SCALE_PRECISION digits of precision.
      REQUIRE( (scaleDiff >= REQUIRED_SCALE_PRECISION || scaleDiff == 0) );
      REQUIRE( (scaleDiffFortran >= REQUIRED_SCALE_PRECISION || scaleDiffFortran == 0) );
   }
}


TEST_CASE("dddivd on device using mirror views", "[kokkos][ddouble]") {

   // Structure to hold dddivd test data. 
   // Since ddfun::ddouble is just two contiguous doubles, this struct is tightly packed.
   struct DataRecord {
      ddfun::ddouble a;
      double b;
      ddfun::ddouble exp;
   };

   // Read test data from file into a vector of DataRecord.
   std::string filename = INPUT_FILES_DIR + "dddivd.bin";
   std::ifstream infile(filename, std::ios::binary);
   INFO("Reading " << filename);
   REQUIRE(infile.good());
   std::vector<DataRecord> hostRecords;
   std::vector<ddfun::ddouble> fortranResults;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      // add data to inputs
      hostRecords.push_back(rec);
      // run Fortran to validate the results
      double a[2] = {rec.a.hi, rec.a.lo};
      double c[2] = {0.0, 0.0};
      dddivd_fortran(a,rec.b, c);
      fortranResults.push_back(ddfun::ddouble(c[0], c[1]));
   }
   int N = hostRecords.size();
   INFO("Read " << N << " records.");
   REQUIRE(N > 0);

   // Create a host-side Kokkos::View from the vector.
   Kokkos::View<ddfun::ddouble*, Kokkos::HostSpace> hA("hA", N);
   Kokkos::View<double*, Kokkos::HostSpace> hB("hB", N);

   for (int i = 0; i < N; i++) {
      hA(i) = hostRecords[i].a;
      hB(i) = hostRecords[i].b;
   }

   // Create a device view by deep copying the host view.
   auto dA = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hA);
   auto dB = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hB);

   // Create a device view to hold the computed results.
   Kokkos::View<ddfun::ddouble*, Kokkos::DefaultExecutionSpace> dResults("dResults", N);

   // Launch a kernel to compute dddivd on each input.
   Kokkos::parallel_for("compute_dddivd", N, KOKKOS_LAMBDA(const int i) {
      dResults(i) = ddfun::dddivd(dA(i),dB(i));
   });
   Kokkos::fence();

   // Create a host mirror of the results.
   auto hResults = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dResults);

   // Validate the results.
   for (int i = 0; i < N; i++) {
      ddfun::ddouble a = hostRecords[i].a;
      double b = hostRecords[i].b;
      ddfun::ddouble computed = hResults(i);
      ddfun::ddouble expected = hostRecords[i].exp;
      ddfun::ddouble fortranComputed = fortranResults[i];
      int scaleDiff = calculate_scale_difference(computed, expected);
      int scaleDiffFortran = calculate_scale_difference(fortranComputed, expected);
      INFO("Record " << i << " scale differences (C++, Fortran): (" << scaleDiff << ", " << scaleDiffFortran << ");\n"
         << " computed: " << computed << ";\n"
         << " fortran:  " << fortranComputed << ";\n"
         << " expected: " << expected);
      // We require at least REQUIRED_SCALE_PRECISION digits of precision.
      REQUIRE( (scaleDiff >= REQUIRED_SCALE_PRECISION || scaleDiff == 0) );
      REQUIRE( (scaleDiffFortran >= REQUIRED_SCALE_PRECISION || scaleDiffFortran == 0) );
   }
}


TEST_CASE("ddabs on device using mirror views", "[kokkos][ddouble]") {

   // Structure to hold ddabs test data. 
   // Since ddfun::ddouble is just two contiguous doubles, this struct is tightly packed.
   struct DataRecord {
      ddfun::ddouble a;
      ddfun::ddouble exp;
   };

   // Read test data from file into a vector of DataRecord.
   std::string filename = INPUT_FILES_DIR + "ddabs.bin";
   std::ifstream infile(filename, std::ios::binary);
   INFO("Reading " << filename);
   REQUIRE(infile.good());
   std::vector<DataRecord> hostRecords;
   std::vector<ddfun::ddouble> fortranResults;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      // add data to inputs
      hostRecords.push_back(rec);
      // run Fortran to validate the results
      double a[2] = {rec.a.hi, rec.a.lo};
      double c[2] = {0.0, 0.0};
      ddabs_fortran(a, c);
      fortranResults.push_back(ddfun::ddouble(c[0], c[1]));
   }
   int N = hostRecords.size();
   INFO("Read " << N << " records.");
   REQUIRE(N > 0);

   // Create a host-side Kokkos::View from the vector.
   Kokkos::View<ddfun::ddouble*, Kokkos::HostSpace> hA("hA", N);

   for (int i = 0; i < N; i++) {
      hA(i) = hostRecords[i].a;
   }

   // Create a device view by deep copying the host view.
   auto dA = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hA);

   // Create a device view to hold the computed results.
   Kokkos::View<ddfun::ddouble*, Kokkos::DefaultExecutionSpace> dResults("dResults", N);

   // Launch a kernel to compute ddabs on each input.
   Kokkos::parallel_for("compute_ddabs", N, KOKKOS_LAMBDA(const int i) {
      dResults(i) = ddfun::ddabs(dA(i));
   });
   Kokkos::fence();

   // Create a host mirror of the results.
   auto hResults = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dResults);

   // Validate the results.
   for (int i = 0; i < N; i++) {
      ddfun::ddouble a = hostRecords[i].a;
      ddfun::ddouble computed = hResults(i);
      ddfun::ddouble expected = hostRecords[i].exp;
      ddfun::ddouble fortranComputed = fortranResults[i];
      int scaleDiff = calculate_scale_difference(computed, expected);
      int scaleDiffFortran = calculate_scale_difference(fortranComputed, expected);
      INFO("Record " << i << " scale differences (C++, Fortran): (" << scaleDiff << ", " << scaleDiffFortran << ");\n"
         << " computed: " << computed << ";\n"
         << " fortran:  " << fortranComputed << ";\n"
         << " expected: " << expected);
      // We require at least REQUIRED_SCALE_PRECISION digits of precision.
      REQUIRE( (scaleDiff >= REQUIRED_SCALE_PRECISION || scaleDiff == 0) );
      REQUIRE( (scaleDiffFortran >= REQUIRED_SCALE_PRECISION || scaleDiffFortran == 0) );
   }
}


TEST_CASE("ddnint on device using mirror views", "[kokkos][ddouble]") {

   // Structure to hold ddnint test data. 
   // Since ddfun::ddouble is just two contiguous doubles, this struct is tightly packed.
   struct DataRecord {
      ddfun::ddouble a;
      ddfun::ddouble exp;
   };

   // Read test data from file into a vector of DataRecord.
   std::string filename = INPUT_FILES_DIR + "ddnint.bin";
   std::ifstream infile(filename, std::ios::binary);
   INFO("Reading " << filename);
   REQUIRE(infile.good());
   std::vector<DataRecord> hostRecords;
   std::vector<ddfun::ddouble> fortranResults;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      // add data to inputs
      hostRecords.push_back(rec);
      // run Fortran to validate the results
      double a[2] = {rec.a.hi, rec.a.lo};
      double c[2] = {0.0, 0.0};
      ddnint_fortran(a, c);
      fortranResults.push_back(ddfun::ddouble(c[0], c[1]));
   }
   int N = hostRecords.size();
   INFO("Read " << N << " records.");
   REQUIRE(N > 0);

   // Create a host-side Kokkos::View from the vector.
   Kokkos::View<ddfun::ddouble*, Kokkos::HostSpace> hA("hA", N);

   for (int i = 0; i < N; i++) {
      hA(i) = hostRecords[i].a;
   }

   // Create a device view by deep copying the host view.
   auto dA = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hA);

   // Create a device view to hold the computed results.
   Kokkos::View<ddfun::ddouble*, Kokkos::DefaultExecutionSpace> dResults("dResults", N);

   // Launch a kernel to compute ddnint on each input.
   Kokkos::parallel_for("compute_ddnint", N, KOKKOS_LAMBDA(const int i) {
      dResults(i) = ddfun::ddnint(dA(i));
   });
   Kokkos::fence();

   // Create a host mirror of the results.
   auto hResults = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dResults);

   // Validate the results.
   for (int i = 0; i < N; i++) {
      ddfun::ddouble a = hostRecords[i].a;
      ddfun::ddouble computed = hResults(i);
      ddfun::ddouble expected = hostRecords[i].exp;
      ddfun::ddouble fortranComputed = fortranResults[i];
      int scaleDiff = calculate_scale_difference(computed, expected);
      int scaleDiffFortran = calculate_scale_difference(fortranComputed, expected);

      INFO("i=" << i << " a=" << a << " computed=" << computed << " expected=" << expected 
         << " fortran=" << fortranComputed 
         << " scaleDiff=" << scaleDiff 
         << " scaleDiffFortran=" << scaleDiffFortran 
         << " expected: " << expected);
      // We require at least REQUIRED_SCALE_PRECISION digits of precision.
      REQUIRE( (scaleDiff >= REQUIRED_SCALE_PRECISION || scaleDiff == 0) );
      REQUIRE( (scaleDiffFortran >= REQUIRED_SCALE_PRECISION || scaleDiffFortran == 0) );
   }
}


TEST_CASE("ddsqrt on device using mirror views", "[kokkos][ddouble]") {

   // Structure to hold ddsqrt test data. 
   // Since ddfun::ddouble is just two contiguous doubles, this struct is tightly packed.
   struct DataRecord {
      ddfun::ddouble a;
      ddfun::ddouble exp;
   };

   // Read test data from file into a vector of DataRecord.
   std::string filename = INPUT_FILES_DIR + "ddsqrt.bin";
   std::ifstream infile(filename, std::ios::binary);
   INFO("Reading " << filename);
   REQUIRE(infile.good());
   std::vector<DataRecord> hostRecords;
   std::vector<ddfun::ddouble> fortranResults;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      // add data to inputs
      hostRecords.push_back(rec);
      // run Fortran to validate the results
      double a[2] = {rec.a.hi, rec.a.lo};
      double c[2] = {0.0, 0.0};
      ddsqrt_fortran(a, c);
      fortranResults.push_back(ddfun::ddouble(c[0], c[1]));
   }
   int N = hostRecords.size();
   INFO("Read " << N << " records.");
   REQUIRE(N > 0);

   // Create a host-side Kokkos::View from the vector.
   Kokkos::View<ddfun::ddouble*, Kokkos::HostSpace> hA("hA", N);

   for (int i = 0; i < N; i++) {
      hA(i) = hostRecords[i].a;
   }

   // Create a device view by deep copying the host view.
   auto dA = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hA);

   // Create a device view to hold the computed results.
   Kokkos::View<ddfun::ddouble*, Kokkos::DefaultExecutionSpace> dResults("dResults", N);

   // Launch a kernel to compute ddsqrt on each input.
   Kokkos::parallel_for("compute_ddsqrt", N, KOKKOS_LAMBDA(const int i) {
      dResults(i) = ddfun::ddsqrt(dA(i));
   });
   Kokkos::fence();

   // Create a host mirror of the results.
   auto hResults = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dResults);

   // Validate the results.
   for (int i = 0; i < N; i++) {
      ddfun::ddouble a = hostRecords[i].a;
      ddfun::ddouble computed = hResults(i);
      ddfun::ddouble expected = hostRecords[i].exp;
      ddfun::ddouble fortranComputed = fortranResults[i];
      int scaleDiff = calculate_scale_difference(computed, expected);
      int scaleDiffFortran = calculate_scale_difference(fortranComputed, expected);
      INFO("Record " << i << " scale differences (C++, Fortran): (" << scaleDiff << ", " << scaleDiffFortran << ");\n"
         << " computed: " << computed << ";\n"
         << " fortran:  " << fortranComputed << ";\n"
         << " expected: " << expected);
      // We require at least REQUIRED_SCALE_PRECISION digits of precision.
      REQUIRE( (scaleDiff >= REQUIRED_SCALE_PRECISION || scaleDiff == 0) );
      REQUIRE( (scaleDiffFortran >= REQUIRED_SCALE_PRECISION || scaleDiffFortran == 0) );
   }
}


TEST_CASE("ddexp on device using mirror views", "[kokkos][ddouble]") {

   // Structure to hold ddexp test data. 
   // Since ddfun::ddouble is just two contiguous doubles, this struct is tightly packed.
   struct DataRecord {
      ddfun::ddouble a;
      ddfun::ddouble exp;
   };

   // Read test data from file into a vector of DataRecord.
   std::string filename = INPUT_FILES_DIR + "ddexp.bin";
   std::ifstream infile(filename, std::ios::binary);
   INFO("Reading " << filename);
   REQUIRE(infile.good());
   std::vector<DataRecord> hostRecords;
   std::vector<ddfun::ddouble> fortranResults;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      // add data to inputs
      hostRecords.push_back(rec);
      // run Fortran to validate the results
      double a[2] = {rec.a.hi, rec.a.lo};
      double c[2] = {0.0, 0.0};
      ddexp_fortran(a, c);
      fortranResults.push_back(ddfun::ddouble(c[0], c[1]));
   }
   int N = hostRecords.size();
   INFO("Read " << N << " records.");
   REQUIRE(N > 0);

   // Create a host-side Kokkos::View from the vector.
   Kokkos::View<ddfun::ddouble*, Kokkos::HostSpace> hA("hA", N);

   for (int i = 0; i < N; i++) {
      hA(i) = hostRecords[i].a;
   }

   // Create a device view by deep copying the host view.
   auto dA = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hA);

   // Create a device view to hold the computed results.
   Kokkos::View<ddfun::ddouble*, Kokkos::DefaultExecutionSpace> dResults("dResults", N);

   // Launch a kernel to compute ddexp on each input.
   Kokkos::parallel_for("compute_ddexp", N, KOKKOS_LAMBDA(const int i) {
      dResults(i) = ddfun::ddexp(dA(i));
   });
   Kokkos::fence();

   // Create a host mirror of the results.
   auto hResults = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dResults);

   // Validate the results.
   for (int i = 0; i < N; i++) {
      ddfun::ddouble a = hostRecords[i].a;
      ddfun::ddouble computed = hResults(i);
      ddfun::ddouble expected = hostRecords[i].exp;
      ddfun::ddouble fortranComputed = fortranResults[i];
      int scaleDiff = calculate_scale_difference(computed, expected);
      int scaleDiffFortran = calculate_scale_difference(fortranComputed, expected);
      INFO("Record " << i << " scale differences (C++, Fortran): (" << scaleDiff << ", " << scaleDiffFortran << ");\n"
         << " computed: " << computed << ";\n"
         << " fortran:  " << fortranComputed << ";\n"
         << " expected: " << expected);
      // We require at least REQUIRED_SCALE_PRECISION digits of precision.
      REQUIRE( (scaleDiff >= REQUIRED_SCALE_PRECISION || scaleDiff == 0) );
      REQUIRE( (scaleDiffFortran >= REQUIRED_SCALE_PRECISION || scaleDiffFortran == 0) );
   }
}


TEST_CASE("ddlog on device using mirror views", "[kokkos][ddouble][ddlog]") {

   // Structure to hold ddlog test data. 
   // Since ddfun::ddouble is just two contiguous doubles, this struct is tightly packed.
   struct DataRecord {
      ddfun::ddouble a;
      ddfun::ddouble exp;
   };

   // Read test data from file into a vector of DataRecord.
   std::string filename = INPUT_FILES_DIR + "ddlog.bin";
   std::ifstream infile(filename, std::ios::binary);
   INFO("Reading " << filename);
   REQUIRE(infile.good());
   std::vector<DataRecord> hostRecords;
   std::vector<ddfun::ddouble> fortranResults;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      // add data to inputs
      hostRecords.push_back(rec);
      // run Fortran to validate the results
      double a[2] = {rec.a.hi, rec.a.lo};
      double c[2] = {0.0, 0.0};
      ddlog_fortran(a, c);
      INFO("input: " << rec.a << "   HEX: " << std::hex << *(uint64_t*)&rec.a.hi << " " << *(uint64_t*)&rec.a.lo << "   output: " << ddfun::ddouble(c[0], c[1]));
      fortranResults.push_back(ddfun::ddouble(c[0], c[1]));
   }
   int N = hostRecords.size();
   INFO("Read " << N << " records.");
   REQUIRE(N > 0);

   // Create a host-side Kokkos::View from the vector.
   Kokkos::View<ddfun::ddouble*, Kokkos::HostSpace> hA("hA", N);

   for (int i = 0; i < N; i++) {
      hA(i) = hostRecords[i].a;
   }

   // Create a device view by deep copying the host view.
   auto dA = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hA);

   // Create a device view to hold the computed results.
   Kokkos::View<ddfun::ddouble*, Kokkos::DefaultExecutionSpace> dResults("dResults", N);

   // Launch a kernel to compute ddlog on each input.
   Kokkos::parallel_for("compute_ddlog", N, KOKKOS_LAMBDA(const int i) {
      dResults(i) = ddfun::ddlog(dA(i));
   });
   Kokkos::fence();

   // Create a host mirror of the results.
   auto hResults = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dResults);

   // Validate the results.
   for (int i = 0; i < N; i++) {
      ddfun::ddouble a = hostRecords[i].a;
      ddfun::ddouble computed = hResults(i);
      ddfun::ddouble expected = hostRecords[i].exp;
      ddfun::ddouble fortranComputed = fortranResults[i];
      int scaleDiff = calculate_scale_difference(computed, expected);
      int scaleDiffFortran = calculate_scale_difference(fortranComputed, expected);
      INFO("Record " << i << " scale differences (C++, Fortran): (" << scaleDiff << ", " << scaleDiffFortran << ");\n"
         << " input:    " << a << "\n"
         << "   HEX:  0x" << std::hex << std::setfill('0') << std::setw(16) << *reinterpret_cast<const uint64_t*>(&a.hi) << " 0x" << *reinterpret_cast<const uint64_t*>(&a.lo) << "\n"
         << " computed: " << computed << "\n"
         << "   HEX:  0x" << std::hex << std::setfill('0') << std::setw(16) << *reinterpret_cast<const uint64_t*>(&computed.hi) << " 0x" << *reinterpret_cast<const uint64_t*>(&computed.lo) << "\n"
         << " fortran:  " << fortranComputed << "\n"
         << "   HEX:  0x" << std::hex << std::setfill('0') << std::setw(16) << *reinterpret_cast<const uint64_t*>(&fortranComputed.hi) << " 0x" << *reinterpret_cast<const uint64_t*>(&fortranComputed.lo) << "\n"
         << " expected: " << expected << "\n"
         << "   HEX:  0x" << std::hex << std::setfill('0') << std::setw(16) << *reinterpret_cast<const uint64_t*>(&expected.hi) << " 0x" << *reinterpret_cast<const uint64_t*>(&expected.lo));
      // We require at least REQUIRED_SCALE_PRECISION digits of precision.
      REQUIRE( (scaleDiff >= REQUIRED_SCALE_PRECISION || scaleDiff == 0) );
      REQUIRE( (scaleDiffFortran >= REQUIRED_SCALE_PRECISION || scaleDiffFortran == 0) );
   }
}


TEST_CASE("ddacosh on device using mirror views", "[kokkos][ddouble]") {

   // Structure to hold ddacosh test data. 
   // Since ddfun::ddouble is just two contiguous doubles, this struct is tightly packed.
   struct DataRecord {
      ddfun::ddouble a;
      ddfun::ddouble exp;
   };

   // Read test data from file into a vector of DataRecord.
   std::string filename = INPUT_FILES_DIR + "ddacosh.bin";
   std::ifstream infile(filename, std::ios::binary);
   INFO("Reading " << filename);
   REQUIRE(infile.good());
   std::vector<DataRecord> hostRecords;
   std::vector<ddfun::ddouble> fortranResults;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      // add data to inputs
      hostRecords.push_back(rec);
      // run Fortran to validate the results
      double a[2] = {rec.a.hi, rec.a.lo};
      double c[2] = {0.0, 0.0};
      ddacosh_fortran(a, c);
      fortranResults.push_back(ddfun::ddouble(c[0], c[1]));
   }
   int N = hostRecords.size();
   INFO("Read " << N << " records.");
   REQUIRE(N > 0);

   // Create a host-side Kokkos::View from the vector.
   Kokkos::View<ddfun::ddouble*, Kokkos::HostSpace> hA("hA", N);

   for (int i = 0; i < N; i++) {
      hA(i) = hostRecords[i].a;
   }

   // Create a device view by deep copying the host view.
   auto dA = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hA);

   // Create a device view to hold the computed results.
   Kokkos::View<ddfun::ddouble*, Kokkos::DefaultExecutionSpace> dResults("dResults", N);

   // Launch a kernel to compute ddacosh on each input.
   Kokkos::parallel_for("compute_ddacosh", N, KOKKOS_LAMBDA(const int i) {
      dResults(i) = ddfun::ddacosh(dA(i));
   });
   Kokkos::fence();

   // Create a host mirror of the results.
   auto hResults = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dResults);

   // Validate the results.
   for (int i = 0; i < N; i++) {
      ddfun::ddouble a = hostRecords[i].a;
      ddfun::ddouble computed = hResults(i);
      ddfun::ddouble expected = hostRecords[i].exp;
      ddfun::ddouble fortranComputed = fortranResults[i];
      int scaleDiff = calculate_scale_difference(computed, expected);
      int scaleDiffFortran = calculate_scale_difference(fortranComputed, expected);
      INFO("Record " << i << " scale differences (C++, Fortran): (" << scaleDiff << ", " << scaleDiffFortran << ");\n"
         << " computed: " << computed << ";\n"
         << " fortran:  " << fortranComputed << ";\n"
         << " expected: " << expected);
      // We require at least REQUIRED_SCALE_PRECISION digits of precision.
      REQUIRE( (scaleDiff >= REQUIRED_SCALE_PRECISION || scaleDiff == 0) );
      REQUIRE( (scaleDiffFortran >= REQUIRED_SCALE_PRECISION || scaleDiffFortran == 0) );
   }
}


TEST_CASE("ddatanh on device using mirror views", "[kokkos][ddouble]") {

   // Structure to hold ddatanh test data. 
   // Since ddfun::ddouble is just two contiguous doubles, this struct is tightly packed.
   struct DataRecord {
      ddfun::ddouble a;
      ddfun::ddouble exp;
   };

   // Read test data from file into a vector of DataRecord.
   std::string filename = INPUT_FILES_DIR + "ddatanh.bin";
   std::ifstream infile(filename, std::ios::binary);
   INFO("Reading " << filename);
   REQUIRE(infile.good());
   std::vector<DataRecord> hostRecords;
   std::vector<ddfun::ddouble> fortranResults;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      // add data to inputs
      hostRecords.push_back(rec);
      // run Fortran to validate the results
      double a[2] = {rec.a.hi, rec.a.lo};
      double c[2] = {0.0, 0.0};
      ddatanh_fortran(a, c);
      fortranResults.push_back(ddfun::ddouble(c[0], c[1]));
   }
   int N = hostRecords.size();
   INFO("Read " << N << " records.");
   REQUIRE(N > 0);

   // Create a host-side Kokkos::View from the vector.
   Kokkos::View<ddfun::ddouble*, Kokkos::HostSpace> hA("hA", N);

   for (int i = 0; i < N; i++) {
      hA(i) = hostRecords[i].a;
   }

   // Create a device view by deep copying the host view.
   auto dA = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hA);

   // Create a device view to hold the computed results.
   Kokkos::View<ddfun::ddouble*, Kokkos::DefaultExecutionSpace> dResults("dResults", N);

   // Launch a kernel to compute ddatanh on each input.
   Kokkos::parallel_for("compute_ddatanh", N, KOKKOS_LAMBDA(const int i) {
      dResults(i) = ddfun::ddatanh(dA(i));
   });
   Kokkos::fence();

   // Create a host mirror of the results.
   auto hResults = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dResults);

   // Validate the results.
   for (int i = 0; i < N; i++) {
      ddfun::ddouble a = hostRecords[i].a;
      ddfun::ddouble computed = hResults(i);
      ddfun::ddouble expected = hostRecords[i].exp;
      ddfun::ddouble fortranComputed = fortranResults[i];
      int scaleDiff = calculate_scale_difference(computed, expected);
      int scaleDiffFortran = calculate_scale_difference(fortranComputed, expected);
      INFO("Record " << i << " scale differences (C++, Fortran): (" << scaleDiff << ", " << scaleDiffFortran << ");\n"
         << " computed: " << computed << ";\n"
         << " fortran:  " << fortranComputed << ";\n"
         << " expected: " << expected);
      // We require at least REQUIRED_SCALE_PRECISION digits of precision.
      REQUIRE( (scaleDiff >= REQUIRED_SCALE_PRECISION || scaleDiff == 0) );
      REQUIRE( (scaleDiffFortran >= REQUIRED_SCALE_PRECISION || scaleDiffFortran == 0) );
   }
}


TEST_CASE("ddasinh on device using mirror views", "[kokkos][ddouble]") {

   // Structure to hold ddasinh test data. 
   // Since ddfun::ddouble is just two contiguous doubles, this struct is tightly packed.
   struct DataRecord {
      ddfun::ddouble a;
      ddfun::ddouble exp;
   };

   // Read test data from file into a vector of DataRecord.
   std::string filename = INPUT_FILES_DIR + "ddasinh.bin";
   std::ifstream infile(filename, std::ios::binary);
   INFO("Reading " << filename);
   REQUIRE(infile.good());
   std::vector<DataRecord> hostRecords;
   std::vector<ddfun::ddouble> fortranResults;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      // add data to inputs
      hostRecords.push_back(rec);
      // run Fortran to validate the results
      double a[2] = {rec.a.hi, rec.a.lo};
      double c[2] = {0.0, 0.0};
      ddasinh_fortran(a, c);
      fortranResults.push_back(ddfun::ddouble(c[0], c[1]));
   }
   int N = hostRecords.size();
   INFO("Read " << N << " records.");
   REQUIRE(N > 0);

   // Create a host-side Kokkos::View from the vector.
   Kokkos::View<ddfun::ddouble*, Kokkos::HostSpace> hA("hA", N);

   for (int i = 0; i < N; i++) {
      hA(i) = hostRecords[i].a;
   }

   // Create a device view by deep copying the host view.
   auto dA = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hA);

   // Create a device view to hold the computed results.
   Kokkos::View<ddfun::ddouble*, Kokkos::DefaultExecutionSpace> dResults("dResults", N);

   // Launch a kernel to compute ddasinh on each input.
   Kokkos::parallel_for("compute_ddasinh", N, KOKKOS_LAMBDA(const int i) {
      dResults(i) = ddfun::ddasinh(dA(i));
   });
   Kokkos::fence();

   // Create a host mirror of the results.
   auto hResults = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dResults);

   // Validate the results.
   for (int i = 0; i < N; i++) {
      ddfun::ddouble a = hostRecords[i].a;
      ddfun::ddouble computed = hResults(i);
      ddfun::ddouble expected = hostRecords[i].exp;
      ddfun::ddouble fortranComputed = fortranResults[i];
      int scaleDiff = calculate_scale_difference(computed, expected);
      int scaleDiffFortran = calculate_scale_difference(fortranComputed, expected);
      INFO("Record " << i << " scale differences (C++, Fortran): (" << scaleDiff << ", " << scaleDiffFortran << ");\n"
         << " computed: " << computed << ";\n"
         << " fortran:  " << fortranComputed << ";\n"
         << " expected: " << expected);
      // We require at least REQUIRED_SCALE_PRECISION digits of precision.
      REQUIRE( (scaleDiff >= REQUIRED_SCALE_PRECISION || scaleDiff == 0) );
      REQUIRE( (scaleDiffFortran >= REQUIRED_SCALE_PRECISION || scaleDiffFortran == 0) );
   }
}


TEST_CASE("ddpower on device using mirror views", "[kokkos][ddouble]") {

   // Structure to hold ddpower test data. 
   // Since ddfun::ddouble is just two contiguous doubles, this struct is tightly packed.
   struct DataRecord {
      ddfun::ddouble a;
      ddfun::ddouble b;
      ddfun::ddouble exp;
   };

   // Read test data from file into a vector of DataRecord.
   std::string filename = INPUT_FILES_DIR + "ddpower.bin";
   std::ifstream infile(filename, std::ios::binary);
   INFO("Reading " << filename);
   REQUIRE(infile.good());
   std::vector<DataRecord> hostRecords;
   std::vector<ddfun::ddouble> fortranResults;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      // add data to inputs
      hostRecords.push_back(rec);
      // run Fortran to validate the results
      double a[2] = {rec.a.hi, rec.a.lo};
      double b[2] = {rec.b.hi, rec.b.lo};
      double c[2] = {0.0, 0.0};
      ddpower_fortran(a, b, c);
      fortranResults.push_back(ddfun::ddouble(c[0], c[1]));
   }
   int N = hostRecords.size();
   INFO("Read " << N << " records.");
   REQUIRE(N > 0);

   // Create a host-side Kokkos::View from the vector.
   Kokkos::View<ddfun::ddouble*, Kokkos::HostSpace> hA("hA", N);
   Kokkos::View<ddfun::ddouble*, Kokkos::HostSpace> hB("hB", N);

   for (int i = 0; i < N; i++) {
      hA(i) = hostRecords[i].a;
      hB(i) = hostRecords[i].b;
   }

   // Create a device view by deep copying the host view.
   auto dA = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hA);
   auto dB = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hB);

   // Create a device view to hold the computed results.
   Kokkos::View<ddfun::ddouble*, Kokkos::DefaultExecutionSpace> dResults("dResults", N);

   // Launch a kernel to compute ddpower on each input.
   Kokkos::parallel_for("compute_ddpower", N, KOKKOS_LAMBDA(const int i) {
      dResults(i) = ddfun::ddpower(dA(i), dB(i));
   });
   Kokkos::fence();

   // Create a host mirror of the results.
   auto hResults = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dResults);

   // Validate the results.
   for (int i = 0; i < N; i++) {
      ddfun::ddouble a = hostRecords[i].a;
      ddfun::ddouble b = hostRecords[i].b;
      ddfun::ddouble computed = hResults(i);
      ddfun::ddouble expected = hostRecords[i].exp;
      ddfun::ddouble fortranComputed = fortranResults[i];
      int scaleDiff = calculate_scale_difference(computed, expected);
      int scaleDiffFortran = calculate_scale_difference(fortranComputed, expected);
      INFO("Record " << i << " scale differences (C++, Fortran): (" << scaleDiff << ", " << scaleDiffFortran << ");\n"
         << " computed: " << computed << ";\n"
         << " fortran:  " << fortranComputed << ";\n"
         << " expected: " << expected);
      // We require at least REQUIRED_SCALE_PRECISION digits of precision.
      REQUIRE( (scaleDiff >= REQUIRED_SCALE_PRECISION || scaleDiff == 0) );
      REQUIRE( (scaleDiffFortran >= REQUIRED_SCALE_PRECISION || scaleDiffFortran == 0) );
   }
}


TEST_CASE("ddnpwr on device using mirror views", "[kokkos][ddouble]") {

   // Structure to hold ddnpwr test data. 
   // Since ddfun::ddouble is just two contiguous doubles, this struct is tightly packed.
   struct DataRecord {
      ddfun::ddouble a;
      int b;
      int ph;
      ddfun::ddouble exp;
   };

   // Read test data from file into a vector of DataRecord.
   std::string filename = INPUT_FILES_DIR + "ddnpwr.bin";
   std::ifstream infile(filename, std::ios::binary);
   INFO("Reading " << filename);
   REQUIRE(infile.good());
   std::vector<DataRecord> hostRecords;
   std::vector<ddfun::ddouble> fortranResults;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      // add data to inputs
      hostRecords.push_back(rec);
      // run Fortran to validate the results
      double a[2] = {rec.a.hi, rec.a.lo};
      int b = rec.b;
      double c[2] = {0.0, 0.0};
      ddnpwr_fortran(a, b, c);
      fortranResults.push_back(ddfun::ddouble(c[0], c[1]));
   }
   int N = hostRecords.size();
   INFO("Read " << N << " records.");
   REQUIRE(N > 0);

   // Create a host-side Kokkos::View from the vector.
   Kokkos::View<ddfun::ddouble*, Kokkos::HostSpace> hA("hA", N);
   Kokkos::View<int*, Kokkos::HostSpace> hB("hB", N);

   for (int i = 0; i < N; i++) {
      hA(i) = hostRecords[i].a;
      hB(i) = hostRecords[i].b;
   }

   // Create a device view by deep copying the host view.
   auto dA = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hA);
   auto dB = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hB);

   // Create a device view to hold the computed results.
   Kokkos::View<ddfun::ddouble*, Kokkos::DefaultExecutionSpace> dResults("dResults", N);

   // Launch a kernel to compute ddnpwr on each input.
   Kokkos::parallel_for("compute_ddnpwr", N, KOKKOS_LAMBDA(const int i) {
      dResults(i) = ddfun::ddnpwr(dA(i), dB(i));
   });
   Kokkos::fence();

   // Create a host mirror of the results.
   auto hResults = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dResults);

   // Validate the results.
   for (int i = 0; i < N; i++) {
      ddfun::ddouble a = hostRecords[i].a;
      int b = hostRecords[i].b;
      ddfun::ddouble computed = hResults(i);
      ddfun::ddouble expected = hostRecords[i].exp;
      ddfun::ddouble fortranComputed = fortranResults[i];
      int scaleDiff = calculate_scale_difference(computed, expected);
      int scaleDiffFortran = calculate_scale_difference(fortranComputed, expected);
      INFO("Record " << i << " scale differences (C++, Fortran): (" << scaleDiff << ", " << scaleDiffFortran << ");\n"
         << " computed: " << computed << ";\n"
         << " fortran:  " << fortranComputed << ";\n"
         << " expected: " << expected);
      // We require at least REQUIRED_SCALE_PRECISION digits of precision.
      REQUIRE( (scaleDiff >= REQUIRED_SCALE_PRECISION || scaleDiff == 0) );
      REQUIRE( (scaleDiffFortran >= REQUIRED_SCALE_PRECISION || scaleDiffFortran == 0) );
   }
}



//////////////////////////////////////////////////////////////////////////
// Example for a single complex operation: ddcadd (complex addition)
//////////////////////////////////////////////////////////////////////////

TEST_CASE("ddcadd on device using mirror views", "[kokkos][ddcomplex]") {

   // Structure to hold ddcadd test data. 
   // Since ddfun::ddcomplex is just two contiguous doubles, this struct is tightly packed.
   struct DataRecord {
      ddfun::ddcomplex a;
      ddfun::ddcomplex b;
      ddfun::ddcomplex exp;
   };

   // Read test data from file into a vector of DataRecord.
   std::string filename = INPUT_FILES_DIR + "ddcadd.bin";
   std::ifstream infile(filename, std::ios::binary);
   INFO("Reading " << filename);
   REQUIRE(infile.good());
   std::vector<DataRecord> hostRecords;
   std::vector<ddfun::ddcomplex> fortranResults;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      // add data to inputs
      hostRecords.push_back(rec);
      // run Fortran to validate the results
      double a[4] = {rec.a.real.hi, rec.a.real.lo, rec.a.imag.hi, rec.a.imag.lo};
      double b[4] = {rec.b.real.hi, rec.b.real.lo, rec.b.imag.hi, rec.b.imag.lo};
      double c[4] = {0.0, 0.0, 0.0, 0.0};
      ddcadd_fortran(a, b, c);
      fortranResults.push_back(ddfun::ddcomplex(ddfun::ddouble(c[0], c[1]), ddfun::ddouble(c[2], c[3])));
   }
   int N = hostRecords.size();
   INFO("Read " << N << " records.");
   REQUIRE(N > 0);

   // Create a host-side Kokkos::View from the vector.
   Kokkos::View<ddfun::ddcomplex*, Kokkos::HostSpace> hA("hA", N);
   Kokkos::View<ddfun::ddcomplex*, Kokkos::HostSpace> hB("hB", N);

   for (int i = 0; i < N; i++) {
      hA(i) = hostRecords[i].a;
      hB(i) = hostRecords[i].b;
   }

   // Create a device view by deep copying the host view.
   auto dA = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hA);
   auto dB = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hB);

   // Create a device view to hold the computed results.
   Kokkos::View<ddfun::ddcomplex*, Kokkos::DefaultExecutionSpace> dResults("dResults", N);

   // Launch a kernel to compute ddcadd on each input.
   Kokkos::parallel_for("compute_ddcadd", N, KOKKOS_LAMBDA(const int i) {
      dResults(i) = dA(i) + dB(i);
   });
   Kokkos::fence();

   // Create a host mirror of the results.
   auto hResults = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dResults);

   // Validate the results.
   for (int i = 0; i < N; i++) {
      ddfun::ddcomplex a = hostRecords[i].a;
      ddfun::ddcomplex b = hostRecords[i].b;
      ddfun::ddcomplex computed = hResults(i);
      ddfun::ddcomplex expected = hostRecords[i].exp;
      ddfun::ddcomplex fortranComputed = fortranResults[i];
      int scaleDiffReal = calculate_scale_difference(computed.real, expected.real);
      int scaleDiffImag = calculate_scale_difference(computed.imag, expected.imag);
      int scaleDiffFortranReal = calculate_scale_difference(fortranComputed.real, expected.real);
      int scaleDiffFortranImag = calculate_scale_difference(fortranComputed.imag, expected.imag);
      INFO("Record " << i << " scale differences (C++, Fortran): (real: (" << scaleDiffReal << ", " << scaleDiffFortranReal << "), imag: (" << scaleDiffImag << ", " << scaleDiffFortranImag << "));\n"
         << " computed: " << computed << ";\n"
         << " fortran:  " << fortranComputed << ";\n"
         << " expected: " << expected);
      // We require at least REQUIRED_SCALE_PRECISION digits of precision.
      REQUIRE( (scaleDiffReal >= REQUIRED_SCALE_PRECISION || scaleDiffReal == 0) );
      REQUIRE( (scaleDiffImag >= REQUIRED_SCALE_PRECISION || scaleDiffImag == 0) );
      REQUIRE( (scaleDiffFortranReal >= REQUIRED_SCALE_PRECISION || scaleDiffFortranReal == 0) );
      REQUIRE( (scaleDiffFortranImag >= REQUIRED_SCALE_PRECISION || scaleDiffFortranImag == 0) );
   }
}


TEST_CASE("ddcsub on device", "[kokkos][ddcomplex]") {

   // Structure for ddcsub test data: two ddcomplex inputs and one expected output.
   struct DataRecord {
      // Each ddcomplex has two ddfun::ddouble members.
      // For binary layout, assume ddcomplex is stored as: real, imag.
      ddfun::ddcomplex a;
      ddfun::ddcomplex b;
      ddfun::ddcomplex exp;
   };

   std::ifstream infile(INPUT_FILES_DIR + "ddcsub.bin", std::ios::binary);
   INFO("Reading " << INPUT_FILES_DIR + "ddcsub.bin");
   REQUIRE(infile.good());
   std::vector<DataRecord> hostData;
   std::vector<ddfun::ddcomplex> fortranResults;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      // add data to inputs
      hostData.push_back(rec);
      // run Fortran to validate the results
      double a[4] = {rec.a.real.hi, rec.a.real.lo, rec.a.imag.hi, rec.a.imag.lo};
      double b[4] = {rec.b.real.hi, rec.b.real.lo, rec.b.imag.hi, rec.b.imag.lo};
      double c[4] = {0.0, 0.0, 0.0, 0.0};
      ddcsub_fortran(a, b, c);
      fortranResults.push_back(ddfun::ddcomplex(ddfun::ddouble(c[0], c[1]), ddfun::ddouble(c[2], c[3])));
   }
   int N = hostData.size();
   INFO("Read " << N << " records.");
   REQUIRE(N > 0);

   Kokkos::View<ddfun::ddcomplex*, Kokkos::HostSpace> hA("hA", N);
   Kokkos::View<ddfun::ddcomplex*, Kokkos::HostSpace> hB("hB", N);
   for (int i = 0; i < N; i++) {
      hA(i) = hostData[i].a;
      hB(i) = hostData[i].b;
   }

   auto dA = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hA);
   auto dB = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hB);

   Kokkos::View<ddfun::ddcomplex*, Kokkos::DefaultExecutionSpace> dResults("dResults", N);
   Kokkos::parallel_for("compute_ddcsub", N, KOKKOS_LAMBDA(const int i) {
      // Build complex numbers from the record.
      dResults(i) = dA(i) - dB(i);
   });
   Kokkos::fence();

   auto hResults = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dResults);
   for (int i = 0; i < N; i++) {
      ddfun::ddcomplex a = hostData[i].a;
      ddfun::ddcomplex b = hostData[i].b;
      ddfun::ddcomplex computed = hResults(i);
      ddfun::ddcomplex expected = hostData[i].exp;
      ddfun::ddcomplex fortranComputed = fortranResults[i];
      int scaleDiffReal = calculate_scale_difference(computed.real, expected.real);
      int scaleDiffImag = calculate_scale_difference(computed.imag, expected.imag);
      int scaleDiffFortranReal = calculate_scale_difference(fortranComputed.real, expected.real);
      int scaleDiffFortranImag = calculate_scale_difference(fortranComputed.imag, expected.imag);
      INFO("Record " << i << " scale differences (C++, Fortran): (real: (" << scaleDiffReal << ", " << scaleDiffFortranReal << "), imag: (" << scaleDiffImag << ", " << scaleDiffFortranImag << "));\n"
         << " computed: " << computed << ";\n"
         << " fortran:  " << fortranComputed << ";\n"
         << " expected: " << expected);
      // We require at least REQUIRED_SCALE_PRECISION digits of precision.
      REQUIRE( (scaleDiffReal >= REQUIRED_SCALE_PRECISION || scaleDiffReal == 0) );
      REQUIRE( (scaleDiffImag >= REQUIRED_SCALE_PRECISION || scaleDiffImag == 0) );
      REQUIRE( (scaleDiffFortranReal >= REQUIRED_SCALE_PRECISION || scaleDiffFortranReal == 0) );
      REQUIRE( (scaleDiffFortranImag >= REQUIRED_SCALE_PRECISION || scaleDiffFortranImag == 0) );
   }
}


TEST_CASE("ddcmul on device", "[kokkos][ddcomplex]") {

   // Structure for ddcmul test data: two ddcomplex inputs and one expected output.
   struct DataRecord {
      // Each ddcomplex has two ddfun::ddouble members.
      // For binary layout, assume ddcomplex is stored as: real, imag.
      ddfun::ddcomplex a;
      ddfun::ddcomplex b;
      ddfun::ddcomplex exp;
   };

   std::ifstream infile(INPUT_FILES_DIR + "ddcmul.bin", std::ios::binary);
   INFO("Reading " << INPUT_FILES_DIR + "ddcmul.bin");
   REQUIRE(infile.good());
   std::vector<DataRecord> hostData;
   std::vector<ddfun::ddcomplex> fortranResults;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      hostData.push_back(rec);
      // Compute Fortran result for each record
      double a[4] = {rec.a.real.hi, rec.a.real.lo, rec.a.imag.hi, rec.a.imag.lo};
      double b[4] = {rec.b.real.hi, rec.b.real.lo, rec.b.imag.hi, rec.b.imag.lo};
      double c[4] = {0.0, 0.0, 0.0, 0.0};
      ddcmul_fortran(a, b, c);
      fortranResults.push_back(ddfun::ddcomplex(ddfun::ddouble(c[0], c[1]), ddfun::ddouble(c[2], c[3])));
   }
   int N = hostData.size();
   INFO("Read " << N << " records.");
   REQUIRE(N > 0);

   Kokkos::View<ddfun::ddcomplex*, Kokkos::HostSpace> hA("hA", N);
   Kokkos::View<ddfun::ddcomplex*, Kokkos::HostSpace> hB("hB", N);
   for (int i = 0; i < N; i++) {
      hA(i) = hostData[i].a;
      hB(i) = hostData[i].b;
   }

   auto dA = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hA);
   auto dB = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hB);

   Kokkos::View<ddfun::ddcomplex*, Kokkos::DefaultExecutionSpace> dResults("dResults", N);
   Kokkos::parallel_for("compute_ddcmul", N, KOKKOS_LAMBDA(const int i) {
      // Build complex numbers from the record.
      dResults(i) = dA(i) * dB(i);
   });
   Kokkos::fence();

   auto hResults = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dResults);
   for (int i = 0; i < N; i++) {
      ddfun::ddcomplex computed = hResults(i);
      ddfun::ddcomplex expected = hostData[i].exp;
      ddfun::ddcomplex fortranComputed = fortranResults[i];

      // Compare C++ result with expected
      int scaleDiffReal = calculate_scale_difference(computed.real, expected.real);
      int scaleDiffImag = calculate_scale_difference(computed.imag, expected.imag);
      int scaleDiffRealFortran = calculate_scale_difference(computed.real, fortranComputed.real);
      int scaleDiffImagFortran = calculate_scale_difference(computed.imag, fortranComputed.imag);
      INFO("Record " << i << " scale differences (C++, Fortran): (real: (" << scaleDiffReal << ", " << scaleDiffRealFortran << "), imag: (" << scaleDiffImag << ", " << scaleDiffImagFortran << "));\n"
         << " computed: " << computed << ";\n"
         << " fortran:  " << fortranComputed << ";\n"
         << " expected: " << expected);
      REQUIRE( (scaleDiffReal >= REQUIRED_SCALE_PRECISION || scaleDiffReal == 0) );
      REQUIRE( (scaleDiffImag >= REQUIRED_SCALE_PRECISION || scaleDiffImag == 0) );
      REQUIRE( (scaleDiffRealFortran >= REQUIRED_SCALE_PRECISION || scaleDiffRealFortran == 0) );
      REQUIRE( (scaleDiffImagFortran >= REQUIRED_SCALE_PRECISION || scaleDiffImagFortran == 0) );
   }
}


TEST_CASE("ddcdiv on device", "[kokkos][ddcomplex]") {

   // Structure for ddcdiv test data: two ddcomplex inputs and one expected output.
   struct DataRecord {
      // Each ddcomplex has two ddfun::ddouble members.
      // For binary layout, assume ddcomplex is stored as: real, imag.
      ddfun::ddcomplex a;
      ddfun::ddcomplex b;
      ddfun::ddcomplex exp;
   };

   std::ifstream infile(INPUT_FILES_DIR + "ddcdiv.bin", std::ios::binary);
   INFO("Reading " << INPUT_FILES_DIR + "ddcdiv.bin");
   REQUIRE(infile.good());
   std::vector<DataRecord> hostData;
   DataRecord rec;
   std::vector<ddfun::ddcomplex> fortranResults;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      // store inputs
      hostData.push_back(rec);
      // calculate fortran results
      double a[4] = {rec.a.real.hi, rec.a.real.lo, rec.a.imag.hi, rec.a.imag.lo};
      double b[4] = {rec.b.real.hi, rec.b.real.lo, rec.b.imag.hi, rec.b.imag.lo};
      double c[4] = {0.0, 0.0, 0.0, 0.0};
      ddcdiv_fortran(a, b, c);
      fortranResults.push_back(ddfun::ddcomplex(ddfun::ddouble(c[0], c[1]), ddfun::ddouble(c[2], c[3])));
   }
   int N = hostData.size();
   INFO("Read " << N << " records.");
   REQUIRE(N > 0);

   Kokkos::View<ddfun::ddcomplex*, Kokkos::HostSpace> hA("hA", N);
   Kokkos::View<ddfun::ddcomplex*, Kokkos::HostSpace> hB("hB", N);
   for (int i = 0; i < N; i++) {
      hA(i) = hostData[i].a;
      hB(i) = hostData[i].b;
   }

   auto dA = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hA);
   auto dB = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hB);

   Kokkos::View<ddfun::ddcomplex*, Kokkos::DefaultExecutionSpace> dResults("dResults", N);
   Kokkos::parallel_for("compute_ddcdiv", N, KOKKOS_LAMBDA(const int i) {
      // Build complex numbers from the record.
      dResults(i) = dA(i) / dB(i);
   });
   Kokkos::fence();

   auto hResults = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dResults);
   for (int i = 0; i < N; i++) {
      ddfun::ddcomplex computed = hResults(i);
      ddfun::ddcomplex expected = hostData[i].exp;
      ddfun::ddcomplex fortranComputed = fortranResults[i];

      // Compare C++ result with expected
      int scaleDiffReal = calculate_scale_difference(computed.real, expected.real);
      int scaleDiffImag = calculate_scale_difference(computed.imag, expected.imag);
      int scaleDiffRealFortran = calculate_scale_difference(computed.real, fortranComputed.real);
      int scaleDiffImagFortran = calculate_scale_difference(computed.imag, fortranComputed.imag);
      INFO("Record " << i << " scale differences (C++, Fortran): (real: (" << scaleDiffReal << ", " << scaleDiffRealFortran << "), imag: (" << scaleDiffImag << ", " << scaleDiffImagFortran << "));\n"
         << " computed: " << computed << ";\n"
         << " fortran:  " << fortranComputed << ";\n"
         << " expected: " << expected);
      REQUIRE( (scaleDiffReal >= REQUIRED_SCALE_PRECISION || scaleDiffReal == 0) );
      REQUIRE( (scaleDiffImag >= REQUIRED_SCALE_PRECISION || scaleDiffImag == 0) );
      REQUIRE( (scaleDiffRealFortran >= REQUIRED_SCALE_PRECISION || scaleDiffRealFortran == 0) );
      REQUIRE( (scaleDiffImagFortran >= REQUIRED_SCALE_PRECISION || scaleDiffImagFortran == 0) );
   }
}



TEST_CASE("ddcpwr on device", "[kokkos][ddcomplex]") {

   // Structure for ddcpwr test data: two ddcomplex inputs and one expected output.
   struct DataRecord {
      // Each ddcomplex has two ddfun::ddouble members.
      // For binary layout, assume ddcomplex is stored as: real, imag.
      ddfun::ddcomplex a;
      int n;
      int ph;
      ddfun::ddcomplex exp;
   };

   std::ifstream infile(INPUT_FILES_DIR + "ddcpwr.bin", std::ios::binary);
   INFO("Reading " << INPUT_FILES_DIR + "ddcpwr.bin");
   REQUIRE(infile.good());
   std::vector<DataRecord> hostData;
   DataRecord rec;
   std::vector<ddfun::ddcomplex> fortranResults;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      hostData.push_back(rec);
      // calculate fortran results
      double a[4] = {rec.a.real.hi, rec.a.real.lo, rec.a.imag.hi, rec.a.imag.lo};
      int n = rec.n;
      double c[4] = {0.0, 0.0, 0.0, 0.0};
      ddcpwr_fortran(a, n, c);
      fortranResults.push_back(ddfun::ddcomplex(ddfun::ddouble(c[0], c[1]), ddfun::ddouble(c[2], c[3])));
   }
   int N = hostData.size();
   INFO("Read " << N << " records.");
   REQUIRE(N > 0);

   Kokkos::View<ddfun::ddcomplex*, Kokkos::HostSpace> hA("hA", N);
   Kokkos::View<int*, Kokkos::HostSpace> hN("hN", N);
   for (int i = 0; i < N; i++) {
      hA(i) = hostData[i].a;
      hN(i) = hostData[i].n;
   }

   auto dA = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hA);
   auto dN = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hN);

   Kokkos::View<ddfun::ddcomplex*, Kokkos::DefaultExecutionSpace> dResults("dResults", N);
   Kokkos::parallel_for("compute_ddcpwr", N, KOKKOS_LAMBDA(const int i) {
      // Build complex numbers from the record.
      dResults(i) = dA(i).pwr(dN(i));
   });
   Kokkos::fence();

   auto hResults = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dResults);
   for (int i = 0; i < N; i++) {
      ddfun::ddcomplex computed = hResults(i);
      ddfun::ddcomplex expected = hostData[i].exp;
      ddfun::ddcomplex fortranComputed = fortranResults[i];
      int scaleDiffReal = calculate_scale_difference(computed.real, expected.real);
      int scaleDiffImag = calculate_scale_difference(computed.imag, expected.imag);
      int scaleDiffRealFortran = calculate_scale_difference(fortranComputed.real, expected.real);
      int scaleDiffImagFortran = calculate_scale_difference(fortranComputed.imag, expected.imag);
      INFO("Record " << i << " scale differences (C++, Fortran): (real: (" << scaleDiffReal << ", " << scaleDiffRealFortran << "), imag: (" << scaleDiffImag << ", " << scaleDiffImagFortran << "));\n"
         << " computed: " << computed << ";\n"
         << " fortran:  " << fortranComputed << ";\n"
         << " expected: " << expected);
      REQUIRE( (scaleDiffReal >= REQUIRED_SCALE_PRECISION || scaleDiffReal == 0) );
      REQUIRE( (scaleDiffImag >= REQUIRED_SCALE_PRECISION || scaleDiffImag == 0) );
      REQUIRE( (scaleDiffRealFortran >= REQUIRED_SCALE_PRECISION || scaleDiffRealFortran == 0) );
      REQUIRE( (scaleDiffImagFortran >= REQUIRED_SCALE_PRECISION || scaleDiffImagFortran == 0) );
   }
}





//////////////////////////////////////////////////////////////////////////
// Example for ddcsshr: function with two outputs.
//////////////////////////////////////////////////////////////////////////


TEST_CASE("ddcsshr on device", "[kokkos][ddouble]") {
    // Structure for ddcsshr test data.
    struct DataRecord {
       ddfun::ddouble a;
       ddfun::ddouble x; // expected output x
       ddfun::ddouble y; // expected output y
    };
    std::ifstream infile( INPUT_FILES_DIR + "ddcsshr.bin", std::ios::binary);
    INFO("Reading " << INPUT_FILES_DIR + "ddcsshr.bin");
    REQUIRE(infile.good());
    std::vector<DataRecord> hostData;
    std::vector<std::pair<ddfun::ddouble, ddfun::ddouble>> fortranResults;
    DataRecord rec;
    while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
       hostData.push_back(rec);
       // calculate fortran results
       double a[2] = {rec.a.hi, rec.a.lo};
       double x[2] = {0.0, 0.0};
       double y[2] = {0.0, 0.0};
       ddcsshr_fortran(a, x, y);
       fortranResults.push_back({ddfun::ddouble(x[0], x[1]), ddfun::ddouble(y[0], y[1])});
    }
    int N = hostData.size();
    INFO("Read " << N << " records.");
    REQUIRE(N > 0);

    Kokkos::View<ddfun::ddouble*, Kokkos::HostSpace> hA("hA", N);
    for (int i = 0; i < N; i++) {
       hA(i) = hostData[i].a;
    }

    auto dA = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hA);
    
    // We'll create two device views to hold the two outputs.
    Kokkos::View<ddfun::ddouble*, Kokkos::DefaultExecutionSpace> dX("dX", N);
    Kokkos::View<ddfun::ddouble*, Kokkos::DefaultExecutionSpace> dY("dY", N);
    Kokkos::parallel_for("compute_ddcsshr", N, KOKKOS_LAMBDA(const int i) {
       ddfun::ddouble a = dA(i);
       ddfun::ddouble x, y;
       ddcsshr(a, x, y);
       dX(i) = x;
       dY(i) = y;
    });
    Kokkos::fence();

    auto hX = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dX);
    auto hY = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dY);
    for (int i = 0; i < N; i++) {
       ddfun::ddouble computedX = hX(i);
       ddfun::ddouble computedY = hY(i);
       ddfun::ddouble expectedX = hostData[i].x;
       ddfun::ddouble expectedY = hostData[i].y;
       ddfun::ddouble fortranX = fortranResults[i].first;
       ddfun::ddouble fortranY = fortranResults[i].second;
       int scaleDiffX = calculate_scale_difference(computedX, expectedX);
       int scaleDiffY = calculate_scale_difference(computedY, expectedY);
       int scaleDiffXFortran = calculate_scale_difference(fortranX, expectedX);
       int scaleDiffYFortran = calculate_scale_difference(fortranY, expectedY);
       INFO("ddcsshr Record " << i << " scale differences (C++, Fortran): (x: (" << scaleDiffX << ", " << scaleDiffXFortran << "), y: (" << scaleDiffY << ", " << scaleDiffYFortran << "));"
          << " computed: (" << computedX << ", " << computedY << ");"
          << " fortran: (" << fortranX << ", " << fortranY << ");"
          << " expected: (" << expectedX << ", " << expectedY << ")");
       REQUIRE( (scaleDiffX >= 20 || scaleDiffX == 0) );
       REQUIRE( (scaleDiffY >= 20 || scaleDiffY == 0) );
       REQUIRE( (scaleDiffXFortran >= 20 || scaleDiffXFortran == 0) );
       REQUIRE( (scaleDiffYFortran >= 20 || scaleDiffYFortran == 0) );
    }
}



TEST_CASE("ddcssnr on device", "[kokkos][ddouble]") {
    // Structure for ddcssnr test data.
    struct DataRecord {
       ddfun::ddouble a;
       ddfun::ddouble x; // expected output x
       ddfun::ddouble y; // expected output y
    };
    std::ifstream infile( INPUT_FILES_DIR + "ddcssnr.bin", std::ios::binary);
    INFO("Reading " << INPUT_FILES_DIR + "ddcssnr.bin");
    REQUIRE(infile.good());
    std::vector<DataRecord> hostData;
    std::vector<std::pair<ddfun::ddouble, ddfun::ddouble>> fortranResults;
    DataRecord rec;
    while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
       hostData.push_back(rec);
       // calculate fortran results
       double a[2] = {rec.a.hi, rec.a.lo};
       double x[2] = {0.0, 0.0};
       double y[2] = {0.0, 0.0};
       ddcssnr_fortran(a, x, y);
       fortranResults.push_back({ddfun::ddouble(x[0], x[1]), ddfun::ddouble(y[0], y[1])});
    }
    int N = hostData.size();
    INFO("Read " << N << " records.");
    REQUIRE(N > 0);

    Kokkos::View<ddfun::ddouble*, Kokkos::HostSpace> hA("hA", N);
    for (int i = 0; i < N; i++) {
       hA(i) = hostData[i].a;
    }

    auto dA = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hA);
    
    // We'll create two device views to hold the two outputs.
    Kokkos::View<ddfun::ddouble*, Kokkos::DefaultExecutionSpace> dX("dX", N);
    Kokkos::View<ddfun::ddouble*, Kokkos::DefaultExecutionSpace> dY("dY", N);
    Kokkos::parallel_for("compute_ddcssnr", N, KOKKOS_LAMBDA(const int i) {
       ddfun::ddouble a = dA(i);
       ddfun::ddouble x, y;
       ddcssnr(a, x, y);
       dX(i) = x;
       dY(i) = y;
    });
    Kokkos::fence();

    auto hX = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dX);
    auto hY = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dY);
    for (int i = 0; i < N; i++) {
       ddfun::ddouble computedX = hX(i);
       ddfun::ddouble computedY = hY(i);
       ddfun::ddouble expectedX = hostData[i].x;
       ddfun::ddouble expectedY = hostData[i].y;
       ddfun::ddouble fortranX = fortranResults[i].first;
       ddfun::ddouble fortranY = fortranResults[i].second;
       int scaleDiffX = calculate_scale_difference(computedX, expectedX);
       int scaleDiffY = calculate_scale_difference(computedY, expectedY);
       int scaleDiffXFortran = calculate_scale_difference(fortranX, expectedX);
       int scaleDiffYFortran = calculate_scale_difference(fortranY, expectedY);
       INFO("ddcssnr Record " << i << " scale differences (C++, Fortran): (x: (" << scaleDiffX << ", " << scaleDiffXFortran << "), y: (" << scaleDiffY << ", " << scaleDiffYFortran << "));"
          << " computed: (" << computedX << ", " << computedY << ");"
          << " fortran: (" << fortranX << ", " << fortranY << ");"
          << " expected: (" << expectedX << ", " << expectedY << ")");
       REQUIRE( (scaleDiffX >= 20 || scaleDiffX == 0) );
       REQUIRE( (scaleDiffY >= 20 || scaleDiffY == 0) );
       REQUIRE( (scaleDiffXFortran >= 20 || scaleDiffXFortran == 0) );
       REQUIRE( (scaleDiffYFortran >= 20 || scaleDiffYFortran == 0) );
    }
}



TEST_CASE("ddagmr on device using mirror views", "[kokkos][ddouble]") {

   // Structure to hold ddagmr test data. 
   // Since ddfun::ddouble is just two contiguous doubles, this struct is tightly packed.
   struct DataRecord {
      ddfun::ddouble a;
      ddfun::ddouble b;
      ddfun::ddouble exp;
   };

   // Read test data from file into a vector of DataRecord.
   std::ifstream infile(INPUT_FILES_DIR + "ddagmr.bin", std::ios::binary);
   INFO("Reading " << INPUT_FILES_DIR + "ddagmr.bin");
   REQUIRE(infile.good());
   std::vector<DataRecord> hostRecords;
   std::vector<ddfun::ddouble> fortranResults;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      hostRecords.push_back(rec);
      // Compute Fortran result
      double a[2] = {rec.a.hi, rec.a.lo};
      double b[2] = {rec.b.hi, rec.b.lo};
      double c[2] = {0.0, 0.0};
      ddagmr_fortran(a, b, c);
      fortranResults.push_back(ddfun::ddouble(c[0], c[1]));
   }
   int N = hostRecords.size();
   INFO("Read " << N << " records.");
   REQUIRE(N > 0);

   // Create a host-side Kokkos::View from the vector.
   Kokkos::View<ddfun::ddouble*, Kokkos::HostSpace> hA("hA", N);
   Kokkos::View<ddfun::ddouble*, Kokkos::HostSpace> hB("hB", N);
   for (int i = 0; i < N; i++) {
      hA(i) = hostRecords[i].a;
      hB(i) = hostRecords[i].b;
   }

   // Create a device view by deep copying the host view.
   auto dA = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hA);
   auto dB = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hB);

   // Create a device view to hold the computed results.
   Kokkos::View<ddfun::ddouble*, Kokkos::DefaultExecutionSpace> dResults("dResults", N);

   // Launch a kernel to compute ddagmr on each input.
   Kokkos::parallel_for("compute_ddagmr", N, KOKKOS_LAMBDA(const int i) {
      dResults(i) = ddfun::ddagmr(dA(i),dB(i));
   });
   Kokkos::fence();

   // Create a host mirror of the results.
   auto hResults = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dResults);

   // Validate the results.
   for (int i = 0; i < N; i++) {
      ddfun::ddouble computed = hResults(i);
      ddfun::ddouble expected = hostRecords[i].exp;
      ddfun::ddouble fortranComputed = fortranResults[i];
      int scaleDiff = calculate_scale_difference(computed, expected);
      int scaleDiffFortran = calculate_scale_difference(fortranComputed, expected);
      INFO("Record " << i << " scale differences (C++, Fortran): (" << scaleDiff << ", " << scaleDiffFortran << ");\n"
         << " computed: " << computed << ";\n"
         << " fortran:  " << fortranComputed << ";\n"
         << " expected: " << expected);
      // We require at least REQUIRED_SCALE_PRECISION digits of precision.
      REQUIRE( (scaleDiff >= REQUIRED_SCALE_PRECISION || scaleDiff == 0) );
      REQUIRE( (scaleDiffFortran >= REQUIRED_SCALE_PRECISION || scaleDiffFortran == 0) );
   }
}


/**
 * ddang implements the following algorithm:
 *
 * Let x and y be double-double numbers.  The function ddang(x, y) returns the
 * angle in radians subtended at the origin by the points (0,0) and (x,y).  The
 * range of the result is (-pi,pi] and the function satisfies the usual
 * identities for the arctangent function.
 *
 * The algorithm is based on the following observation.  If x and y are both
 * non-zero, then the angle in radians subtended at the origin by the points
 * (0,0) and (x,y) is the same as the angle subtended by the points (0,0) and
 * (x/y, 1), which is just arctan(x/y).  If x is zero, the angle is either zero
 * or pi, depending on the sign of y.  If y is zero, the angle is either pi/2 or
 * -pi/2, depending on the sign of x.  The function ddang handles all of these
 * cases correctly.
 *
 * The algorithm is implemented in terms of the following steps:
 *
 * 1. If x is zero, return either zero or pi, depending on the sign of y.
 *
 * 2. If y is zero, return either pi/2 or -pi/2, depending on the sign of x.
 *
 * 3. Otherwise, return arctan(x/y).
 */

TEST_CASE("ddang on device using mirror views", "[kokkos][ddouble]") {
   struct DataRecord {
      // Structure to hold ddang test data. 
      // Since ddfun::ddouble is just two contiguous doubles, this struct is tightly packed.
      ddfun::ddouble x;
      ddfun::ddouble y;
      ddfun::ddouble exp;
   };

   // Read test data from file into a vector of DataRecord.
   std::string filename = INPUT_FILES_DIR + "ddang.bin";
   std::ifstream infile(filename, std::ios::binary);
   INFO("Reading " << filename);
   REQUIRE(infile.good());
   std::vector<DataRecord> hostRecords;
   std::vector<ddfun::ddouble> fortranResults;
   DataRecord rec;

   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      hostRecords.push_back(rec);
      // Compute Fortran result
      double x[2] = {rec.x.hi, rec.x.lo};
      double y[2] = {rec.y.hi, rec.y.lo};
      double a[2] = {0.0, 0.0};
      ddang_fortran(x, y, a);
      fortranResults.push_back(ddfun::ddouble(a[0], a[1]));
   }
   int N = hostRecords.size();
   INFO("Read " << N << " records.");
   REQUIRE(N > 0);

   // Create host Views for input and output
   Kokkos::View<ddfun::ddouble*, Kokkos::HostSpace> hX("hX", N);
   Kokkos::View<ddfun::ddouble*, Kokkos::HostSpace> hY("hY", N);
   for (int i = 0; i < N; i++) {
      hX(i) = hostRecords[i].x;
      hY(i) = hostRecords[i].y;
   }

   // Create device Views
   auto dX = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hX);
   auto dY = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hY);
   Kokkos::View<ddfun::ddouble*, Kokkos::DefaultExecutionSpace> dResults("dResults", N);

   // Launch kernel to compute angles
   Kokkos::parallel_for("compute_ddang", N, KOKKOS_LAMBDA(const int i) {
      dResults(i) = ddfun::ddang(dX(i), dY(i));
   });
   Kokkos::fence();

   // Copy results back to host
   auto hResults = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dResults);

   // Validate results
   for (int i = 0; i < N; i++) {
      ddfun::ddouble computed = hResults(i);
      ddfun::ddouble expected = hostRecords[i].exp;
      ddfun::ddouble fortranResult = fortranResults[i];

      int scaleDiff = calculate_scale_difference(computed, expected);
      int scaleDiffFortran = calculate_scale_difference(computed, fortranResult);

      INFO("Record " << i << " scale differences (C++, Fortran): (" << scaleDiff << ", " << scaleDiffFortran << ");\n"
           << " Point (" << hostRecords[i].x << ", " << hostRecords[i].y << ");\n"
           << " computed: " << computed << ";\n"
           << " fortran:  " << fortranResult << ";\n"
           << " expected: " << expected);

      // We require at least REQUIRED_SCALE_PRECISION digits of precision
      REQUIRE((scaleDiff >= REQUIRED_SCALE_PRECISION || scaleDiff == 0));
      REQUIRE((scaleDiffFortran >= REQUIRED_SCALE_PRECISION || scaleDiffFortran == 0));
   }
}

TEST_CASE("ddpolyr", "[ddmath]") {
   constexpr int MAX_DEGREE = 4;
   struct DataRecord {
      int n;
      ddfun::ddouble a[MAX_DEGREE+1];  // Maximum degree 4
      ddfun::ddouble x0;
      ddfun::ddouble exp;
   };

   std::string filename = INPUT_FILES_DIR + "ddpolyr.bin";
   std::ifstream infile(filename, std::ios::binary);
   INFO("Reading " << filename);
   REQUIRE(infile.good());

   // Read all records into vectors
   std::vector<DataRecord> records;
   std::vector<ddfun::ddouble> fortranResults;
   while (infile.peek() != EOF) {
      DataRecord record;
      double fortran_a[2*(MAX_DEGREE+1)];
      // Read n and placeholder
      int n =0,ph=0;
      infile.read(reinterpret_cast<char*>(&n), sizeof(int));
      infile.read(reinterpret_cast<char*>(&ph), sizeof(int));
      record.n = n;
      // Read coefficients
      for (int i = 0; i <= MAX_DEGREE; ++i) {
         if (i > record.n) {
            record.a[i] = ddfun::ddouble(0.0, 0.0);
         } else {
            double hi, lo;
            infile.read(reinterpret_cast<char*>(&hi), sizeof(double));
            infile.read(reinterpret_cast<char*>(&lo), sizeof(double));
            record.a[i] = ddfun::ddouble(hi, lo);
            fortran_a[2*i] = hi;
            fortran_a[2*i+1] = lo;
         }
      }
      
      // Read x0
      double x0_hi, x0_lo;
      infile.read(reinterpret_cast<char*>(&x0_hi), sizeof(double));
      infile.read(reinterpret_cast<char*>(&x0_lo), sizeof(double));
      record.x0 = ddfun::ddouble(x0_hi, x0_lo);
      
      // Read expected result
      double exp_hi, exp_lo;
      infile.read(reinterpret_cast<char*>(&exp_hi), sizeof(double));
      infile.read(reinterpret_cast<char*>(&exp_lo), sizeof(double));
      record.exp = ddfun::ddouble(exp_hi, exp_lo);

      records.push_back(record);

      // call fortran ddpolyr
      double x0[2] = {record.x0.hi, record.x0.lo};
      double exp[2] = {0.0, 0.0};
      ddpolyr_fortran(record.n, fortran_a, x0, exp);
      fortranResults.push_back(ddfun::ddouble(exp[0], exp[1]));
   }

   const int num_records = records.size();
   REQUIRE(num_records > 0);
   INFO("Read " << num_records << " records.");

   // Create Kokkos Views for input and output
   Kokkos::View<int*, Kokkos::HostSpace> hn("n", num_records);
   Kokkos::View<ddfun::ddouble**, Kokkos::HostSpace> ha("a", num_records, MAX_DEGREE+1);
   Kokkos::View<ddfun::ddouble*, Kokkos::HostSpace> hx0("x0", num_records);
   Kokkos::View<ddfun::ddouble*, Kokkos::HostSpace> hResults("result", num_records);

   // fill host views
   for (int i = 0; i < num_records; ++i) {
      hn(i) = records[i].n;
      for (int j = 0; j <= records[i].n; ++j) {
         ha(i, j) = records[i].a[j];
      }
      hx0(i) = records[i].x0;
   }

   // Create host mirrors and copy data
   auto dn = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hn);
   auto da = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), ha);
   auto dx0 = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hx0);
   auto dResults = Kokkos::create_mirror_view(Kokkos::DefaultExecutionSpace(), hResults);

   // Run the test in parallel
   Kokkos::parallel_for("ddpolyr_test", num_records, KOKKOS_LAMBDA(const int i) {
      // Call ddpolyr
      dResults(i) = ddfun::ddpolyr(dn(i), Kokkos::subview(da,i,Kokkos::ALL), dx0(i));
   });
   Kokkos::fence();
   // Copy results back to host
   Kokkos::deep_copy(hResults, dResults);

   // Validate results
   for (int i = 0; i < num_records; ++i) {
      ddfun::ddouble computed = hResults(i);
      ddfun::ddouble expected = records[i].exp;
      ddfun::ddouble fortranResult = fortranResults[i];

      int scaleDiff = calculate_scale_difference(computed, expected);
      int scaleDiffFortran = calculate_scale_difference(computed, fortranResult);

      INFO("Record " << i << " scale differences (C++, Fortran): (" << scaleDiff << ", " << scaleDiffFortran << ");\n"
           << " computed: " << computed << ";\n"
           << " fortran:  " << fortranResult << ";\n"
           << " expected: " << expected);

      // We require at least REQUIRED_SCALE_PRECISION digits of precision
      REQUIRE((scaleDiff >= REQUIRED_SCALE_PRECISION || scaleDiff == 0));
      REQUIRE((scaleDiffFortran >= REQUIRED_SCALE_PRECISION || scaleDiffFortran == 0));
   }
}

TEST_CASE("DDRandom generates uniform and normal distributions", "[kokkos][ddouble][random]") {
    // Initialize random number generator
    INFO("Initializing random number generator with seed 321233");
    ddfun::DDRandom rng(321233);
    
    // Create views for results
    INFO("Creating views for results");
    const int N = 10000;
    Kokkos::View<ddfun::ddouble*> d_uniform("uniform_randoms", N);
    Kokkos::View<ddfun::ddouble*> d_normal("normal_randoms", N);
    
    // Generate random numbers
    INFO("Generating random numbers");
    rng.generate_uniform(d_uniform);
    INFO("Generated uniform random numbers");
    rng.generate_normal(d_normal);
    
    // Copy to host for validation
    INFO("Copying to host for validation");
    auto h_uniform = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_uniform);
    auto h_normal = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_normal);
    
    // Test uniform distribution
    double uniform_mean = 0.0;
    for(int i = 0; i < N; i++) {
        REQUIRE(h_uniform(i).hi >= 0.0);
        REQUIRE(h_uniform(i).hi < 1.0);
        uniform_mean += h_uniform(i).hi;
    }
    uniform_mean /= N;
    INFO("Uniform mean: " << uniform_mean);
    REQUIRE(std::abs(uniform_mean - 0.5) < 0.05);  // Should be close to 0.5
    
    // Test normal distribution
    double normal_mean = 0.0;
    double normal_var = 0.0;
    for(int i = 0; i < N; i++) {
        normal_mean += h_normal(i).hi;
    }
    normal_mean /= N;
    for(int i = 0; i < N; i++) {
        normal_var += (h_normal(i).hi - normal_mean) * (h_normal(i).hi - normal_mean);
    }
    normal_var /= (N - 1);
    
    INFO("Normal mean: " << normal_mean << ", variance: " << normal_var);
    REQUIRE(std::abs(normal_mean) < 0.1);      // Should be close to 0
    REQUIRE(std::abs(normal_var - 1.0) < 0.1); // Should be close to 1
}

TEST_CASE("DDRandom device-callable functions", "[kokkos][ddouble][random]") {
    // Initialize random number generator
    INFO("Initializing random number generator with seed 123456");
    ddfun::DDRandom rng(123456);
    
    // Create views for results
    INFO("Creating views for results");
    const int N = 10000;
    Kokkos::View<ddfun::ddouble*> d_results("results", N);
    
    // Capture pointer to rng for device
    const ddfun::DDRandom* rng_ptr = &rng;
    
    // Test device-callable methods in a kernel
    INFO("Running kernel to test device-callable methods");
    Kokkos::parallel_for("test_device_random", N, 
    KOKKOS_LAMBDA(const int i) {
        // Get a generator instance for this thread
        auto gen = rng_ptr->get_state();
        
        // Use the appropriate random function based on index
        if (i % 2 == 0) {
            // Even indices: uniform random
            d_results(i) = rng_ptr->get_uniform_ddouble(gen);
        } else {
            // Odd indices: normal random
            d_results(i) = rng_ptr->get_normal_ddouble(gen);
        }
        
        // Return the generator
        rng_ptr->free_state(gen);
    });
    Kokkos::fence();
    
    INFO("Copying to host for validation");
    auto h_results = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_results);
    
    // Test values
    double uniform_min = 0.0;
    double uniform_max = 1.0;
    int uniform_count = 0;
    double uniform_sum = 0.0;
    
    double normal_sum = 0.0;
    int normal_count = 0;
    
    for(int i = 0; i < N; i++) {
        if (i % 2 == 0) {
            // Uniform randoms
            uniform_count++;
            REQUIRE(h_results(i).hi >= 0.0);
            REQUIRE(h_results(i).hi < 1.0);
            uniform_sum += h_results(i).hi;
        } else {
            // Normal randoms
            normal_count++;
            normal_sum += h_results(i).hi;
        }
    }
    
    double uniform_mean = uniform_sum / uniform_count;
    double normal_mean = normal_sum / normal_count;
    
    INFO("Uniform mean: " << uniform_mean);
    INFO("Normal mean: " << normal_mean);
    
    // Check that uniform mean is close to 0.5
    REQUIRE(std::abs(uniform_mean - 0.5) < 0.05);
    
    // Check that normal mean is close to 0
    REQUIRE(std::abs(normal_mean) < 0.1);
}
