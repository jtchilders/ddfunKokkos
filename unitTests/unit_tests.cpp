#include <Kokkos_Core.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_session.hpp>
#include "ddouble.hpp"
#include "ddcomplex.hpp"
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

const std::string INPUT_FILES_DIR("/home/jchilders/git/precisionStudies/ddfun/ddTestInputs/");

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
   std::string filename = INPUT_FILES_DIR + "data/ddadd.bin";
   std::ifstream infile(filename, std::ios::binary);
   INFO("Reading " << filename);
   REQUIRE(infile.good());
   std::vector<DataRecord> hostRecords;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      hostRecords.push_back(rec);
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
      ddfun::ddouble computed = hResults(i);
      ddfun::ddouble expected = hostRecords[i].exp;
      int scaleDiff = calculate_scale_difference(computed, expected);
      INFO("Record " << i << " scale difference: " << scaleDiff << "; computed: " << computed << "; expected: " << expected);
      // We require at least 20 digits of precision.
      REQUIRE( (scaleDiff >= 20 || scaleDiff == 0) );
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
   std::string filename = INPUT_FILES_DIR + "data/ddsub.bin";
   std::ifstream infile(filename, std::ios::binary);
   INFO("Reading " << filename);
   REQUIRE(infile.good());
   std::vector<DataRecord> hostRecords;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      hostRecords.push_back(rec);
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
      ddfun::ddouble computed = hResults(i);
      ddfun::ddouble expected = hostRecords[i].exp;
      int scaleDiff = calculate_scale_difference(computed, expected);
      INFO("Record " << i << " scale difference: " << scaleDiff << "; computed: " << computed << "; expected: " << expected);
      // We require at least 20 digits of precision.
      REQUIRE( (scaleDiff >= 20 || scaleDiff == 0) );
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
   std::string filename = INPUT_FILES_DIR + "data/ddmul.bin";
   std::ifstream infile(filename, std::ios::binary);
   INFO("Reading " << filename);
   REQUIRE(infile.good());
   std::vector<DataRecord> hostRecords;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      hostRecords.push_back(rec);
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
      ddfun::ddouble computed = hResults(i);
      ddfun::ddouble expected = hostRecords[i].exp;
      int scaleDiff = calculate_scale_difference(computed, expected);
      INFO("Record " << i << " scale difference: " << scaleDiff << "; computed: " << computed << "; expected: " << expected);
      // We require at least 20 digits of precision.
      REQUIRE( (scaleDiff >= 20 || scaleDiff == 0) );
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
   std::string filename = INPUT_FILES_DIR + "data/ddmuld.bin";
   std::ifstream infile(filename, std::ios::binary);
   INFO("Reading " << filename);
   REQUIRE(infile.good());
   std::vector<DataRecord> hostRecords;
   std::vector<ddfun::ddouble> fortranResults;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      hostRecords.push_back(rec);
      // run Fortran version
      fortranResults.push_back(ddfun::ddmuld(rec.a,rec.b));
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
      int scaleDiff = calculate_scale_difference(computed, expected);
      INFO("Record " << i << " scale difference: " << scaleDiff << "; computed: " << computed << "; expected: " << expected);
      // We require at least 20 digits of precision.
      REQUIRE( (scaleDiff >= 20 || scaleDiff == 0) );
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
   std::string filename = INPUT_FILES_DIR + "data/ddmuldd.bin";
   std::ifstream infile(filename, std::ios::binary);
   INFO("Reading " << filename);
   REQUIRE(infile.good());
   std::vector<DataRecord> hostRecords;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      hostRecords.push_back(rec);
   }
   int N = hostRecords.size();
   INFO("Read " << N << " records.");
   REQUIRE(N > 0);

   // Create a host-side Kokkos::View from the vector.
   Kokkos::View<double*, Kokkos::HostSpace> hA("hA", N);
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
      dResults(i) = ddfun::ddmuldd(dA(i),dB(i));
   });
   Kokkos::fence();

   // Create a host mirror of the results.
   auto hResults = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dResults);

   // Validate the results.
   for (int i = 0; i < N; i++) {
      ddfun::ddouble computed = hResults(i);
      ddfun::ddouble expected = hostRecords[i].exp;
      int scaleDiff = calculate_scale_difference(computed, expected);
      INFO("Record " << i << " scale difference: " << scaleDiff << "; computed: " << computed << "; expected: " << expected);
      // We require at least 20 digits of precision.
      REQUIRE( (scaleDiff >= 20 || scaleDiff == 0) );
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
   std::string filename = INPUT_FILES_DIR + "data/dddiv.bin";
   std::ifstream infile(filename, std::ios::binary);
   INFO("Reading " << filename);
   REQUIRE(infile.good());
   std::vector<DataRecord> hostRecords;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      hostRecords.push_back(rec);
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
      ddfun::ddouble computed = hResults(i);
      ddfun::ddouble expected = hostRecords[i].exp;
      int scaleDiff = calculate_scale_difference(computed, expected);
      INFO("Record " << i << " scale difference: " << scaleDiff << "; computed: " << computed << "; expected: " << expected);
      // We require at least 20 digits of precision.
      REQUIRE( (scaleDiff >= 20 || scaleDiff == 0) );
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
   std::string filename = INPUT_FILES_DIR + "data/dddivd.bin";
   std::ifstream infile(filename, std::ios::binary);
   INFO("Reading " << filename);
   REQUIRE(infile.good());
   std::vector<DataRecord> hostRecords;
   std::vector<ddfun::ddouble> fortranResults;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      hostRecords.push_back(rec);
      // run Fortran to validate the results
      fortranResults.push_back(ddfun::dddivd(rec.a,rec.b));
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
      int scaleDiff = calculate_scale_difference(computed, expected);
      int scaleDiffFortran = calculate_scale_difference(computed, fortranResults[i]);
      INFO("dddivd: a = " << ddouble_to_hex(a) << " * b = " << ddouble_to_hex(b));
      INFO("Record " << i << " scale difference: " << scaleDiff << ";\n computed: " << ddouble_to_hex(computed) 
         << ";\n expected: " << ddouble_to_hex(expected) << ";\n fortran: " << ddouble_to_hex(fortranResults[i]));
      // We require at least 20 digits of precision.
      REQUIRE( (scaleDiff >= 20 || scaleDiff == 0) );
      REQUIRE( (scaleDiffFortran >= 20 || scaleDiffFortran == 0) );
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
   std::ifstream infile(INPUT_FILES_DIR + "data/ddabs.bin", std::ios::binary);
   INFO("Reading " << INPUT_FILES_DIR + "data/ddabs.bin");
   REQUIRE(infile.good());
   std::vector<DataRecord> hostRecords;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      hostRecords.push_back(rec);
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
      dResults(i) = dA(i).abs();
   });
   Kokkos::fence();

   // Create a host mirror of the results.
   auto hResults = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dResults);

   // Validate the results.
   for (int i = 0; i < N; i++) {
      ddfun::ddouble computed = hResults(i);
      ddfun::ddouble expected = hostRecords[i].exp;
      int scaleDiff = calculate_scale_difference(computed, expected);
      INFO("Record " << i << " scale difference: " << scaleDiff << "; computed: " << computed << "; expected: " << expected);
      // We require at least 20 digits of precision.
      REQUIRE( (scaleDiff >= 20 || scaleDiff == 0) );
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
   std::ifstream infile(INPUT_FILES_DIR + "data/ddnint.bin", std::ios::binary);
   INFO("Reading " << INPUT_FILES_DIR + "data/ddnint.bin");
   REQUIRE(infile.good());
   std::vector<DataRecord> hostRecords;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      hostRecords.push_back(rec);
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
      dResults(i) = dA(i).nint();
   });
   Kokkos::fence();

   // Create a host mirror of the results.
   auto hResults = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dResults);

   // Validate the results.
   for (int i = 0; i < N; i++) {
      ddfun::ddouble computed = hResults(i);
      ddfun::ddouble expected = hostRecords[i].exp;
      int scaleDiff = calculate_scale_difference(computed, expected);
      INFO("Record " << i << " scale difference: " << scaleDiff << "; computed: " << computed << "; expected: " << expected);
      // We require at least 20 digits of precision.
      REQUIRE( (scaleDiff >= 20 || scaleDiff == 0) );
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
   std::ifstream infile(INPUT_FILES_DIR + "data/ddsqrt.bin", std::ios::binary);
   INFO("Reading " << INPUT_FILES_DIR + "data/ddsqrt.bin");
   REQUIRE(infile.good());
   std::vector<DataRecord> hostRecords;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      hostRecords.push_back(rec);
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
      dResults(i) = dA(i).sqrt();
   });
   Kokkos::fence();

   // Create a host mirror of the results.
   auto hResults = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dResults);

   // Validate the results.
   for (int i = 0; i < N; i++) {
      ddfun::ddouble computed = hResults(i);
      ddfun::ddouble expected = hostRecords[i].exp;
      int scaleDiff = calculate_scale_difference(computed, expected);
      INFO("Record " << i << " scale difference: " << scaleDiff << "; computed: " << computed << "; expected: " << expected);
      // We require at least 20 digits of precision.
      REQUIRE( (scaleDiff >= 20 || scaleDiff == 0) );
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
   std::ifstream infile(INPUT_FILES_DIR + "data/ddexp.bin", std::ios::binary);
   INFO("Reading " << INPUT_FILES_DIR + "data/ddexp.bin");
   REQUIRE(infile.good());
   std::vector<DataRecord> hostRecords;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      hostRecords.push_back(rec);
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
      dResults(i) = dA(i).exp();
   });
   Kokkos::fence();

   // Create a host mirror of the results.
   auto hResults = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dResults);

   // Validate the results.
   for (int i = 0; i < N; i++) {
      ddfun::ddouble computed = hResults(i);
      ddfun::ddouble expected = hostRecords[i].exp;
      int scaleDiff = calculate_scale_difference(computed, expected);
      INFO("Record " << i << " scale difference: " << scaleDiff << "; computed: " << computed << "; expected: " << expected);
      // We require at least 20 digits of precision.
      REQUIRE( (scaleDiff >= 20 || scaleDiff == 0) );
   }
}


TEST_CASE("ddlog on device using mirror views", "[kokkos][ddouble]") {

   // Structure to hold ddlog test data. 
   // Since ddfun::ddouble is just two contiguous doubles, this struct is tightly packed.
   struct DataRecord {
      ddfun::ddouble a;
      ddfun::ddouble exp;
   };

   // Read test data from file into a vector of DataRecord.
   std::ifstream infile(INPUT_FILES_DIR + "data/ddlog.bin", std::ios::binary);
   INFO("Reading " << INPUT_FILES_DIR + "data/ddlog.bin");
   REQUIRE(infile.good());
   std::vector<DataRecord> hostRecords;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      hostRecords.push_back(rec);
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
   Kokkos::parallel_for("compute_ddlog", N, KOKKOS_LAMBDA(const int i) {
      dResults(i) = dA(i).log();
   });
   Kokkos::fence();

   // Create a host mirror of the results.
   auto hResults = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dResults);

   // Validate the results.
   for (int i = 0; i < N; i++) {
      ddfun::ddouble computed = hResults(i);
      ddfun::ddouble expected = hostRecords[i].exp;
      int scaleDiff = calculate_scale_difference(computed, expected);
      INFO("Record " << i << " scale difference: " << scaleDiff << "; computed: " << computed << "; expected: " << expected);
      // We require at least 20 digits of precision.
      REQUIRE( (scaleDiff >= 20 || scaleDiff == 0) );
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
   std::ifstream infile(INPUT_FILES_DIR + "data/ddacosh.bin", std::ios::binary);
   INFO("Reading " << INPUT_FILES_DIR + "data/ddacosh.bin");
   REQUIRE(infile.good());
   std::vector<DataRecord> hostRecords;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      hostRecords.push_back(rec);
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
      dResults(i) = dA(i).acosh();
   });
   Kokkos::fence();

   // Create a host mirror of the results.
   auto hResults = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dResults);

   // Validate the results.
   for (int i = 0; i < N; i++) {
      ddfun::ddouble computed = hResults(i);
      ddfun::ddouble expected = hostRecords[i].exp;
      int scaleDiff = calculate_scale_difference(computed, expected);
      INFO("Record " << i << " scale difference: " << scaleDiff << "; computed: " << computed << "; expected: " << expected);
      // We require at least 20 digits of precision.
      REQUIRE( (scaleDiff >= 20 || scaleDiff == 0) );
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
   std::ifstream infile(INPUT_FILES_DIR + "data/ddatanh.bin", std::ios::binary);
   INFO("Reading " << INPUT_FILES_DIR + "data/ddatanh.bin");
   REQUIRE(infile.good());
   std::vector<DataRecord> hostRecords;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      hostRecords.push_back(rec);
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
      dResults(i) = dA(i).atanh();
   });
   Kokkos::fence();

   // Create a host mirror of the results.
   auto hResults = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dResults);

   // Validate the results.
   for (int i = 0; i < N; i++) {
      ddfun::ddouble computed = hResults(i);
      ddfun::ddouble expected = hostRecords[i].exp;
      int scaleDiff = calculate_scale_difference(computed, expected);
      INFO("Record " << i << " scale difference: " << scaleDiff << "; computed: " << computed << "; expected: " << expected);
      // We require at least 20 digits of precision.
      REQUIRE( (scaleDiff >= 20 || scaleDiff == 0) );
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
   std::ifstream infile(INPUT_FILES_DIR + "data/ddasinh.bin", std::ios::binary);
   INFO("Reading " << INPUT_FILES_DIR + "data/ddasinh.bin");
   REQUIRE(infile.good());
   std::vector<DataRecord> hostRecords;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      hostRecords.push_back(rec);
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
      dResults(i) = dA(i).asinh();
   });
   Kokkos::fence();

   // Create a host mirror of the results.
   auto hResults = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dResults);

   // Validate the results.
   for (int i = 0; i < N; i++) {
      ddfun::ddouble computed = hResults(i);
      ddfun::ddouble expected = hostRecords[i].exp;
      int scaleDiff = calculate_scale_difference(computed, expected);
      INFO("Record " << i << " scale difference: " << scaleDiff << "; computed: " << computed << "; expected: " << expected);
      // We require at least 20 digits of precision.
      REQUIRE( (scaleDiff >= 20 || scaleDiff == 0) );
   }
}


TEST_CASE("ddpower on device using mirror views", "[kokkos][ddouble]") {

   // Structure to hold ddpower test data. 
   // Since ddfun::ddouble is just two contiguous doubles, this struct is tightly packed.
   struct DataRecord {
      ddfun::ddouble a;
      ddfun::ddouble n;
      ddfun::ddouble exp;
   };

   // Read test data from file into a vector of DataRecord.
   std::ifstream infile(INPUT_FILES_DIR + "data/ddpower.bin", std::ios::binary);
   INFO("Reading " << INPUT_FILES_DIR + "data/ddpower.bin");
   REQUIRE(infile.good());
   std::vector<DataRecord> hostRecords;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      hostRecords.push_back(rec);
   }
   int N = hostRecords.size();
   INFO("Read " << N << " records.");
   REQUIRE(N > 0);

   // Create a host-side Kokkos::View from the vector.
   Kokkos::View<ddfun::ddouble*, Kokkos::HostSpace> hA("hA", N);
   Kokkos::View<ddfun::ddouble*, Kokkos::HostSpace> hN("hN", N);
   for (int i = 0; i < N; i++) {
      hA(i) = hostRecords[i].a;
      hN(i) = hostRecords[i].n;
   }

   // Create a device view by deep copying the host view.
   auto dA = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hA);
   auto dN = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hN);

   // Create a device view to hold the computed results.
   Kokkos::View<ddfun::ddouble*, Kokkos::DefaultExecutionSpace> dResults("dResults", N);

   // Launch a kernel to compute ddexp on each input.
   Kokkos::parallel_for("compute_ddlog", N, KOKKOS_LAMBDA(const int i) {
      dResults(i) = dA(i).power(dN(i));
   });
   Kokkos::fence();

   // Create a host mirror of the results.
   auto hResults = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dResults);

   // Validate the results.
   for (int i = 0; i < N; i++) {
      ddfun::ddouble computed = hResults(i);
      ddfun::ddouble expected = hostRecords[i].exp;
      int scaleDiff = calculate_scale_difference(computed, expected);
      INFO("Record " << i << " scale difference: " << scaleDiff << "; computed: " << computed << "; expected: " << expected);
      // We require at least 20 digits of precision.
      REQUIRE( (scaleDiff >= 20 || scaleDiff == 0) );
   }
}


TEST_CASE("ddnpwr on device using mirror views", "[kokkos][ddouble]") {

   // Structure to hold ddnpwr test data. 
   // Since ddfun::ddouble is just two contiguous doubles, this struct is tightly packed.
   struct DataRecord {
      ddfun::ddouble a;
      int n;
      int ph;
      ddfun::ddouble exp;
   };

   // Read test data from file into a vector of DataRecord.
   std::ifstream infile(INPUT_FILES_DIR + "data/ddnpwr.bin", std::ios::binary);
   INFO("Reading " << INPUT_FILES_DIR + "data/ddnpwr.bin");
   REQUIRE(infile.good());
   std::vector<DataRecord> hostRecords;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      hostRecords.push_back(rec);
   }
   int N = hostRecords.size();
   INFO("Read " << N << " records.");
   REQUIRE(N > 0);

   // Create a host-side Kokkos::View from the vector.
   Kokkos::View<ddfun::ddouble*, Kokkos::HostSpace> hA("hA", N);
   Kokkos::View<int*, Kokkos::HostSpace> hN("hN", N);
   for (int i = 0; i < N; i++) {
      hA(i) = hostRecords[i].a;
      hN(i) = hostRecords[i].n;
   }

   // Create a device view by deep copying the host view.
   auto dA = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hA);
   auto dN = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), hN);

   // Create a device view to hold the computed results.
   Kokkos::View<ddfun::ddouble*, Kokkos::DefaultExecutionSpace> dResults("dResults", N);

   // Launch a kernel to compute ddexp on each input.
   Kokkos::parallel_for("compute_ddlog", N, KOKKOS_LAMBDA(const int i) {
      dResults(i) = dA(i).npwr(dN(i));
   });
   Kokkos::fence();

   // Create a host mirror of the results.
   auto hResults = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dResults);

   // Validate the results.
   for (int i = 0; i < N; i++) {
      ddfun::ddouble computed = hResults(i);
      ddfun::ddouble expected = hostRecords[i].exp;
      int scaleDiff = calculate_scale_difference(computed, expected);
      INFO("Record " << i << " scale difference: " << scaleDiff << "; computed: " << computed << "; expected: " << expected);
      // We require at least 20 digits of precision.
      REQUIRE( (scaleDiff >= 20 || scaleDiff == 0) );
   }
}



//////////////////////////////////////////////////////////////////////////
// Example for a single complex operation: ddcadd (complex addition)
//////////////////////////////////////////////////////////////////////////

TEST_CASE("ddcadd on device", "[kokkos][ddcomplex]") {

   // Structure for ddcadd test data: two ddcomplex inputs and one expected output.
   struct DataRecord {
      // Each ddcomplex has two ddfun::ddouble members.
      // For binary layout, assume ddcomplex is stored as: real, imag.
      ddfun::ddcomplex a;
      ddfun::ddcomplex b;
      ddfun::ddcomplex exp;
   };

   std::ifstream infile(INPUT_FILES_DIR + "data/ddcadd.bin", std::ios::binary);
   INFO("Reading " << INPUT_FILES_DIR + "data/ddcadd.bin");
   REQUIRE(infile.good());
   std::vector<DataRecord> hostData;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      hostData.push_back(rec);
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
   Kokkos::parallel_for("compute_ddcadd", N, KOKKOS_LAMBDA(const int i) {
      // Build complex numbers from the record.
      dResults(i) = dA(i) + dB(i);
   });
   Kokkos::fence();

   auto hResults = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dResults);
   for (int i = 0; i < N; i++) {
      ddfun::ddcomplex computed = hResults(i);
      ddfun::ddcomplex expected = hostData[i].exp;
      int scaleDiffReal = calculate_scale_difference(computed.real, expected.real);
      int scaleDiffImag = calculate_scale_difference(computed.imag, expected.imag);
      INFO("ddcadd Record " << i << " scale difference (real): " << scaleDiffReal << "; computed: " << computed.real << "; expected: " << expected.real);
      INFO("ddcadd Record " << i << " scale difference (imag): " << scaleDiffImag << "; computed: " << computed.imag << "; expected: " << expected.imag);
      REQUIRE( (scaleDiffReal >= 20 || scaleDiffReal == 0) );
      REQUIRE( (scaleDiffImag >= 20 || scaleDiffImag == 0) );
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

   std::ifstream infile(INPUT_FILES_DIR + "data/ddcsub.bin", std::ios::binary);
   INFO("Reading " << INPUT_FILES_DIR + "data/ddcsub.bin");
   REQUIRE(infile.good());
   std::vector<DataRecord> hostData;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      hostData.push_back(rec);
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
      ddfun::ddcomplex computed = hResults(i);
      ddfun::ddcomplex expected = hostData[i].exp;
      int scaleDiffReal = calculate_scale_difference(computed.real, expected.real);
      int scaleDiffImag = calculate_scale_difference(computed.imag, expected.imag);
      INFO("ddcsub Record " << i << " scale difference (real): " << scaleDiffReal << "; computed: " << computed.real << "; expected: " << expected.real);
      INFO("ddcsub Record " << i << " scale difference (imag): " << scaleDiffImag << "; computed: " << computed.imag << "; expected: " << expected.imag);
      REQUIRE( (scaleDiffReal >= 20 || scaleDiffReal == 0) );
      REQUIRE( (scaleDiffImag >= 20 || scaleDiffImag == 0) );
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

   std::ifstream infile(INPUT_FILES_DIR + "data/ddcmul.bin", std::ios::binary);
   INFO("Reading " << INPUT_FILES_DIR + "data/ddcmul.bin");
   REQUIRE(infile.good());
   std::vector<DataRecord> hostData;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      hostData.push_back(rec);
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
      int scaleDiffReal = calculate_scale_difference(computed.real, expected.real);
      int scaleDiffImag = calculate_scale_difference(computed.imag, expected.imag);
      INFO("ddcmul Record " << i << " scale difference (real): " << scaleDiffReal << "; computed: " << computed.real << "; expected: " << expected.real);
      INFO("ddcmul Record " << i << " scale difference (imag): " << scaleDiffImag << "; computed: " << computed.imag << "; expected: " << expected.imag);
      REQUIRE( (scaleDiffReal >= 20 || scaleDiffReal == 0) );
      REQUIRE( (scaleDiffImag >= 20 || scaleDiffImag == 0) );
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

   std::ifstream infile(INPUT_FILES_DIR + "data/ddcdiv.bin", std::ios::binary);
   INFO("Reading " << INPUT_FILES_DIR + "data/ddcdiv.bin");
   REQUIRE(infile.good());
   std::vector<DataRecord> hostData;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      hostData.push_back(rec);
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
      int scaleDiffReal = calculate_scale_difference(computed.real, expected.real);
      int scaleDiffImag = calculate_scale_difference(computed.imag, expected.imag);
      INFO("ddcdiv Record " << i << " scale difference (real): " << scaleDiffReal << "; computed: " << computed.real << "; expected: " << expected.real);
      INFO("ddcdiv Record " << i << " scale difference (imag): " << scaleDiffImag << "; computed: " << computed.imag << "; expected: " << expected.imag);
      REQUIRE( (scaleDiffReal >= 20 || scaleDiffReal == 0) );
      REQUIRE( (scaleDiffImag >= 20 || scaleDiffImag == 0) );
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

   std::ifstream infile(INPUT_FILES_DIR + "data/ddcpwr.bin", std::ios::binary);
   INFO("Reading " << INPUT_FILES_DIR + "data/ddcpwr.bin");
   REQUIRE(infile.good());
   std::vector<DataRecord> hostData;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      hostData.push_back(rec);
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
   Kokkos::parallel_for("compute_ddcdiv", N, KOKKOS_LAMBDA(const int i) {
      // Build complex numbers from the record.
      dResults(i) = dA(i).pwr(dN(i));
   });
   Kokkos::fence();

   auto hResults = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dResults);
   for (int i = 0; i < N; i++) {
      ddfun::ddcomplex computed = hResults(i);
      ddfun::ddcomplex expected = hostData[i].exp;
      int scaleDiffReal = calculate_scale_difference(computed.real, expected.real);
      int scaleDiffImag = calculate_scale_difference(computed.imag, expected.imag);
      INFO("ddcdiv Record " << i << " scale difference (real): " << scaleDiffReal << "; computed: " << computed.real << "; expected: " << expected.real);
      INFO("ddcdiv Record " << i << " scale difference (imag): " << scaleDiffImag << "; computed: " << computed.imag << "; expected: " << expected.imag);
      REQUIRE( (scaleDiffReal >= 20 || scaleDiffReal == 0) );
      REQUIRE( (scaleDiffImag >= 20 || scaleDiffImag == 0) );
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
   std::ifstream infile( INPUT_FILES_DIR + "data/ddcsshr.bin", std::ios::binary);
   INFO("Reading " << INPUT_FILES_DIR + "data/ddcsshr.bin");
   REQUIRE(infile.good());
   std::vector<DataRecord> hostData;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      hostData.push_back(rec);
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
      int scaleDiffX = calculate_scale_difference(computedX, expectedX);
      int scaleDiffY = calculate_scale_difference(computedY, expectedY);
      INFO("ddcsshr Record " << i << " scale difference (x): " << scaleDiffX << "; computed: " << computedX << "; expected: " << expectedX);
      INFO("ddcsshr Record " << i << " scale difference (y): " << scaleDiffY << "; computed: " << computedY << "; expected: " << expectedY);
      REQUIRE( (scaleDiffX >= 20 || scaleDiffX == 0) );
      REQUIRE( (scaleDiffY >= 20 || scaleDiffY == 0) );
   }
}


TEST_CASE("ddcssnr on device", "[kokkos][ddouble]") {
   // Structure for ddcssnr test data.
   struct DataRecord {
      ddfun::ddouble a;
      ddfun::ddouble x; // expected output x
      ddfun::ddouble y; // expected output y
   };
   std::ifstream infile( INPUT_FILES_DIR + "data/ddcssnr.bin", std::ios::binary);
   INFO("Reading " << INPUT_FILES_DIR + "data/ddcssnr.bin");
   REQUIRE(infile.good());
   std::vector<DataRecord> hostData;
   DataRecord rec;
   while (infile.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
      hostData.push_back(rec);
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
      int scaleDiffX = calculate_scale_difference(computedX, expectedX);
      int scaleDiffY = calculate_scale_difference(computedY, expectedY);
      INFO("ddcssnr Record " << i << " scale difference (x): " << scaleDiffX << "; computed: " << computedX << "; expected: " << expectedX);
      INFO("ddcssnr Record " << i << " scale difference (y): " << scaleDiffY << "; computed: " << computedY << "; expected: " << expectedY);
      REQUIRE( (scaleDiffX >= 20 || scaleDiffX == 0) );
      REQUIRE( (scaleDiffY >= 20 || scaleDiffY == 0) );
   }
}
