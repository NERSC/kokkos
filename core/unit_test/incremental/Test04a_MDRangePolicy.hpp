/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <sstream>
#include <type_traits>
#include <gtest/gtest.h>

/// @Kokkos_Feature_Level_Required:4

namespace Test {

using dataType = double;
const int N    = 10;
const int M    = 10;

// Unit Test for Reduction

struct MDFunctor {
  const double value = 0.5;
  dataType *_data;

  MDFunctor(dataType *data) : _data(data) {}

  // 2D
  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, const int j) const { _data[i * M + j] = i * j; }

  // 3D
  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, const int j, const int k) const {
    _data[i * M * N + j * M + k] = i * j * k;
  }

  // 4D
  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, const int j, const int k, const int l) const {
    _data[i * M * N * M + j * M * N + k * M + l] = i * j * k * l;
  }
};

template <class ExecSpace>
struct TestMDRangePolicy {
  dataType *deviceData, *hostData;

  // memory_space for the memory allocation
  // memory_space for the memory allocation
  typedef typename TEST_EXECSPACE::memory_space MemSpaceD;
  typedef Kokkos::HostSpace MemSpaceH;

  // compare and equal
  int compare_equal_2D(double sum) {
    int error = 0;
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < M; ++j)
        if (hostData[i * M + j] != i * j) ++error;

    return error;
  }

  // compare and equal
  int compare_equal_3D(double sum) {
    int error = 0;
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < M; ++j)
        for (int k = 0; k < N; ++k)
          if (hostData[i * M * N + j * M + k] != i * j * k) ++error;

    return error;
  }

  // compare and equal
  int compare_equal_4D(double sum) {
    int error = 0;
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < M; ++j)
        for (int k = 0; k < N; ++k)
          for (int l = 0; l < M; ++l)
            if (hostData[i * M * N * M + j * M * N + k * M + l] !=
                i * j * k * l)
              ++error;

    return error;
  }

  // A 2-D MDRangePolicy
  void mdRange2D() {
    double sum = 0.0;

    // iterator type
    typedef Kokkos::IndexType<int> int_index;

    typedef typename Kokkos::Experimental::MDRangePolicy<
        ExecSpace, Kokkos::Experimental::Rank<2>, int_index>
        MDPolicyType_2D;
    MDPolicyType_2D mdPolicy_2D({0, 0}, {N, M});

    // Total number of elements
    size_t num_elements = N * M;

    // Allocate Memory for both device and host memory spaces
    // Data[M*N]
    deviceData = (dataType *)Kokkos::kokkos_malloc<MemSpaceD>(
        "dataD", num_elements * sizeof(dataType));
    hostData = (dataType *)Kokkos::kokkos_malloc<MemSpaceH>(
        "dataH", num_elements * sizeof(dataType));

    // parallel_reduce call
    MDFunctor Functor_2D(deviceData);
    Kokkos::parallel_for("MDRange2D", mdPolicy_2D, Functor_2D);

    // Copy the data back to Host memory space
    Kokkos::Impl::DeepCopy<MemSpaceD, MemSpaceH>(
        hostData, deviceData, num_elements * sizeof(dataType));

    // Check if all data has been update correctly
    int sumError = compare_equal_2D(sum);
    ASSERT_EQ(sumError, 0);

    // Free the allocated memory
    Kokkos::kokkos_free<MemSpaceD>(deviceData);
    Kokkos::kokkos_free<MemSpaceH>(hostData);
  }

  // A 3-D MDRangePolicy
  void mdRange3D() {
    double sum = 0.0;

    typedef Kokkos::IndexType<int> int_index;

    typedef typename Kokkos::Experimental::MDRangePolicy<
        ExecSpace, Kokkos::Experimental::Rank<3>, int_index>
        MDPolicyType_3D;
    MDPolicyType_3D mdPolicy_3D({0, 0, 0}, {N, M, N});

    // Total number of elements
    size_t num_elements = N * M * N;

    // Allocate Memory for both device and host memory spaces
    // Data[M*N*N]
    deviceData = (dataType *)Kokkos::kokkos_malloc<MemSpaceD>(
        "dataD", num_elements * sizeof(dataType));
    hostData = (dataType *)Kokkos::kokkos_malloc<MemSpaceH>(
        "dataH", num_elements * sizeof(dataType));

    // parallel_reduce call
    MDFunctor Functor_3D(deviceData);
    Kokkos::parallel_for("MDRange3D", mdPolicy_3D, Functor_3D);

    // Copy the data back to Host memory space
    Kokkos::Impl::DeepCopy<MemSpaceD, MemSpaceH>(
        hostData, deviceData, num_elements * sizeof(dataType));

    // Check if all data has been update correctly
    int sumError = compare_equal_3D(sum);
    ASSERT_EQ(sumError, 0);

    // Free the allocated memory
    Kokkos::kokkos_free<MemSpaceD>(deviceData);
    Kokkos::kokkos_free<MemSpaceH>(hostData);
  }

  // A 4-D MDRangePolicy
  void mdRange4D() {
    double sum = 0.0;

    typedef Kokkos::IndexType<int> int_index;

    typedef typename Kokkos::Experimental::MDRangePolicy<
        ExecSpace, Kokkos::Experimental::Rank<4>, int_index>
        MDPolicyType_4D;
    MDPolicyType_4D mdPolicy_4D({0, 0, 0, 0}, {N, M, N, M});

    // Total number of elements
    size_t num_elements = N * M * N * M;

    // Allocate Memory for both device and host memory spaces
    // Data[M*N*N]
    deviceData = (dataType *)Kokkos::kokkos_malloc<MemSpaceD>(
        "dataD", num_elements * sizeof(dataType));
    hostData = (dataType *)Kokkos::kokkos_malloc<MemSpaceH>(
        "dataH", num_elements * sizeof(dataType));

    // parallel_reduce call
    MDFunctor Functor_4D(deviceData);
    Kokkos::parallel_for("MDRange4D", mdPolicy_4D, Functor_4D);

    // Copy the data back to Host memory space
    Kokkos::Impl::DeepCopy<MemSpaceD, MemSpaceH>(
        hostData, deviceData, num_elements * sizeof(dataType));

    // Check if all data has been update correctly
    int sumError = compare_equal_4D(sum);
    ASSERT_EQ(sumError, 0);

    // Free the allocated memory
    Kokkos::kokkos_free<MemSpaceD>(deviceData);
    Kokkos::kokkos_free<MemSpaceH>(hostData);
  }
};

// 2D MDRangePolicy
TEST(TEST_CATEGORY, incr_04_mdrange2D) {
  TestMDRangePolicy<TEST_EXECSPACE> test;
  test.mdRange2D();
}

// 3D MDRangePolicy
TEST(TEST_CATEGORY, incr_04_mdrange3D) {
  TestMDRangePolicy<TEST_EXECSPACE> test;
  test.mdRange3D();
}

// 4D MDRangePolicy
TEST(TEST_CATEGORY, incr_04_mdrange4D) {
  TestMDRangePolicy<TEST_EXECSPACE> test;
  test.mdRange4D();
}

}  // namespace Test
