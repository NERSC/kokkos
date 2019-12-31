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

/// @Kokkos_Feature_Level_Required:5
// Unit test for MDRangePolicy with Views

namespace Test {

using DataType = double;
const int N    = 10;
const int M    = 10;

template <class ExecSpace>
struct MDFunctor {
  // 2D View
  typedef typename Kokkos::View<DataType **, ExecSpace> View_2D;
  typedef typename View_2D::HostMirror Host_view_2D;

  // 3D View
  typedef typename Kokkos::View<DataType ***, ExecSpace> View_3D;
  typedef typename View_2D::HostMirror Host_view_3D;

  // 4D View
  typedef typename Kokkos::View<DataType ****, ExecSpace> View_4D;
  typedef typename View_2D::HostMirror Host_view_4D;

  View_2D _dataView2D;
  View_3D _dataView3D;
  View_4D _dataView4D;

  MDFunctor(View_2D dataView) : _dataView2D(dataView) {}
  MDFunctor(View_3D dataView) : _dataView3D(dataView) {}
  MDFunctor(View_4D dataView) : _dataView4D(dataView) {}

  // 2D
  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, const int j) const { _dataView2D(i, j) = i * j; }

  // 3D
  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, const int j, const int k) const {
    _dataView3D(i, j, k) = i * j * k;
  }

  // 4D
  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, const int j, const int k, const int l) const {
    _dataView4D(i, j, k, l) = i * j * k * l;
  }
};

template <class ExecSpace>
struct TestMDRangePolicy {
  // 2D View
  typedef typename Kokkos::View<DataType **, ExecSpace> View_2D;
  typedef typename View_2D::HostMirror Host_view_2D;

  // 3D View
  typedef typename Kokkos::View<DataType ***, ExecSpace> View_3D;
  typedef typename View_3D::HostMirror Host_view_3D;

  // 3D View
  typedef typename Kokkos::View<DataType ****, ExecSpace> View_4D;
  typedef typename View_4D::HostMirror Host_view_4D;

  DataType *deviceData, *hostData;

  // memory_space for the memory allocation
  // memory_space for the memory allocation
  typedef typename TEST_EXECSPACE::memory_space MemSpaceD;
  typedef Kokkos::HostSpace MemSpaceH;

  // compare and equal
  void compare_equal_2D(Host_view_2D hostData) {
    int error = 0;
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < M; ++j) ASSERT_EQ(hostData(i, j), i * j);
  }

  // compare and equal
  void compare_equal_3D(Host_view_3D hostData) {
    int error = 0;
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < M; ++j)
        for (int k = 0; k < N; ++k) ASSERT_EQ(hostData(i, j, k), i * j * k);
  }

  // compare and equal
  void compare_equal_4D(Host_view_4D hostData) {
    int error = 0;
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < M; ++j)
        for (int k = 0; k < N; ++k)
          for (int l = 0; l < M; ++l)
            ASSERT_EQ(hostData(i, j, k, l), i * j * k * l);
  }

  // A 2-D MDRangePolicy
  void mdRange2D() {
    // Index Type for the iterator
    typedef Kokkos::IndexType<int> int_index;

    // An MDRangePolicy for 2 nested loops
    typedef typename Kokkos::Experimental::MDRangePolicy<
        ExecSpace, Kokkos::Experimental::Rank<2>, int_index>
        MDPolicyType_2D;
    MDPolicyType_2D mdPolicy_2D({0, 0}, {N, M});

    View_2D deviceDataView("deviceData", N, M);
    Host_view_2D hostDataView = create_mirror_view(deviceDataView);

    // parallel_for call
    MDFunctor<ExecSpace> Functor_2D(deviceDataView);
    Kokkos::parallel_for("MDRange2D", mdPolicy_2D, Functor_2D);

    // Copy the data back to Host memory space
    Kokkos::deep_copy(hostDataView, deviceDataView);

    // Check if all data has been update correctly
    compare_equal_2D(hostDataView);
  }

  // A 3-D MDRangePolicy
  void mdRange3D() {
    // Index Type for the iterator
    typedef Kokkos::IndexType<int> int_index;

    // An MDRangePolicy for 3 nested loops
    typedef typename Kokkos::Experimental::MDRangePolicy<
        ExecSpace, Kokkos::Experimental::Rank<3>, int_index>
        MDPolicyType_3D;
    MDPolicyType_3D mdPolicy_3D({0, 0, 0}, {N, M, N});

    // Allocate Memory for both device and host memory spaces
    // Data[M*N*N]
    View_3D deviceDataView("deviceData", N, M, N);
    Host_view_3D hostDataView = create_mirror_view(deviceDataView);

    // parallel_for call
    MDFunctor<ExecSpace> Functor_3D(deviceDataView);
    Kokkos::parallel_for("MDRange3D", mdPolicy_3D, Functor_3D);

    // Copy the data back to Host memory space
    Kokkos::deep_copy(hostDataView, deviceDataView);

    // Check if all data has been update correctly
    compare_equal_3D(hostDataView);
  }

  // A 4-D MDRangePolicy
  void mdRange4D() {
    // Index Type for the iterator
    typedef Kokkos::IndexType<int> int_index;

    // An MDRangePolicy for 4 nested loops
    typedef typename Kokkos::Experimental::MDRangePolicy<
        ExecSpace, Kokkos::Experimental::Rank<4>, int_index>
        MDPolicyType_4D;
    MDPolicyType_4D mdPolicy_4D({0, 0, 0, 0}, {N, M, N, M});

    // Total number of elements
    size_t num_elements = N * M * N * M;

    // Allocate Memory for both device and host memory spaces
    // Data[M*N*N]
    View_4D deviceDataView("deviceData", N, M, N, M);
    Host_view_4D hostDataView = create_mirror_view(deviceDataView);

    // parallel_for call
    MDFunctor<ExecSpace> Functor_4D(deviceDataView);
    Kokkos::parallel_for("MDRange4D", mdPolicy_4D, Functor_4D);

    // Copy the data back to Host memory space
    Kokkos::deep_copy(hostDataView, deviceDataView);

    // Check if all data has been update correctly
    compare_equal_4D(hostDataView);
  }
};

// 2D MDRangePolicy
TEST(TEST_CATEGORY, incr_05_mdrange2D) {
  TestMDRangePolicy<TEST_EXECSPACE> test;
  test.mdRange2D();
}

// 3D MDRangePolicy
TEST(TEST_CATEGORY, incr_05_mdrange3D) {
  TestMDRangePolicy<TEST_EXECSPACE> test;
  test.mdRange3D();
}

// 4D MDRangePolicy
TEST(TEST_CATEGORY, incr_05_mdrange4D) {
  TestMDRangePolicy<TEST_EXECSPACE> test;
  test.mdRange4D();
}

}  // namespace Test
