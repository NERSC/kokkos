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
#include <gtest/gtest.h>

/// @Kokkos_Feature_Level_Required:3
// parallel-for unit test
// create an functor which multiplies the index with a constant double value

namespace Test {

using DataType       = double;
const DataType value = 0.5;
#define UpdateValue i *value

struct ParallelForFunctor {
  DataType *_data;

  ParallelForFunctor(DataType *data) : _data(data) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const { _data[i] = UpdateValue; }
};

template <class ExecSpace>
struct TestParallel_For {
  int num_elements = 10;
  DataType *deviceData, *hostData;

  // memory_space for the memory allocation
  typedef typename TEST_EXECSPACE::memory_space MemSpaceD;
  typedef Kokkos::HostSpace MemSpaceH;

  void compare_equal(DataType *data) {
    for (int i = 0; i < num_elements; ++i) {
      ASSERT_EQ(data[i], UpdateValue);
    }
  }

  // Allocate Memory for both device and host memory spaces
  void alloc_mem() {
    deviceData = (DataType *)Kokkos::kokkos_malloc<MemSpaceD>(
        "dataD", num_elements * sizeof(DataType));
    hostData = (DataType *)Kokkos::kokkos_malloc<MemSpaceH>(
        "dataH", num_elements * sizeof(DataType));
  }

  // Free the allocated memory
  void free_mem() {
    Kokkos::kokkos_free<MemSpaceD>(deviceData);
    Kokkos::kokkos_free<MemSpaceH>(hostData);
  }

  // A simple parallel for test with functors
  void simple_test() {
    alloc_mem();

    // parallel-for functor called for num_elements elements
    Kokkos::parallel_for("parallel_for", num_elements,
                         ParallelForFunctor(deviceData));

    // Copy the data back to Host memory space
    Kokkos::Impl::DeepCopy<MemSpaceD, MemSpaceH>(
        hostData, deviceData, num_elements * sizeof(DataType));

    // Check if all data has been update correctly
    compare_equal(hostData);

    free_mem();
  }

  // A parallel_for test with user defined RangePolicy
  void range_policy() {
    alloc_mem();

    // Create a range policy for the parallel_for
#if defined(KOKKOS_ENABLE_CUDA)
    typedef Kokkos::RangePolicy<ExecSpace, Kokkos::Schedule<Kokkos::Static> >
        range_policy;
#else
    typedef Kokkos::RangePolicy<ExecSpace, Kokkos::Schedule<Kokkos::Dynamic> >
        range_policy;
#endif

    // parallel-for functor called for num_elements elements
    Kokkos::parallel_for("RangePolicy_ParallelFor",
                         range_policy(0, num_elements),
                         ParallelForFunctor(deviceData));

    // Copy the data back to Host memory space
    Kokkos::Impl::DeepCopy<MemSpaceD, MemSpaceH>(
        hostData, deviceData, num_elements * sizeof(DataType));

    // Check if all data has been update correctly
    compare_equal(hostData);

    free_mem();
  }
};

TEST(TEST_CATEGORY, incr_03a_simple_parallelFor) {
  TestParallel_For<TEST_EXECSPACE> test;
  test.simple_test();
}

TEST(TEST_CATEGORY, incr_03a_RangePolicy_parallelFor) {
  TestParallel_For<TEST_EXECSPACE> test;
  test.range_policy();
}

}  // namespace Test
