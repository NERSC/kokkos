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

/// @Kokkos_Feature_Level_Required:1

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <sstream>
#include <type_traits>
#include <gtest/gtest.h>

using DataType = double;

namespace Test {

// Unit test for Atomic commands

template <class T>
struct TestIncrAtomic {
  T old_value = 0.5, new_value = 1.5;
  void testExchange() {
    T ret_value = Kokkos::atomic_exchange(&old_value, new_value);

    ASSERT_EQ(old_value, 1.5);
    ASSERT_EQ(ret_value, 0.5);
  }

  void testAdd() {
    T ret_value = Kokkos::atomic_exchange(&old_value, new_value);
    Kokkos::atomic_add(&old_value, ret_value);

    ASSERT_EQ(old_value, 2.0);
  }

  void testSub() {
    T ret_value = Kokkos::atomic_exchange(&old_value, new_value);
    Kokkos::atomic_sub(&old_value, ret_value);

    ASSERT_EQ(old_value, 1.0);
  }
};

TEST(TEST_CATEGORY, incr_01_AtomicExchange) {
  TestIncrAtomic<DataType> test;
  test.testExchange();
}

TEST(TEST_CATEGORY, incr_01_AtomicAdd) {
  TestIncrAtomic<DataType> test;
  test.testAdd();
}

TEST(TEST_CATEGORY, incr_01_AtomicSub) {
  TestIncrAtomic<DataType> test;
  test.testSub();
}

}  // namespace Test
