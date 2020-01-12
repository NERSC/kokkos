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

/// @Kokkos_Feature_Level_Required:8
// Unit test for scratch space, team_scratch and thread_scratch

#define updateValue i* j* value

namespace Test {

using DataType       = double;
const int N          = 10;
const int M          = 10;
const DataType value = 0.5;
const int shared_elements = 3;

template <class ExecSpace>
struct TeamThreadHPFunctor {
  // 2D View
  typedef typename Kokkos::View<DataType**, ExecSpace> View_2D;

  // Team policy and member type for kokkos
  typedef typename Kokkos::TeamPolicy<ExecSpace> team_policy;
  typedef typename team_policy::member_type team_member;

  typedef typename ExecSpace::scratch_memory_space shared_space;
  typedef typename Kokkos::View<DataType*, shared_space> View_Shared_1D;

  View_2D _dataView2D;

  TeamThreadHPFunctor(View_2D dataView) : _dataView2D(dataView) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const team_member& thread) const {
    const int i = thread.league_rank();

    //Allocate shared array between threads in a team
    //Create a shared array of the size of 1st dimension
    View_Shared_1D shared_array(thread.team_shmem(),_dataView2D.extent(1));
    Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, M), [&](const int& j) {
      _dataView2D(i, j) = updateValue;
    });

    Kokkos::single(Kokkos::PerTeam(thread),[=]()
    {
      shared_array(i) = thread.team_size() * i * value;
    });
  }
};

template <class ExecSpace>
struct TestScratchSpace {
  typedef typename Kokkos::View<DataType**, ExecSpace> View_2D;
  typedef typename View_2D::HostMirror Host_View_2D;

  typedef typename Kokkos::TeamPolicy<ExecSpace> team_policy;
  typedef typename team_policy::member_type team_member;

  // compare and equal
  void compare_equal(Host_View_2D hostData) {
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < M; ++j) {
        ASSERT_EQ(hostData(i, j), updateValue);
      }
  }

  void testit() {
    View_2D deviceDataView("deviceData", N, M);
    Host_View_2D hostDataView = create_mirror_view(deviceDataView);
    team_policy policy_2D1(N, Kokkos::AUTO());

    TeamThreadHPFunctor<ExecSpace> func(deviceDataView);
    Kokkos::parallel_for(policy_2D1, func);

    // Copy the data back to Host memory space
    Kokkos::deep_copy(hostDataView, deviceDataView);

    // Compare and equal for correctness
    compare_equal(hostDataView);
  }
};

TEST(TEST_CATEGORY, incr_08a_ScratchSpace) {
  TestScratchSpace<TEST_EXECSPACE> test;
  test.testit();
}

}  // namespace Test
