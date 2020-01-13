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

/// @Kokkos_Feature_Level_Required:9
// Unit test for hierarchial parallelism with ThreadTeam and TeamVector lanes

namespace Test {

  using DataType = double;
  const int N    = 10;
  const int M    = 10;
  const DataType value = 0.5;

#define updateValue i*j*k*value

template <class ExecSpace>
struct HPFunctor_TeamVector {
  // 2D View
  typedef typename Kokkos::View<DataType***, ExecSpace> View_3D;
  typedef typename View_3D::HostMirror Host_view_3D;

  //Team policy and member type for kokkos
  typedef typename Kokkos::TeamPolicy<ExecSpace> team_policy;
  typedef typename team_policy::member_type team_member;

  View_3D _dataView3D;

  HPFunctor_TeamVector(View_3D dataView) : _dataView3D(dataView) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const team_member& team) const
  {
    const int i = team.league_rank();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,M), [&](const int& j)
    {
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,N), [&](const int& k)
      {
        _dataView3D(i,j,k) = updateValue;
      });
    });
  }
};

template <class ExecSpace>
struct TestHierarchialParallelism_TeamVector {

    typedef typename Kokkos::View<DataType***, ExecSpace> View_3D;
    typedef typename View_3D::HostMirror Host_View_3D;

    typedef typename Kokkos::TeamPolicy<ExecSpace> team_policy;
    typedef typename team_policy::member_type team_member;

  // compare and equal
  void compare_equal(Host_View_3D hostData) {
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < M; ++j)
        for(int k = 0; k < N; ++k)
          ASSERT_EQ(hostData(i, j, k), updateValue);

  }

  void test_HP_TeamVector() {

    View_3D deviceDataView("deviceData",N,M,N);
    Host_View_3D hostDataView = create_mirror_view(deviceDataView);
    team_policy policy_2D(N,Kokkos::AUTO() );

    HPFunctor_TeamVector<ExecSpace> func(deviceDataView);
    Kokkos::parallel_for(policy_2D,func);

    // Copy the data back to Host memory space
    Kokkos::deep_copy(hostDataView, deviceDataView);

    //Compare and equal for correctness
    compare_equal(hostDataView);
  }
};

TEST(TEST_CATEGORY, incr_09a_hierarchialParallelism) {
  TestHierarchialParallelism_TeamVector<TEST_EXECSPACE> test;
  test.test_HP_TeamVector();
}

}  // namespace Test
