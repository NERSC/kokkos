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

/// @Kokkos_Feature_Level_Required:6
// Unit test for Custom Reduction

namespace Test {

using DataType = double;
const int N    = 10;
const int M    = 10;

struct compl_num
{
  DataType _re, _im;

  compl_num(DataType re, DataType im)
    :_re(re), _im(im)
  {}

  compl_num() = default;

  KOKKOS_INLINE_FUNCTION
  void operator +=(compl_num &update)
  {
    _re += update._re;
    _im += update._im;
  }
};

template <class ExecSpace>
struct MyComplexArray
{
  typedef typename Kokkos::View<compl_num *, ExecSpace> Complex_View;
  typedef typename Complex_View::HostMirror Complex_Host_View;

  Complex_View _myComplexArray;

  MyComplexArray() = default;

  MyComplexArray(Complex_View complexArray)
    :_myComplexArray(complexArray)
  {}

  KOKKOS_INLINE_FUNCTION
  compl_num&  operator() (const int i) const
  {
    return _myComplexArray(i);
  }

  KOKKOS_INLINE_FUNCTION
  MyComplexArray& operator += (const MyComplexArray& src)
  {
    for(int i = 0; i < N; ++i)
    {
      _myComplexArray(i) += src;
    }
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  void operator += (const volatile compl_num src) volatile
  {
    for(int i = 0; i < N; ++i)
    {
      _myComplexArray(i) += src;
    }
  }
};

template <class ExecSpace>
struct SumMyComplexArray
{
  MyComplexArray<ExecSpace> _complexArray;

  SumMyComplexArray(MyComplexArray<ExecSpace> complexData)
    :_complexArray(complexData)
  {}

  SumMyComplexArray() = default;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const
  {
    _complexArray(i) = compl_num(0.5*i,-0.5*i);
  }
};

template <class ExecSpace>
struct ReduceMyComplexArray
{
  MyComplexArray<ExecSpace> _complexArray;

  ReduceMyComplexArray(MyComplexArray<ExecSpace> complexData)
    :_complexArray(complexData)
  {}

  ReduceMyComplexArray() = default;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, compl_num &Update) const
  {
    Update += _complexArray(i);
  }

//  KOKKOS_INLINE_FUNCTION
//  void join(compl_num& dest, const compl_num& src)  const {
//    dest += src;
//  }

};

template <class ExecSpace>
struct TestCustomReduction {
  typedef typename Kokkos::View<compl_num *, ExecSpace> Complex_View;
  typedef typename Complex_View::HostMirror Complex_Host_View;

  void parallel_for_correctness(Complex_Host_View hostData)
  {
    for(int i = 0; i < N; ++i)
    {
      ASSERT_EQ(hostData(i)._re , 0.5*i);
      ASSERT_EQ(hostData(i)._im , -0.5*i);
    }
  }

  void customParallelFor() {
    Complex_View complexArray("complexArray", N);
    Complex_Host_View host_complexArray = create_mirror_view(complexArray);

    //Parallel_for over complex_num
    SumMyComplexArray<ExecSpace> sumArrayFunctor(complexArray);
    Kokkos::parallel_for("Complex Parallel For", N, sumArrayFunctor);
    Kokkos::deep_copy(host_complexArray, complexArray);
    parallel_for_correctness(host_complexArray);

    //Parallel_reduce over complex_num
    compl_num sumNumber(0.0,0.0);
    ReduceMyComplexArray<ExecSpace> reduceArrayFunctor(complexArray);
    Kokkos::parallel_reduce("Complex Parallel reduce",N,reduceArrayFunctor,sumNumber);
  }

};

// Custom Reduction test
TEST(TEST_CATEGORY, incr_06_customReduction) {
  TestCustomReduction<TEST_EXECSPACE> test;
  test.customParallelFor();
}

}  // namespace Test
