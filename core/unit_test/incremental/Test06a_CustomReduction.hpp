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
// Unit test for parallel_reduce with custom reductions
// Define a complex structure and a customized reduction for updating the real and imaginary parts of the complex number

namespace Test {

using DataType = double;
const int num_elements    = 10;
const DataType value = 0.5;

struct complex_num
{
  DataType _re,_im;

  complex_num():_re(0.0),_im(0.0){}
  complex_num(DataType re, DataType im):_re(re),_im(im){}

  void operator += (const complex_num update)
  {
    _re += update._re;
    _im += update._im;
  }
};

template <class ExecSpace>
struct myComplexArray
{
  typedef typename Kokkos::View<complex_num*, ExecSpace> complex_View;
  typedef typename complex_View::HostMirror host_complex_View;

  //An array of complex numbers (complex number is represented by the above structure called complex_num
  complex_View array;
  //Host mirror view of the complex array
  host_complex_View host_array;
  //Number of elements in the array, passed in the constructor
  int _num_elements;

  myComplexArray(const int num_elements)
    :_num_elements(num_elements)
  {

    array = complex_View("complex_array",_num_elements);
    host_array = create_mirror_view(array);
  }

  //Initialize all the elements of the array with the product of the element index and a const value passed
  KOKKOS_INLINE_FUNCTION
  void init(const double val)
  {
    for(int i = 0; i < _num_elements; ++i)
    {
      host_array(i)._re = i*val;
      host_array(i)._im = -i*val;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void copyToDevice()
  {
    Kokkos::deep_copy(array, host_array);
  }

  //reduction operator for sum
  KOKKOS_INLINE_FUNCTION
  void operator += (myComplexArray &otherArray) {}


  // return a complex number that has the reduced value of all the elements in the complex array
  complex_num reduce()
  {

    complex_num result;
    for(int i = 0; i < _num_elements; ++i)
      result += host_array(i);

    return result;
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const int i, complex_num &updateSum) const
  {
    updateSum += array(i);
  }

};

template <class ExecSpace>
struct TestCustomReduce {

  void reduce() {
    myComplexArray<ExecSpace> complexArray(num_elements);
    complexArray.init(value);
    complexArray.copyToDevice();

//    complex_num result;
//    Kokkos::parallel_reduce("complex_reduce",num_elements,complexArray, result);
//    Kokkos::parallel_reduce("complex_reduce",num_elements, KOKKOS_LAMBDA(const int i, complex_num &Update)
//        {
//      Update += complexArray.host_array(i);
//      },result);

    complex_num result = complexArray.reduce();

    int sum = 0;
    for(int i = 0; i < num_elements; ++i)
      sum += i;

    ASSERT_EQ(sum*value,result._re);
    ASSERT_EQ(sum*-value,result._im);
  }
};

TEST(TEST_CATEGORY, incr_06_customReduce) {
  TestCustomReduce<TEST_EXECSPACE> test;
  test.reduce();
}

}  // namespace Test
