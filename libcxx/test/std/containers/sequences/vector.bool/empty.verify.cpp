//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// class vector

// bool empty() const noexcept;

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <vector>

void f() {
  std::vector<bool> c;
  c.empty(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
