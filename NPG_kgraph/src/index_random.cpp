//
// Copyright (c) 2017 ZJULearning. All rights reserved.
// Modified by WMZ on 2020/6/16.
// This source code is licensed under the MIT license.
//

#include <efanna2e/index_random.h>


namespace efanna2e {

IndexRandom::IndexRandom(const size_t dimension, const size_t n):Index(dimension, n, L2){
  has_built = true;
}
IndexRandom::~IndexRandom() {}
void IndexRandom::Build(size_t n, const float *data, const Parameters &parameters) {
  data_ = data;
  nd_ = n;

  // Do Nothing

  has_built = true;
}
void IndexRandom::Search(const float *query, const float *x, size_t k, const Parameters &parameters, unsigned *indices) {

    GenRandom(rng, indices, k, nd_);
}

}
