/*
 * benchmark.hpp
 *
 *  Created on: Jun 23, 2015
 *      Author: Fabian Tschopp
 */

#ifndef BENCHMARK_HPP_
#define BENCHMARK_HPP_

#include "caffe_neural_tool.hpp"
#include "boost/shared_ptr.hpp"

using boost::shared_ptr;

namespace caffe_neural {

void FillNet(shared_ptr< Layer<float> > data_layer,
             shared_ptr< Layer<float> > label_layer, int num_output);

int Benchmark(ToolParam &tool_param, CommonSettings &settings);
}  // namespace caffe_neural


#endif /* BENCHMARK_HPP_ */
