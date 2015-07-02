/*
 * neural_utils.hpp
 *
 *  Created on: Mar 25, 2015
 *      Author: Fabian Tschopp
 */

#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <functional>
#include <string>

namespace caffe_neural {

std::string ZeroPadNumber(int number, int total_size);

std::function<unsigned int()> GetRandomSelector(unsigned int set_size);

std::function<unsigned int()> GetRandomOffset(unsigned int min,
                                              unsigned int max);

template<typename Dtype>
std::function<Dtype()> GetRandomUniform(Dtype min, Dtype max);

template<typename Dtype>
std::function<Dtype()> GetRandomNormal(Dtype mu, Dtype std);

std::function<int()> GetRandomUniform(int min, int max);

}

#endif /* UTILS_HPP_ */
