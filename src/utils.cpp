/*
 * neural_utils.cpp
 *
 *  Created on: Apr 3, 2015
 *      Author: Fabian Tschopp
 */

#include <sys/time.h>
#include <utils.hpp>
#include <random>
#include <sstream>

namespace caffe_neural {

std::string ZeroPadNumber(int number, int total_size) {
  std::stringstream ss;

  ss << number;
  std::string ret;
  ss >> ret;

  int str_length = ret.length();
  for (int i = 0; i < total_size - str_length; ++i) {
    ret = "0" + ret;
  }

  return ret;
}

std::function<unsigned int()> GetRandomSelector(unsigned int set_size) {
  struct timeval start_time;
  gettimeofday(&start_time, NULL);
  std::seed_seq seq { start_time.tv_sec, start_time.tv_usec };
  std::mt19937_64 generator(seq);
  std::uniform_int_distribution<unsigned int> distribution(0, set_size - 1);
  std::function<unsigned int()> selector = std::bind(distribution, generator);
  return selector;
}

std::function<unsigned int()> GetRandomOffset(unsigned int min, unsigned int max) {
  struct timeval start_time;
  gettimeofday(&start_time, NULL);
  std::seed_seq seq { start_time.tv_sec, start_time.tv_usec };
  std::mt19937_64 generator(seq);
  std::uniform_int_distribution<unsigned int> distribution(min, max);
  std::function<unsigned int()> offset = std::bind(distribution, generator);
  return offset;
}

template<typename Dtype>
std::function<Dtype()> GetRandomUniform(Dtype min, Dtype max) {
  struct timeval start_time;
  gettimeofday(&start_time, NULL);
  std::seed_seq seq { start_time.tv_sec, start_time.tv_usec };
  std::mt19937_64 generator(seq);
  std::uniform_real_distribution<Dtype> distribution(min, max);
  std::function<Dtype()> uniform = std::bind(distribution, generator);
  return uniform;
}

template std::function<float()> GetRandomUniform(float min, float max);
template std::function<double()> GetRandomUniform(double min, double max);

std::function<int()> GetRandomUniform(int min, int max) {
  struct timeval start_time;
  gettimeofday(&start_time, NULL);
  std::seed_seq seq { start_time.tv_sec, start_time.tv_usec };
  std::mt19937_64 generator(seq);
  std::uniform_int_distribution<int> distribution(min, max);
  std::function<int()> uniform = std::bind(distribution, generator);
  return uniform;
}

template<typename Dtype>
std::function<Dtype()> GetRandomNormal(Dtype mu, Dtype std) {
  struct timeval start_time;
  gettimeofday(&start_time, NULL);
  std::seed_seq seq { start_time.tv_sec, start_time.tv_usec };
  std::mt19937_64 generator(seq);
  std::normal_distribution<Dtype> distribution(mu, std);
  std::function<Dtype()> normal = std::bind(distribution, generator);
  return normal;
}

template std::function<float()> GetRandomNormal(float mu, float std);
template std::function<double()> GetRandomNormal(double mu, double std);


}
