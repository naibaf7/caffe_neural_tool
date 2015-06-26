/*
 * filesystem_utils.hpp
 *
 *  Created on: May 14, 2015
 *      Author: Fabian Tschopp
 */

#ifndef FILESYSTEM_UTILS_HPP_
#define FILESYSTEM_UTILS_HPP_

#define FSU_ERR_EXCEPTION 1
#define FSU_ERR_NO_FOLDER 2
//#define FSU_ERR_

#include <vector>
#include <string>
#include <set>
#include "boost/filesystem.hpp"

namespace bofs = boost::filesystem;

namespace caffe_neural {

std::vector<std::vector<bofs::path>> LoadTrainingSetItems(std::set<std::string> filetypes, std::string raw_path, std::string label_path, int* error);
std::vector<bofs::path> LoadProcessSetItems(std::set<std::string> filetypes, std::string raw_path, int* error);

std::set<std::string> CreateImageTypesSet();

std::function<unsigned int()> GetRandomSelector(unsigned int set_size);
std::function<unsigned int()> GetRandomOffset(unsigned int min, unsigned int max);
template<typename Dtype>
std::function<Dtype()> GetRandomUniform(Dtype min, Dtype max);
std::function<int()> GetRandomUniform(int min, int max);
template<typename Dtype>
std::function<Dtype()> GetRandomNormal(Dtype mu, Dtype std);



}

#endif /* FILESYSTEM_UTILS_HPP_ */
