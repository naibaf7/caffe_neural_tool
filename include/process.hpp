/*
 * process.hpp
 *
 *  Created on: Jun 23, 2015
 *      Author: Fabian Tschopp
 */

#ifndef PROCESS_HPP_
#define PROCESS_HPP_

#include "caffe_neural_tool.hpp"
#include "filesystem_utils.hpp"

namespace caffe_neural {

int Process(caffe_neural::ToolParam &tool_param, CommonSettings &settings);

int ExportFilters(Net<float> *net, std::string output_folder,
                  bofs::path input_name, int st, int y, int x, bool store_diff);

}

#endif /* PROCESS_HPP_ */
