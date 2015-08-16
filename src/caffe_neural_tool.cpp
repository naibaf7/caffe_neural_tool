/*
 * caffe_neural_tool.cpp
 *
 *  Created on: Feb 26, 2015
 *      Author: Fabian Tschopp
 */

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif // USE_CUDA
#include <cstring>
#include <cstdlib>
#include <vector>
#include <sstream>
#include <string>
#include <iostream>
#include <map>
#include <stdio.h>
#include <omp.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <boost/program_options.hpp>
#include "caffe_neural_tool.hpp"
#include "train.hpp"
#include "process.hpp"
#include "benchmark.hpp"

namespace bopo = boost::program_options;
namespace gpb = google::protobuf;

using namespace caffe_neural;

int main(int argc, const char** argv) {

  google::InitGoogleLogging(argv[0]);

  int device_id;
  int thread_count;
  std::string proto;
  int train_index;
  int process_index;
  int benchmark_index;

  bopo::options_description desc("Allowed options");
  desc.add_options()      //
  ("help", "help message")      //
  ("devices", "show all available GPU devices")      //
  ("gpu", bopo::value<int>(&device_id)->default_value(0), "set GPU to use")  //
  ("cpu", "use fallback CPU backend")  //
  ("debug", "enable debugging messages")  //
  ("graphic", "graphical debug output")  //
  ("ompthreads",
   bopo::value<int>(&thread_count)->default_value(omp_get_num_procs()),
   "number of OpenMP threads to use)")  //
  ("proto", bopo::value<std::string>(&proto), "configuration prototxt file")  //
  ("train", bopo::value<int>(&train_index),
   "training mode with training parameter set")  //
  ("process", bopo::value<int>(&process_index),
   "process mode with process parameter set")  //
  ("silent", "silence all logging")  //
  ("benchmark", bopo::value<int>(&benchmark_index), "start a benchmarking run")  //
   ;

  bopo::variables_map varmap;
  bopo::store(bopo::parse_command_line(argc, argv, desc), varmap);
  bopo::notify(varmap);

  omp_set_num_threads(thread_count);

  if (varmap.count("silent")) {
    FLAGS_logtostderr = 0;
  } else {
    FLAGS_logtostderr = 1;
  }

  if (varmap.count("help")) {
    std::cout << desc << std::endl;
    return 1;
  }

  if (varmap.count("devices")) {
    Caffe::EnumerateDevices();
    return 1;
  }

  if (varmap.count("graphic")) {
    cv::namedWindow(OCVDBGW, cv::WINDOW_AUTOSIZE);
  }

  if (varmap.count("cpu")) {
    Caffe::set_mode(Caffe::CPU);
  } else {
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevices(std::vector<int>{device_id});
    Caffe::SetDevice(device_id);
  }

  if (varmap.count("proto")) {

    ToolParam tool_param;
    caffe::ReadProtoFromTextFileOrDie(proto, &tool_param);

    CommonSettings settings;
    settings.graphic = varmap.count("graphic");
    settings.debug = varmap.count("debug");

    if (varmap.count("benchmark")) {
      LOG(INFO)<< "Benchmarking mode.";
      settings.param_index = benchmark_index;
      Benchmark(tool_param, settings);
    }

    if (varmap.count("train")) {
      LOG(INFO)<< "Training mode.";
      settings.param_index = train_index;
      Train(tool_param, settings);
    }

    if (varmap.count("process")) {
      LOG(INFO)<< "Processing mode.";
      settings.param_index = process_index;
      Process(tool_param, settings);
    }

  } else {
    LOG(FATAL)<< "Missing prototxt argument.";
  }

  return 0;
}
