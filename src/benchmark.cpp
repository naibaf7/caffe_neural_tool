/*
 * benchmark.cpp
 *
 *  Created on: Jun 23, 2015
 *      Author: Fabian Tschopp
 */

#include "benchmark.hpp"
#include <vector>
#include <string>
#include <iostream>
#include <functional>
#include <chrono>
#include <cassert>
#include "filesystem_utils.hpp"
#include "utils.hpp"
#include "caffe/layers/memory_data_layer.hpp"

namespace caffe_neural {


void FillNet(shared_ptr< Layer<float> > data_layer,
             shared_ptr< Layer<float> > label_layer, int num_output) {

  std::function<float()> rfu = GetRandomUniform<float>(-1.0, 1.0);
  std::function<float()> riu = GetRandomUniform(0, num_output - 1);

  if (data_layer != NULL) {
    std::vector<cv::Mat> images;
    std::vector<int_tp> labels;

    shared_ptr<caffe::MemoryDataLayer<float>> data_layer_ptr =
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(data_layer);

    // int bn = data_layer_ptr->batch_size();
    int bc = data_layer_ptr->channels();
    int bh = data_layer_ptr->height();
    int bw = data_layer_ptr->width();

    cv::Mat image(bh, bw, CV_32FC(bc));

#pragma omp parallel for
    for (int h = 0; h < bh; ++h) {
      for (int w = 0; w < bw; ++w) {
        for (int c = 0; c < bc; ++c) {
          *(image.ptr<float>(h, w) + c) = rfu();
        }
      }
    }

    images.push_back(image);
    labels.push_back(0);
    data_layer_ptr->AddMatVector(images, labels);

  }

  if (label_layer != NULL) {
    std::vector<cv::Mat> images;
    std::vector<int_tp> labels;

    shared_ptr<caffe::MemoryDataLayer<float>> layer_ptr =
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(label_layer);

    // int bn = layer_ptr->batch_size();
    int bc = layer_ptr->channels();
    int bh = layer_ptr->height();
    int bw = layer_ptr->width();

    cv::Mat image(bh, bw, CV_32FC(bc));

#pragma omp parallel for
    for (int h = 0; h < bh; ++h) {
      for (int w = 0; w < bw; ++w) {
        for (int c = 0; c < bc; ++c) {
          *(image.ptr<float>(h, w) + c) = riu();
        }
      }
    }

    images.push_back(image);
    labels.push_back(0);
    layer_ptr->AddMatVector(images, labels);
  }
}

int Benchmark(ToolParam &tool_param, CommonSettings &settings) {

  BenchmarkParam benchmark_param = tool_param.benchmark(settings.param_index);
  TrainParam train_param = tool_param.train(
      benchmark_param.has_train_index() ? benchmark_param.train_index() : 0);
  ProcessParam process_param = tool_param.process(
      benchmark_param.has_process_index() ?
          benchmark_param.process_index() : 0);

  int warmup_runs =
      benchmark_param.has_warmup_runs() ? benchmark_param.warmup_runs() : 0;
  int bench_runs =
      benchmark_param.has_bench_runs() ? benchmark_param.bench_runs() : 1;

  if (!benchmark_param.has_output()) {
    LOG(FATAL)<< "Missing output path for benchmarking.";

  }

  double tmp_time = 0;

  // Create output directories
  bofs::path benchpath(benchmark_param.output());
  bofs::create_directories(benchpath);

  std::string proto_solver = train_param.solver();
  std::string process_net = process_param.process_net();

  std::chrono::time_point<std::chrono::high_resolution_clock> t_start, t_end;

  // Benchmark block 1: Training Net
  if (benchmark_param.has_train_index()) {
    caffe::SolverParameter solver_param;
    caffe::ReadProtoFromTextFileOrDie(proto_solver, &solver_param);
    shared_ptr<caffe::Solver<float> >
          solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
    boost::shared_ptr<caffe::Net<float>> net = solver->net();
    net->layers()[0L]->get_device()->ResetPeakMemoryUsage();

    std::vector<double> layer_forward_times(net->layers().size());
    std::vector<double> layer_backward_times(net->layers().size());

    double total_forward_time = 0;
    double total_backward_time = 0;

    for (int run = 0; run < warmup_runs + bench_runs; ++run) {
      FillNet(net->layers()[1], net->layers()[0], train_param.input().labels());

      tmp_time = 0;

      // Benchmark 1: Layer wise measurements (forward)
      for (int_tp l = 0; l < net->layers().size(); ++l) {
        t_start = std::chrono::high_resolution_clock::now();
        net->ForwardFromTo(l, l);
        Caffe::Synchronize(net->layers()[l]->get_device()->list_id());
        t_end = std::chrono::high_resolution_clock::now();
        tmp_time += (t_end - t_start).count();
        if (run >= warmup_runs) {
          layer_forward_times[l] += (t_end - t_start).count();
        }
      }
      LOG(INFO) << "Forward pass: " << std::setprecision(10)
          << (tmp_time)/((double)1e6) << " ms";

      tmp_time = 0;

      // Benchmark 2: Layer wise measurements (backward)
      for (int_tp l = net->layers().size() - 1; l >= 0; --l) {
        t_start = std::chrono::high_resolution_clock::now();
        net->BackwardFromTo(l, l);
        Caffe::Synchronize(net->layers()[l]->get_device()->list_id());
        t_end = std::chrono::high_resolution_clock::now();
        tmp_time += (t_end - t_start).count();
        if (run >= warmup_runs) {
          layer_backward_times[l] += (t_end - t_start).count();
        }
      }
      LOG(INFO) << "Backward pass: " << std::setprecision(10)
          << (tmp_time)/((double)1e6) << " ms";

    }

    for (int run = 0; run < warmup_runs + bench_runs; ++run) {
      FillNet(net->layers()[1], net->layers()[0], train_param.input().labels());

      // Benchmark 3: Whole forward pass
      t_start = std::chrono::high_resolution_clock::now();
      net->ForwardPrefilled();
      Caffe::Synchronize(
          net->layers()[net->layers().size() - 1]->get_device()->list_id());
      t_end = std::chrono::high_resolution_clock::now();
      LOG(INFO) << "Forward pass: " << std::setprecision(10)
          << (double)((t_end - t_start).count())/((double)1e6) << " ms";
      if (run >= warmup_runs) {
        total_forward_time += (t_end - t_start).count();
      }

      // Benchmark 4: Whole backward pass
      t_start = std::chrono::high_resolution_clock::now();
      net->Backward();
      Caffe::Synchronize(net->layers()[0]->get_device()->list_id());
      t_end = std::chrono::high_resolution_clock::now();
      LOG(INFO) << "Backward pass: " << std::setprecision(10)
          << (double)((t_end - t_start).count())/((double)1e6) << " ms";
      if (run >= warmup_runs) {
        total_backward_time += (t_end - t_start).count();
      }
    }

    // Write outputs
    std::vector<std::string> layers = net->layer_names();

    {
      bofs::path filep = benchpath;
      filep /= ("/train_forward_layers_timings.csv");

      std::ofstream out_file;
      out_file.open(filep.string());
      assert(out_file.is_open());
      for (int l = 0; l < net->layers().size(); ++l) {
        out_file << l << ";" << layers[l] << ";"
                 << std::setprecision(10)
                 << layer_forward_times[l] / ((double) bench_runs * 1e6)
                 << ";" << net->layers()[l]->ForwardFlops()
                 << std::endl;
      }
      out_file.close();
    }

    {
      bofs::path filep = benchpath;
      filep /= ("/train_backward_layers_timings.csv");

      std::ofstream out_file;
      out_file.open(filep.string());
      assert(out_file.is_open());
      for (int l = 0; l < net->layers().size(); ++l) {
        out_file << l << ";" << layers[l] << ";"
                 << std::setprecision(10)
                 << layer_backward_times[l] / ((double) bench_runs * 1e6)
                 << ";" << net->layers()[l]->BackwardFlops()
                 << std::endl;
      }
      out_file.close();
    }

    {
      bofs::path filep = benchpath;
      filep /= ("/train_total_timings.csv");

      std::ofstream out_file;
      out_file.open(filep.string());
      assert(out_file.is_open());

      out_file << "Forward;" << std::setprecision(10)
               << total_forward_time / ((double) bench_runs * 1e6)
               << std::endl;

      out_file << "Backward;" << std::setprecision(10)
               << total_backward_time / ((double) bench_runs * 1e6)
               << std::endl;
      out_file.close();
    }

    {
      bofs::path filep = benchpath;
      filep /= ("/train_memory_usage.csv");

      std::ofstream out_file;
      out_file.open(filep.string());
      assert(out_file.is_open());

      out_file << "Peak memory usage;"
          << net->layers()[0]->get_device()->peak_memory_usage()
          << std::endl;
      for(int i = 0; i < net->blobs().size(); ++i) {
        out_file << net->blob_names()[i] << ";" << net->blobs()[i]->count() * sizeof(float) << std::endl;
      }
      out_file.close();
    }

  }

  // Benchmark block 2: Processing Net
  if (benchmark_param.has_process_index()) {
    Net<float> net(process_net, caffe::TEST, Caffe::GetDefaultDevice());
    net.layers()[0]->get_device()->ResetPeakMemoryUsage();

    std::vector<double> layer_forward_times(net.layers().size());
    double total_forward_time = 0;

    // Benchmark 1: Layer wise measurements (forward)
    for (int run = 0; run < warmup_runs + bench_runs; ++run) {
      FillNet(net.layers()[0], NULL, train_param.input().labels());

      tmp_time = 0;
      // Benchmark 1: Layer wise measurements (forward)
      for (int l = 0; l < net.layers().size(); ++l) {
        t_start = std::chrono::high_resolution_clock::now();
        net.ForwardFromTo(l, l);
        Caffe::Synchronize(net.layers()[l]->get_device()->list_id());
        t_end = std::chrono::high_resolution_clock::now();
        tmp_time += (t_end - t_start).count();
        if (run >= warmup_runs) {
          layer_forward_times[l] += (t_end - t_start).count();
        }
      }
      LOG(INFO) << "Forward pass: " << std::setprecision(10)
          << (tmp_time)/((double)1e6) << " ms";
    }

    // Benchmark 2: Whole forward pass
    for (int run = 0; run < warmup_runs + bench_runs; ++run) {
      FillNet(net.layers()[0], NULL, train_param.input().labels());
      t_start = std::chrono::high_resolution_clock::now();
      net.ForwardPrefilled();
      Caffe::Synchronize(
          net.layers()[net.layers().size() - 1]->get_device()->list_id());
      t_end = std::chrono::high_resolution_clock::now();
      LOG(INFO) << "Forward pass: " << std::setprecision(10)
          << (double)((t_end - t_start).count())/((double)1e6) << " ms";
      if (run >= warmup_runs) {
        total_forward_time += (t_end - t_start).count();
      }
    }

    // Write outputs
    std::vector<std::string> layers = net.layer_names();

    {
      bofs::path filep = benchpath;
      filep /= ("/process_forward_layers_timings.csv");

      std::ofstream out_file;
      out_file.open(filep.string());
      assert(out_file.is_open());
      for (int l = 0; l < net.layers().size(); ++l) {
        out_file << l << ";" << layers[l] << ";"
                 << std::setprecision(10)
                 << layer_forward_times[l] / ((double) bench_runs * 1e6)
                 << ";" << net.layers()[l]->ForwardFlops()
                 << std::endl;
      }
      out_file.close();
    }

    {
      bofs::path filep = benchpath;
      filep /= ("/process_total_timings.csv");

      std::ofstream out_file;
      out_file.open(filep.string());
      assert(out_file.is_open());

      out_file << "Forward;" << std::setprecision(10)
               << total_forward_time / ((double) bench_runs * 1e6)
               << std::endl;
      out_file.close();
    }

    {
      bofs::path filep = benchpath;
      filep /= ("/process_memory_usage.csv");

      std::ofstream out_file;
      out_file.open(filep.string());
      assert(out_file.is_open());

      out_file << "Peak memory usage;"
          << net.layers()[0]->get_device()->peak_memory_usage()
          << std::endl;
      for(int i = 0; i < net.blobs().size(); ++i) {
        out_file << net.blob_names()[i] << ";" << net.blobs()[i]->count() * sizeof(float) << std::endl;
      }
      out_file.close();
    }
  }

  return 0;
}

}  // namespace caffe_neural
