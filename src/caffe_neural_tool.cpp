/*
 * caffe_neural_tool.cpp
 *
 *  Created on: Feb 26, 2015
 *      Author: Fabian Tschopp
 */

#include <cuda_runtime.h>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <sstream>
#include <string>
#include <iostream>
#include <map>
#include <stdio.h>
#include <omp.h>
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include "caffe/caffe.hpp"
#include "caffe/common.hpp"
#include "caffe/solver.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <sys/stat.h>
#include <sys/types.h>
#include <boost/program_options.hpp>
#include "caffe_neural_tool.hpp"

#include "image_processor.hpp"
#include "neural_utils.hpp"
#include "tiffio_wrapper.hpp"
#include "filesystem_utils.hpp"

#include <glog/logging.h>
#include "google/protobuf/message.h"
#include "caffetool.pb.h"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;
using caffe::Datum;
using caffe::Solver;
using caffe::NetParameter;

namespace bopo = boost::program_options;
namespace gpb = google::protobuf;

using namespace caffe_neural;

#define OCVDBGW "OpenCV Debug Window"

int Train(ToolParam &tool_param, CommonSettings &settings) {

  if (tool_param.train_size() <= settings.param_index) {
    LOG(FATAL)<< "Train parameter index does not exist.";
  }

  TrainParam train_param = tool_param.train(settings.param_index);
  InputParam input_param = train_param.input();

  if(!(input_param.has_patch_size() && input_param.has_padding_size() && input_param.has_labels() && input_param.has_channels())) {
    LOG(FATAL) << "Patch size, padding size, label count or channel count parameter missing.";
  }
  int patch_size = input_param.patch_size();
  int padding_size = input_param.padding_size();
  unsigned int nr_labels = input_param.labels();
  unsigned int nr_channels = input_param.channels();

  std::string proto_solver = "";
  if(!train_param.has_solver()) {
    LOG(FATAL) << "Solver prototxt file argument missing";
  }

  proto_solver = train_param.solver();

  caffe::SolverParameter solver_param;
  caffe::ReadProtoFromTextFileOrDie(proto_solver, &solver_param);

  int test_interval = solver_param.has_test_interval()?solver_param.test_interval():-1;

  shared_ptr<caffe::Solver<float> > solver(
      caffe::GetSolver<float>(solver_param));

  if(train_param.has_solverstate()) {
    // Continue from previous solverstate
    const char* solver_state_c = train_param.solverstate().c_str();
    solver->Restore(solver_state_c);
  }

  // Get handles to the test and train network of the Caffe solver
  boost::shared_ptr<caffe::Net<float>> train_net = solver->net();
  boost::shared_ptr<caffe::Net<float>> test_net;
  if(solver->test_nets().size() > 0) {
    test_net = solver->test_nets()[0];
  }

  TrainImageProcessor image_processor(patch_size, nr_labels);

  if(input_param.has_preprocessor()) {

    PreprocessorParam preprocessor_param = input_param.preprocessor();

    image_processor.SetBorderParams(input_param.has_padding_size(), padding_size / 2);
    image_processor.SetRotationParams(preprocessor_param.has_rotation() && preprocessor_param.rotation());
    image_processor.SetPatchMirrorParams(preprocessor_param.has_mirror() && preprocessor_param.mirror());
    image_processor.SetNormalizationParams(preprocessor_param.has_normalization() && preprocessor_param.normalization());

    if(preprocessor_param.has_histeq()) {
      PrepHistEqParam histeq_param = preprocessor_param.histeq();
      std::vector<float> label_boost(nr_labels, 1.0);
      for(int i = 0; i < histeq_param.label_boost().size(); ++i) {
        label_boost[i] = histeq_param.label_boost().Get(i);
      }
      image_processor.SetLabelHistEqParams(true, histeq_param.has_patch_prior()&&histeq_param.patch_prior(), histeq_param.has_masking()&&histeq_param.masking(), label_boost);
    }

    if(preprocessor_param.has_crop()) {
      PrepCropParam crop_param = preprocessor_param.crop();
      image_processor.SetCropParams(crop_param.has_imagecrop()?crop_param.imagecrop():0, crop_param.has_labelcrop()?crop_param.labelcrop():0);
    }

    if(preprocessor_param.has_clahe()) {
      PrepClaheParam clahe_param = preprocessor_param.clahe();
      image_processor.SetClaheParams(true, clahe_param.has_clip()?clahe_param.clip():4.0);
    }

    if(preprocessor_param.has_blur()) {
      PrepBlurParam blur_param = preprocessor_param.blur();
      image_processor.SetBlurParams(true, blur_param.has_mean()?blur_param.mean():0.0, blur_param.has_std()?blur_param.std():0.1, blur_param.has_ksize()?blur_param.ksize():5);
    }

  }

  if(!(input_param.has_raw_images() && input_param.has_label_images())) {
    LOG(FATAL) << "Raw images or label images folder missing.";
  }

  std::set<std::string> filetypes = CreateImageTypesSet();

  int error;
  std::vector<std::vector<bofs::path>> training_set = LoadTrainingSetItems(filetypes, input_param.raw_images(),input_param.label_images(),&error);

  unsigned int ijsum = 0;
  // Preload and preprocess all images
  for (unsigned int i = 0; i < training_set.size(); ++i) {
    std::vector<bofs::path> training_item = training_set[i];

    std::vector<cv::Mat> raw_stack;
    std::vector<std::vector<cv::Mat>> labels_stack(training_item.size() - 1);

    std::string type = bofs::extension(training_item[0]);
    std::transform(type.begin(), type.end(), type.begin(), ::tolower);

    if(type == ".tif" || type == ".tiff") {
      // TIFF and multipage TIFF mode
      raw_stack = LoadTiff(training_item[0].string(), nr_channels);
    } else {
      // All other image types
      cv::Mat raw_image = cv::imread(training_item[0].string(), nr_channels == 1 ? CV_LOAD_IMAGE_GRAYSCALE :
          CV_LOAD_IMAGE_COLOR);
      raw_stack.push_back(raw_image);
    }

    for(unsigned int k = 0; k < training_item.size() - 1; ++k) {
      std::string type = bofs::extension(training_item[k+1]);
      std::transform(type.begin(), type.end(), type.begin(), ::tolower);
      if(type == ".tif" || type == ".tiff") {
        std::vector<cv::Mat> label_stack = LoadTiff(training_item[k+1].string(), 1);
        labels_stack[k] = label_stack;
      }
      else {
        std::vector<cv::Mat> label_stack;
        cv::Mat label_image = cv::imread(training_item[k+1].string(), CV_LOAD_IMAGE_GRAYSCALE);
        label_stack.push_back(label_image);
        labels_stack[k] = label_stack;
      }
    }

    for (unsigned int j = 0; j < raw_stack.size(); ++j) {
      std::vector<cv::Mat> label_images;
      for(unsigned int k = 0; k < labels_stack.size(); ++k) {
        label_images.push_back(labels_stack[k][j]);
      }

      if(label_images.size() > 1 && nr_labels != 2 && label_images.size() < nr_labels) {
        // Generate complement label
        cv::Mat clabel(label_images[0].rows, label_images[0].cols, CV_8UC(1), 255.0);
        for(unsigned int k = 0; k < label_images.size(); ++k) {
          cv::subtract(clabel,label_images[k],clabel);
        }
        label_images.push_back(clabel);
      }

      image_processor.SubmitImage(raw_stack[j], ijsum, label_images);
      ++ijsum;
    }
  }

  image_processor.Init();

  std::vector<long> labelcounter(nr_labels + 1);

  int train_iters = solver_param.has_max_iter()?solver_param.max_iter():0;

  // Do the training
  for (int i = 0; i < train_iters; ++i) {
    std::vector<cv::Mat> patch = image_processor.DrawPatchRandom();

    std::vector<cv::Mat> images;
    std::vector<cv::Mat> labels;

    images.push_back(patch[0]);
    labels.push_back(patch[1]);

    // TODO: Only enable in debug or statistics mode
    for (int y = 0; y < patch_size; ++y) {
      for (int x = 0; x < patch_size; ++x) {
        labelcounter[patch[1].at<float>(y, x) + 1] += 1;
      }
    }

    if(settings.debug) {
      for (unsigned int k = 0; k < nr_labels + 1; ++k) {
        std::cout << "Label: " << k << ", " << labelcounter[k] << std::endl;
      }
    }

    if(settings.graphic) {

      cv::Mat test;

      double minVal, maxVal;
      cv::minMaxLoc(patch[1], &minVal, &maxVal);
      patch[1].convertTo(test, CV_32FC1, 1.0 / (maxVal - minVal),
          -minVal * 1.0 / (maxVal - minVal));

      std::vector<cv::Mat> tv;
      tv.push_back(test);
      tv.push_back(test);
      tv.push_back(test);
      cv::Mat tvl;

      cv::merge(tv, tvl);

      cv::Mat patchclone = patch[0].clone();

      tvl.copyTo(
          patchclone(
              cv::Rect(padding_size / 2, padding_size / 2, patch_size,
                  patch_size)));

      cv::imshow(OCVDBGW, patchclone);
      cv::waitKey(10);
    }

    // The labels
    std::vector<int> lalabels;
    lalabels.push_back(0);
    boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
        train_net->layers()[0])->AddMatVector(labels, lalabels);

    // The images
    std::vector<int> imlabels;
    imlabels.push_back(0);
    boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
        train_net->layers()[1])->AddMatVector(images, imlabels);

    solver->Step(1);

    if(test_interval > -1 && i % test_interval == 0) {
      // TODO: Run tests with the testset and testnet
      // TODO: Apply ISBI and other quality measures (cross, rand, pixel, warp, loss)
      // TODO: Write out statistics to file
    }
  }

  LOG(INFO) << "Training done!";

  return 0;
}

int Process(caffe_neural::ToolParam &tool_param, CommonSettings &settings) {

  if (tool_param.train_size() <= settings.param_index) {
    LOG(FATAL)<< "Train parameter index does not exist.";
  }

  ProcessParam process_param = tool_param.process(settings.param_index);
  InputParam input_param = process_param.input();
  OutputParam output_param = process_param.output();

  if(!output_param.has_output()) {
    LOG(FATAL) << "Processing output path missing.";
  }

  std::string outpath = output_param.output();
  std::string format = output_param.has_format()? output_param.format() : ".tif";
  bool fp32out = output_param.has_fp32_out()? output_param.fp32_out() : false;
  if(fp32out) {
    format = ".tif";
  }
  std::set<std::string> filetypes = CreateImageTypesSet();
  if (filetypes.find(format) == filetypes.end()) {
    format = ".tif";
  }
  std::transform(format.begin(), format.end(), format.begin(), ::tolower);

  if(!(input_param.has_patch_size() && input_param.has_padding_size() && input_param.has_labels() && input_param.has_channels())) {
    LOG(FATAL) << "Patch size, padding size, label count or channel count parameter missing.";
  }
  int patch_size = input_param.patch_size();
  int padding_size = input_param.padding_size();
  unsigned int nr_labels = input_param.labels();
  unsigned int nr_channels = input_param.channels();

  if(!(process_param.has_process_net() && process_param.has_caffemodel())) {
    LOG(FATAL) << "Processing network prototxt or caffemodel argument missing.";
  }

  std::string process_net = process_param.process_net();
  std::string caffe_model = process_param.caffemodel();

  Net<float> net(process_net, caffe::TEST);
  net.CopyTrainedLayersFrom(caffe_model);

  ProcessImageProcessor image_processor(patch_size, nr_labels);

  int imagecrop = 0;
  if(input_param.has_preprocessor()) {

    PreprocessorParam preprocessor_param = input_param.preprocessor();

    image_processor.SetBorderParams(input_param.has_padding_size(), padding_size / 2);
    image_processor.SetRotationParams(preprocessor_param.has_rotation() && preprocessor_param.rotation());
    image_processor.SetPatchMirrorParams(preprocessor_param.has_mirror() && preprocessor_param.mirror());
    image_processor.SetNormalizationParams(preprocessor_param.has_normalization() && preprocessor_param.normalization());

    if(preprocessor_param.has_histeq()) {
      PrepHistEqParam histeq_param = preprocessor_param.histeq();
      std::vector<float> label_boost(nr_labels, 1.0);
      for(int i = 0; i < histeq_param.label_boost().size(); ++i) {
        label_boost[i] = histeq_param.label_boost().Get(i);
      }
      image_processor.SetLabelHistEqParams(true, histeq_param.has_patch_prior()&&histeq_param.patch_prior(), histeq_param.has_masking()&&histeq_param.masking(), label_boost);
    }

    if(preprocessor_param.has_crop()) {
      PrepCropParam crop_param = preprocessor_param.crop();
      image_processor.SetCropParams(crop_param.has_imagecrop()?crop_param.imagecrop():0, crop_param.has_labelcrop()?crop_param.labelcrop():0);
      imagecrop = crop_param.has_imagecrop()?crop_param.imagecrop():0;
    }

    if(preprocessor_param.has_clahe()) {
      PrepClaheParam clahe_param = preprocessor_param.clahe();
      image_processor.SetClaheParams(true, clahe_param.has_clip()?clahe_param.clip():4.0);
    }

    if(preprocessor_param.has_blur()) {
      PrepBlurParam blur_param = preprocessor_param.blur();
      image_processor.SetBlurParams(true, blur_param.has_mean()?blur_param.mean():0.0, blur_param.has_std()?blur_param.std():0.1, blur_param.has_ksize()?blur_param.ksize():5);
    }
  }

  int error;
  std::vector<bofs::path> process_set = LoadProcessSetItems(filetypes, input_param.raw_images(),&error);

  for (unsigned int i = 0; i < process_set.size(); ++i) {
    LOG(INFO) << "Processing file: " << process_set[i];

    std::vector<cv::Mat> image_stack;

    std::string type = bofs::extension(process_set[0]);
    std::transform(type.begin(), type.end(), type.begin(), ::tolower);

    if(type == ".tif" || type == ".tiff") {
      // TIFF and multipage TIFF mode
      image_stack = LoadTiff(process_set[i].string(),
          nr_channels);
    } else {
      // All other image types
      cv::Mat image = cv::imread(process_set[i].string(),
          nr_channels == 1 ? CV_LOAD_IMAGE_GRAYSCALE:CV_LOAD_IMAGE_COLOR);
      image_stack.push_back(image);
    }

    std::vector<std::vector<cv::Mat>> output_stack;
    for (unsigned int st = 0; st < image_stack.size(); ++st) {
      image_processor.ClearImages();

      LOG(INFO) << "Processing subdirectory: " << st;

      cv::Mat image = image_stack[st];

      std::vector<cv::Mat> labels;
      image_processor.SubmitImage(image, i, labels);

      cv::Mat padimage = image_processor.raw_images()[0];

      int image_size_x = image.cols;
      int image_size_y = image.rows;

      std::vector<cv::Mat> outimgs;
      for(unsigned int k = 0; k < nr_labels; ++k) {
        cv::Mat outimg(image_size_y, image_size_x, CV_32FC1);
        outimgs.push_back(outimg);
      }

      for (int yoff = 0; yoff < (image_size_y - 1) / patch_size + 1; ++yoff) {
        for (int xoff = 0; xoff < (image_size_x - 1) / patch_size + 1; ++xoff) {

          int xoffp = xoff * patch_size;
          int yoffp = yoff * patch_size;

          if(xoffp + patch_size > image_size_x) {
            xoffp = image_size_x - patch_size;
          }

          if(yoffp + patch_size > image_size_y) {
            yoffp = image_size_y - patch_size;
          }

          cv::Rect roi(xoffp, yoffp,
              padding_size + patch_size - imagecrop,
              padding_size + patch_size - imagecrop);
          cv::Mat crop = padimage(roi);

          std::vector<cv::Mat> images;
          std::vector<int> labels;

          images.push_back(crop);
          labels.push_back(0);

          boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
              (&net)->layers()[0])->AddMatVector(images, labels);

          float loss = 0.0;
          const vector<Blob<float>*>& result = net.ForwardPrefilled(&loss);

          const float* cpuresult = result[0]->cpu_data();

#pragma omp parallel for
          for (unsigned int k = 0; k < nr_labels; ++k) {
            for (int y = 0; y < patch_size; ++y) {
              for (int x = 0; x < patch_size; ++x) {
                (outimgs[k].at<float>(y + yoffp,
                        x + xoffp)) =
                cpuresult[(k * patch_size + y) * patch_size + x];
              }
            }
          }

          if (settings.graphic) {
            for (unsigned int k = 0; k < nr_labels; ++k) {
              cv::imshow(OCVDBGW, outimgs[k]);
              cv::waitKey(100);
            }
          }

        }
      }
      output_stack.push_back(outimgs);
    }

    // Output stacks only supported with multipage TIFFs
    if(output_stack.size() > 1) {
      format = ".tif";
    }

    bofs::path outp(outpath);
    bofs::create_directories(outp);

    unsigned int nr_out_labels = ((output_param.has_out_all_labels() && output_param.out_all_labels()) || nr_labels > 2)?nr_labels:1;

    // In the two label case, export the second and not the first label output
    unsigned int label_offset = nr_out_labels==1?1:0;

    for(unsigned int k = 0; k < nr_out_labels; ++k) {
      bofs::path outpl = outp;
      if(nr_out_labels > 1) {
        outpl /= ("/" + ZeroPadNumber(k,std::log10(nr_labels)+1));
        bofs::create_directories(outpl);
      }
      bofs::path filep = outpl;
      filep /= ("/" + process_set[i].stem().string()+format);

      std::vector<cv::Mat> saveout(output_stack.size());
      for(unsigned int st = 0; st < output_stack.size();++st) {
        if(fp32out) {
          output_stack[st][k+label_offset].convertTo(saveout[st], CV_32FC1, 1.0, 0.0);
        } else {
          output_stack[st][k+label_offset].convertTo(saveout[st], CV_8UC1, 255.0, 0.0);
        }
      }

      if(format == ".tif" || format == ".tiff") {
        SaveTiff(saveout,filep.string());
      } else {
        cv::imwrite(filep.string(),saveout[0]);
      }
    }
  }
  return 0;
}

int main(int argc, const char** argv) {

  google::InitGoogleLogging(argv[0]);

  int device_id;
  int thread_count;
  std::string proto;
  int train_index;
  int process_index;

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
    Caffe::SetDevice(device_id);
  }

  if (varmap.count("proto")) {

    ToolParam tool_param;
    caffe::ReadProtoFromTextFileOrDie(proto, &tool_param);

    CommonSettings settings;
    settings.graphic = varmap.count("graphic");
    settings.debug = varmap.count("debug");

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
