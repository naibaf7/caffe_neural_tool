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

#include "image_processor.hpp"
#include "neural_utils.hpp"
#include "tiffio_wrapper.hpp"

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

using namespace caffe_neural;

// For the dataset_03:

#define MODE_3DTIFF true
#define TESTING

#define PROTO_SOLVER "../project_data/net_sk_2out/neuraltissue_solver.prototxt"
 #define PROTO_NET "../project_data/net_sk_2out/neuraltissue_process.prototxt"
 #define MODEL_WEIGHTS "../project_data/net_sk_2out/neuraltissue_iter_62000.caffemodel"
 #define SOLVER_STATE "../project_data/net_sk_2out/neuraltissue_iter_62000.solverstate"

 #define INPUT_IMAGES "../project_data/dataset_03/input/"
 #define INPUT_PREFIX "test-volume"
 #define INPUT_START_INDEX 0
 #define INPUT_COUNT 1
 #define INPUT_DIGITS 2
 #define INPUT_FORMAT ".tif"
 #define OUTPUT_LABELS "../project_data/dataset_03/output/"
 #define OUTPUT_FORMAT ".tif"

 #define TRAIN_SET_SIZE 1
 #define TRAIN_FOLDER "../project_data/dataset_03/train/"
 #define TRAIN_LABEL_PREFIX "labels/train-labels"
 #define TRAIN_RAW_PREFIX "raw/train-volume"
 #define TRAIN_LABEL_DIGITS 2
 #define TRAIN_RAW_DIGITS 2
 #define TRAIN_LABEL_FORMAT ".tif"
 #define TRAIN_RAW_FORMAT ".tif"

 #define BATCH_SIZE 1
 #define NR_CHANNELS 3
 #define NR_LABELS 2
 #define PATCH_SIZE_TRAIN 64
 #define PATCH_SIZE_PROCESS 128
 #define PADDING_SIZE 102
 #define TRAIN_IMAGE_SIZE 512
 #define PROCESS_IMAGE_SIZE 512

// For the dataset_01:
/*#define PROTO_SOLVER "../project_data/net_sk_9out/neuraltissue_solver.prototxt"
 #define PROTO_NET "../project_data/net_sk_9out/neuraltissue_process.prototxt"
 #define MODEL_WEIGHTS "../project_data/net_sk_9out/neuraltissue_iter_10000.caffemodel"
 #define SOLVER_STATE "../project_data/net_sk_9out/neuraltissue_iter_10000.solverstate"

 #define INPUT_IMAGES "../project_data/dataset_01/input/"
 #define INPUT_PREFIX ""
 #define INPUT_START_INDEX 0
 #define INPUT_COUNT 20
 #define INPUT_DIGITS 2
 #define INPUT_FORMAT ".tif"
 #define OUTPUT_LABELS "../project_data/dataset_01/output/"
 #define OUTPUT_FORMAT ".png"

 #define TRAIN_SET_SIZE 20
 #define TRAIN_FOLDER "../project_data/dataset_01/train/"
 #define TRAIN_LABEL_PREFIX "labels/labels"
 #define TRAIN_RAW_PREFIX "raw/"
 #define TRAIN_LABEL_DIGITS 8
 #define TRAIN_RAW_DIGITS 2
 #define TRAIN_LABEL_FORMAT ".png"
 #define TRAIN_RAW_FORMAT ".tif"

 #define BATCH_SIZE 1
 #define NR_CHANNELS 3
 #define NR_LABELS 9
 #define PATCH_SIZE_TRAIN 64
 #define PATCH_SIZE_PROCESS 128
 #define PADDING_SIZE 102
 #define TRAIN_IMAGE_SIZE 1024
 #define PROCESS_IMAGE_SIZE 1024*/

// For the dataset_02:
/*#define PROTO_SOLVER "../project_data/net_sk_3out/neuraltissue_solver.prototxt"
#define PROTO_NET "../project_data/net_sk_3out/neuraltissue_process.prototxt"
#define MODEL_WEIGHTS "../project_data/net_sk_3out/neuraltissue_iter_10000.caffemodel"
#define SOLVER_STATE "../project_data/net_sk_3out/neuraltissue_iter_10000.solverstate"

#define INPUT_IMAGES "../project_data/dataset_02/input/"
#define INPUT_PREFIX "crop."
#define INPUT_START_INDEX 4352
#define INPUT_COUNT 200
#define INPUT_DIGITS 8
#define INPUT_FORMAT ".png"
#define OUTPUT_LABELS "../project_data/dataset_02/output_3/"
#define OUTPUT_FORMAT ".png"

#define TRAIN_SET_SIZE 20
#define TRAIN_FOLDER "../project_data/dataset_02/train/"
#define TRAIN_LABEL_PREFIX "labels/"
#define TRAIN_RAW_PREFIX "raw/raw_"
#define TRAIN_LABEL_DIGITS 3
#define TRAIN_RAW_DIGITS 3
#define TRAIN_LABEL_FORMAT ".tif"
#define TRAIN_RAW_FORMAT ".tif"

#define BATCH_SIZE 1
#define NR_CHANNELS 3
#define NR_LABELS 3
#define PATCH_SIZE_TRAIN 64
#define PATCH_SIZE_PROCESS 128
#define PADDING_SIZE 102
#define TRAIN_IMAGE_SIZE 512
#define PROCESS_IMAGE_SIZE 3072*/

#define NUM_THREADS 8

#define OCVDBGW "OpenCV Debug Window"

int Train() {
  std::string proto_solver = PROTO_SOLVER;
  std::string train_folder = TRAIN_FOLDER;
  std::string train_raw_prefix = TRAIN_RAW_PREFIX;
  std::string train_raw_format = TRAIN_RAW_FORMAT;
  std::string train_label_prefix = TRAIN_LABEL_PREFIX;
  std::string train_label_format = TRAIN_LABEL_FORMAT;
  std::string solver_state = SOLVER_STATE;

  int train_set_size = TRAIN_SET_SIZE;

  int train_raw_digits = TRAIN_RAW_DIGITS;
  int train_label_digits = TRAIN_LABEL_DIGITS;

  int patch_size = PATCH_SIZE_TRAIN;
  int padding_size = PADDING_SIZE;
  int image_size = TRAIN_IMAGE_SIZE;
  int nr_labels = NR_LABELS;

  caffe::SolverParameter solver_param;
  caffe::ReadProtoFromTextFileOrDie(proto_solver, &solver_param);

  shared_ptr<caffe::Solver<float> > solver(
      caffe::GetSolver<float>(solver_param));

  // Continue from previous solverstate
  const char* solver_state_c = solver_state.c_str();
  solver->Restore(solver_state_c);

  boost::shared_ptr<caffe::Net<float>> train_net = solver->net();

  TrainImageProcessor image_processor(image_size, patch_size, nr_labels);

  image_processor.SetBorderParams(true, padding_size / 2);
  image_processor.SetRotationParams(true);
  image_processor.SetPatchMirrorParams(true);

  std::vector<float> label_boost { 1, 1, 1, 1, 1, 1, 1, 1, 1 };

  //image_processor.SetLabelHistEqParams(true, true, true, label_boost);
  image_processor.SetCropParams(1, 0);
  image_processor.SetClaheParams(true, 4.0);
  image_processor.SetBlurParams(true, 0, 0.1, 5);
  image_processor.SetNormalizationParams(true);

  int ijsum = 0;
  // Preload and preprocess all images
  for (int i = 0; i < train_set_size; ++i) {
    if (MODE_3DTIFF) {

      std::vector<cv::Mat> raw_stack = LoadTiff(
          train_folder + train_raw_prefix + ZeroPadNumber(i, train_raw_digits)
              + train_raw_format,
          3);

      std::vector<cv::Mat> label_stack = LoadTiff(
          train_folder + train_label_prefix
              + ZeroPadNumber(i, train_label_digits) + train_label_format,
          1);

      for (unsigned int j = 0; j < raw_stack.size(); ++j) {
        std::vector<cv::Mat> label_images;
        label_images.push_back(label_stack[j]);
        image_processor.SubmitImage(raw_stack[j], ijsum, label_images, 255.0);
        ++ijsum;
      }

    } else {
      cv::Mat raw_image = cv::imread(
          train_folder + train_raw_prefix + ZeroPadNumber(i, train_raw_digits)
              + train_raw_format,
          CV_LOAD_IMAGE_COLOR);

      /*cv::Mat label_image = cv::imread(
          train_folder + train_label_prefix
              + ZeroPadNumber(i, train_label_digits) + train_label_format,
          CV_LOAD_IMAGE_GRAYSCALE);

      std::vector<cv::Mat> label_images;
      label_images.push_back(label_image);*/

      cv::Mat label_image_1 = cv::imread(
       train_folder + train_label_prefix + "label_01/mitochondria_"
       + ZeroPadNumber(i, train_label_digits) + train_label_format,
       CV_LOAD_IMAGE_GRAYSCALE);

       cv::Mat label_image_2 = cv::imread(
       train_folder + train_label_prefix + "label_02/membrane_"
       + ZeroPadNumber(i, train_label_digits) + train_label_format,
       CV_LOAD_IMAGE_GRAYSCALE);

       cv::Mat label_image_3(label_image_1.cols, label_image_2.cols, CV_8UC(1),
       255.0);
       cv::subtract(label_image_3, label_image_1, label_image_3);
       cv::subtract(label_image_3, label_image_2, label_image_3);
       cv::subtract(label_image_2, label_image_1, label_image_2);

       //cv::imshow(OCVDBGW, raw_image);
       //cv::waitKey(0);
       //cv::imshow(OCVDBGW, label_image_2);
       //cv::waitKey(0);
       //cv::imshow(OCVDBGW, label_image_3);
       //cv::waitKey(0);

       std::vector<cv::Mat> label_images;
       label_images.push_back(label_image_1);
       label_images.push_back(label_image_2);
       label_images.push_back(label_image_3);

      image_processor.SubmitImage(raw_image, i, label_images, 0.0);
    }
  }

  image_processor.Init();

#ifdef TESTING
  std::vector<long> labelcounter(nr_labels + 1);
#endif

  // Do the training
  for (int i = 0; i < 30000; ++i) {
    std::vector<cv::Mat> patch = image_processor.DrawPatchRandom();

    std::vector<cv::Mat> images;
    std::vector<cv::Mat> labels;

    images.push_back(patch[0]);
    labels.push_back(patch[1]);

#ifdef TESTING
    for (int y = 0; y < patch_size; ++y) {
      for (int x = 0; x < patch_size; ++x) {
        labelcounter[patch[1].at<float>(y, x) + 1] += 1;
      }
    }

    for (int k = 0; k < nr_labels + 1; ++k) {
      std::cout << "Label: " << k << ", " << labelcounter[k] << std::endl;
    }

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

    tvl.copyTo(
        patch[0](
            cv::Rect(padding_size / 2, padding_size / 2, patch_size,
                     patch_size)));

    cv::imshow(OCVDBGW, patch[0]);
    cv::waitKey(0);
#endif

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

    solver->StepPrefilled();

  }

  solver->Snapshot();

  std::cout << "DONE!" << std::endl;

  return 0;
}

int Process() {

  std::string proto_net = PROTO_NET;
  std::string input_images = INPUT_IMAGES;
  std::string model_weights = MODEL_WEIGHTS;

  int input_start_index = INPUT_START_INDEX;
  int input_count = INPUT_COUNT;

  int nr_labels = NR_LABELS;

  std::string input_prefix = INPUT_PREFIX;
  std::string input_format = INPUT_FORMAT;
  std::string output_labels = OUTPUT_LABELS;
  std::string output_format = OUTPUT_FORMAT;

  int input_digits = INPUT_DIGITS;
  int padding_size = PADDING_SIZE;
  int patch_size = PATCH_SIZE_PROCESS;
  int image_size = PROCESS_IMAGE_SIZE;

  Net<float> net(proto_net, caffe::TEST);
  net.CopyTrainedLayersFrom(MODEL_WEIGHTS);

  ProcessImageProcessor image_processor(image_size, patch_size, nr_labels);
  image_processor.SetBorderParams(true, padding_size / 2);
  image_processor.SetClaheParams(true, 4.0);
  image_processor.SetCropParams(1, 0);
  image_processor.SetNormalizationParams(true);

  for (int i = input_start_index; i < input_start_index + input_count; ++i) {
    std::cout
        << input_images + input_prefix + ZeroPadNumber(i, input_digits)
            + input_format
        << std::endl;

    std::vector<cv::Mat> image_stack;

    if (MODE_3DTIFF) {
      image_stack = LoadTiff(
          input_images + input_prefix + ZeroPadNumber(i, input_digits)
              + input_format,
          3);
    } else {

      cv::Mat image = cv::imread(
          input_images + input_prefix + ZeroPadNumber(i, input_digits)
              + input_format,
          CV_LOAD_IMAGE_COLOR);

      image_stack.push_back(image);
    }

    std::vector<cv::Mat> output_stack;
    for (unsigned int st = 0; st < image_stack.size(); ++st) {
      image_processor.ClearImages();

      std::cout << "Subdirectory: " << st << std::endl;

      cv::Mat image = image_stack[st];

      std::vector<cv::Mat> labels;
      image_processor.SubmitImage(image, i, labels, 0.0);

      cv::Mat padimage = image_processor.raw_images()[0];

      cv::Mat outimg(image_size, image_size, CV_32FC(nr_labels));

      for (int yoff = 0; yoff < image_size / patch_size; ++yoff) {
        for (int xoff = 0; xoff < image_size / patch_size; ++xoff) {

          cv::Rect roi(xoff * patch_size, yoff * patch_size,
                       padding_size + patch_size - 1,
                       padding_size + patch_size - 1);
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
          for (int y = 0; y < patch_size; ++y) {
            for (int x = 0; x < patch_size; ++x) {
              for (int k = 0; k < nr_labels; ++k) {
                (outimg.at<cv::Vec<float, NR_LABELS>>(y + yoff * patch_size,
                                                      x + xoff * patch_size))[k] =
                    cpuresult[(k * patch_size + y) * patch_size + x];
              }
            }
          }

#ifdef TESTING
          std::vector<cv::Mat> channels(nr_labels);
          cv::split(outimg, channels);

          cv::imshow(OCVDBGW, channels[1]);
          cv::waitKey(1);
#endif

        }
      }

      std::vector<cv::Mat> channels(nr_labels);
      cv::split(outimg, channels);

      if (!MODE_3DTIFF) {
        std::string dir = output_labels + ZeroPadNumber(i, input_digits) + "/";
        const char* cdir = dir.c_str();
        mkdir(cdir, 0777);

        for (int c = 0; c < nr_labels; ++c) {

          cv::Mat out;

          channels[c].convertTo(out, CV_8UC(1), 255.0, 0.0);

          cv::imwrite(
              output_labels + ZeroPadNumber(i, input_digits) + "/"
                  + std::to_string(c) + output_format,
              out);

        }
      } else {
        cv::Mat out;
        channels[1].convertTo(out, CV_8UC(1), 255.0, 0.0);
        output_stack.push_back(out);
      }

    }
    if (MODE_3DTIFF) {
      SaveTiff(output_stack,
               output_labels + ZeroPadNumber(i, input_digits) + output_format);
    }
  }

  return 0;
}

int DummyTest() {

  Net<float> net("net/dummy_process.prototxt", caffe::TEST);

  int padding_size = PADDING_SIZE;
  int patch_size = PATCH_SIZE_PROCESS;

  std::string input_images = INPUT_IMAGES;

  cv::Mat image = cv::imread(input_images + "/00.tif", CV_LOAD_IMAGE_COLOR);
  cv::Mat padimage;
  cv::copyMakeBorder(image, padimage, padding_size / 2, padding_size / 2,
                     padding_size / 2, padding_size / 2, IPL_BORDER_REFLECT,
                     cv::Scalar::all(0.0));

  cv::Rect roi(0, 0, padding_size + patch_size - 1,
               padding_size + patch_size - 1);
  cv::Mat crop = padimage(roi);

  cv::Mat outimg(padding_size + patch_size - 1, padding_size + patch_size - 1,
                 CV_32FC(3));

  std::vector<cv::Mat> images;
  std::vector<int> labels;

  // Testing
  for (int y = 0; y < padding_size + patch_size - 1; ++y) {
    for (int x = 0; x < padding_size + patch_size - 1; ++x) {
      int k = y / ((padding_size + patch_size) / 3);
      (crop.at<cv::Vec<unsigned char, 3>>(y, x))[k] = 255;
      (crop.at<cv::Vec<unsigned char, 3>>(y, x))[(k + 1) % 3] = 200;
      (crop.at<cv::Vec<unsigned char, 3>>(y, x))[(k + 2) % 3] = 200;
    }
  }

  cv::imshow(OCVDBGW, crop);
  cv::waitKey();

  images.push_back(crop);
  labels.push_back(0);

  boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
      (&net)->layers()[0])->AddMatVector(images, labels);

  float loss = 0.0;
  const vector<Blob<float>*>& result = net.ForwardPrefilled(&loss);

  const float* cpuresult = result[1]->cpu_data();

  std::cout << "Result size: " << result.size() << std::endl;

#pragma omp parallel for
  for (int y = 0; y < padding_size + patch_size - 1; ++y) {
    for (int x = 0; x < padding_size + patch_size - 1; ++x) {
      for (int k = 0; k < 3; ++k) {
        (outimg.at<cv::Vec<float, 3>>(y, x))[k] = cpuresult[(k
            * (padding_size + patch_size - 1) + y)
            * (padding_size + patch_size - 1) + x];
      }
    }
  }

  std::vector<cv::Mat> channels(3);

  cv::split(outimg, channels);
  std::cout << "DONE" << std::endl;

  for (int i = 0; i < 1; ++i) {

  }

  return 0;
}

int main(int argc, const char** argv) {

  omp_set_num_threads(NUM_THREADS);

  cv::namedWindow(OCVDBGW, cv::WINDOW_AUTOSIZE);

  Caffe::set_mode(Caffe::GPU);

  Caffe::EnumerateDevices();

  std::cout << "Select a device: " << std::endl;

  int device_id = 0;

  std::cin >> device_id;

  Caffe::SetDevice(device_id);

  //DummyTest();

#ifndef TESTING
  Train();
  //Process();
#else
  //Train();
  Process();
#endif
}
