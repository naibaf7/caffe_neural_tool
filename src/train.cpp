/*
 * train.cpp
 *
 *  Created on: Jun 23, 2015
 *      Author: Fabian Tschopp
 */

#include "process.hpp"
#include "train.hpp"
#include "filesystem_utils.hpp"
#include "utils.hpp"
#include "caffe/layers/memory_data_layer.hpp"


namespace caffe_neural {

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

  shared_ptr<caffe::Solver<float> >
        solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

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

  // Overwrite label count from the desired count to the pre-consolidation count
  if(input_param.has_preprocessor()) {
    PreprocessorParam preprocessor_param = input_param.preprocessor();
    if(preprocessor_param.has_label_consolidate()) {
      nr_labels = preprocessor_param.label_consolidate().label_size();
    }
  }

  TrainImageProcessor image_processor(patch_size, nr_labels);

  if(input_param.has_preprocessor()) {

    PreprocessorParam preprocessor_param = input_param.preprocessor();

    image_processor.SetBorderParams(input_param.has_padding_size(), padding_size / 2);
    image_processor.SetRotationParams(preprocessor_param.has_rotation() && preprocessor_param.rotation());
    image_processor.SetPatchMirrorParams(preprocessor_param.has_mirror() && preprocessor_param.mirror());
    image_processor.SetNormalizationParams(preprocessor_param.has_normalization() && preprocessor_param.normalization());

    if(preprocessor_param.has_label_consolidate()) {
      LabelConsolidateParam label_consolidate_param = preprocessor_param.label_consolidate();
      std::vector<int> con_labels;
      for(int cl = 0; cl < label_consolidate_param.label_size(); ++ cl) {
        con_labels.push_back(label_consolidate_param.label(cl));
      }
      image_processor.SetLabelConsolidateParams(preprocessor_param.has_label_consolidate(), con_labels);
    }

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
        std::cout << "Label: " << ((int)k - 1) << ", " << labelcounter[k] << std::endl;
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
    std::vector<int_tp> lalabels;
    lalabels.push_back(0);
    boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
        train_net->layers()[0])->AddMatVector(labels, lalabels);

    // The images
    std::vector<int_tp> imlabels;
    imlabels.push_back(0);
    boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
        train_net->layers()[1])->AddMatVector(images, imlabels);

    solver->Step(1L);

    if (train_param.has_filter_output()) {
      FilterOutputParam filter_param = train_param.filter_output();
      if (filter_param.has_output_filters() && filter_param.output_filters() && filter_param.has_output()) {
        ExportFilters(solver->net().get(), filter_param.output(), bofs::path("train"), 0, 0, 0, true);
      }
    }

    if(test_interval > -1 && i % test_interval == 0) {
      // TODO: Run tests with the testset and testnet
      // TODO: Apply ISBI and other quality measures (cross, rand, pixel, warp, loss)
      // TODO: Write out statistics to file
    }
  }

  LOG(INFO) << "Training done!";

  return 0;
}
}  // namespace caffe_neural
