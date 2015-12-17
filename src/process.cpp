/*
 * process.cpp
 *
 *  Created on: Jun 23, 2015
 *      Author: Fabian Tschopp
 */

#include "process.hpp"
#include "filesystem_utils.hpp"
#include "utils.hpp"
#include "caffe/layers/memory_data_layer.hpp"

namespace caffe_neural {

int ExportFilters(Net<float> *net, std::string output_folder,
                  bofs::path input_name, int st, int y, int x, bool store_diff) {
  std::vector<std::string> names = net->blob_names();
  std::vector<boost::shared_ptr<Blob<float>>>blobs = net->blobs();

  for (unsigned int i = 0; i < names.size(); ++i) {
    std::string name = names[i];
    shared_ptr<Blob<float>> blob = blobs[i];

    const float* cpu_ptr = blob->cpu_data();
    const float* cpu_diff_ptr = nullptr;
    if (store_diff) {
      cpu_diff_ptr = blob->cpu_diff();
    }

    for (int n = 0; n < blob->num(); ++n) {
      for (int c = 0; c < blob->channels(); ++c) {

        cv::Mat mat(blob->height(), blob->width(), CV_32FC1);
        cv::Mat matd(blob->height(), blob->width(), CV_32FC1);


        for (int h = 0; h < blob->height(); ++h) {
#pragma omp parallel for
          for (int w = 0; w < blob->width(); ++w) {
            mat.at<float>(h, w) = *(cpu_ptr + w);
          }
          cpu_ptr += blob->width();
        }

        if (store_diff) {
          for (int h = 0; h < blob->height(); ++h) {
  #pragma omp parallel for
            for (int w = 0; w < blob->width(); ++w) {
              matd.at<float>(h, w) = *(cpu_diff_ptr + w);
            }
            cpu_diff_ptr += blob->width();
          }
        }

        bofs::path outp(output_folder);

        std::stringstream ssp;
        ssp << "/";
        ssp << input_name.stem().string();
        ssp << "_" << st << "_" << y << "_" << x;

        std::stringstream ssf;
        ssf << "/";
        ssf << name;
        ssf << "_" << n << "_" << c;
        ssf << ".png";

        std::stringstream ssfd;
        ssfd << "/";
        ssfd << name;
        ssfd << "_" << n << "_" << c << "_diff";
        ssfd << ".png";

        bofs::path outpl = outp;
        outpl /= (ssp.str());
        bofs::create_directories(outpl);
        bofs::path filep = outpl;
        filep /= (ssf.str());

        bofs::path filepd = outpl;
        filepd /= (ssfd.str());

        cv::Mat outfc;
        cv::Mat outuc;

        {
          double minVal, maxVal;
          cv::minMaxLoc(mat, &minVal, &maxVal);

          mat.convertTo(outfc, CV_32FC1, 1.0 / (maxVal - minVal),
                    -minVal * 1.0 / (maxVal - minVal));
          outfc.convertTo(outuc, CV_8UC1, 255.0, 0.0);

          cv::imwrite(filep.string(), outuc);
        }
        if (store_diff) {
          double minVal, maxVal;
          cv::minMaxLoc(matd, &minVal, &maxVal);

          matd.convertTo(outfc, CV_32FC1, 1.0 / (maxVal - minVal),
                    -minVal * 1.0 / (maxVal - minVal));
          outfc.convertTo(outuc, CV_8UC1, 255.0, 0.0);

          cv::imwrite(filepd.string(), outuc);
        }
      }
    }
  }
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
  std::string format = output_param.has_format() ? output_param.format() : ".tif";
  std::transform(format.begin(), format.end(), format.begin(), ::tolower);
  bool fp32out = output_param.has_fp32_out() ? output_param.fp32_out() : false;
  if(fp32out) {
    format = ".tif";
  }
  std::set<std::string> filetypes = CreateImageTypesSet();
  if (filetypes.find(format) == filetypes.end()) {
    format = ".tif";
  }

  if(!(input_param.has_patch_size() && input_param.has_padding_size() && input_param.has_labels() && input_param.has_channels())) {
    LOG(FATAL) << "Patch size, padding size, label count or channel count parameter missing.";
  }
  int patch_size = input_param.patch_size();
  int padding_size = input_param.padding_size();
  unsigned int nr_labels = input_param.labels();
  unsigned int nr_channels = input_param.channels();

  if(!(process_param.has_process_net())) {
    LOG(FATAL) << "Processing network prototxt argument missing.";
  }

  std::string process_net = process_param.process_net();

  Net<float> net(process_net, caffe::TEST, Caffe::GetDefaultDevice());

  if(process_param.has_caffemodel()) {
    std::string caffe_model = process_param.caffemodel();
    net.CopyTrainedLayersFrom(caffe_model);
  }

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
          std::vector<int_tp> labels;

          images.push_back(crop);
          labels.push_back(0);

          boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
              (&net)->layers()[0])->AddMatVector(images, labels);

          float loss = 0.0;
          const vector<Blob<float>*>& result = net.ForwardPrefilled(&loss);

          if(process_param.has_filter_output()) {
            FilterOutputParam filter_param = process_param.filter_output();
            if(filter_param.has_output_filters() && filter_param.output_filters() && filter_param.has_output()) {
              ExportFilters(&net, filter_param.output(), process_set[i], st, yoff, xoff, false);
            }
          }

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
}  // namespace caffe_neural
