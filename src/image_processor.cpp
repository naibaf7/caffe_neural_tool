/*
 * image_preprocessor.cpp
 *
 *  Created on: Apr 3, 2015
 *      Author: Fabian Tschopp
 */

#include "image_processor.hpp"
#include <glog/logging.h>

#include <omp.h>
#include <iostream>
#include <set>
#include "utils.hpp"

namespace caffe_neural {

ImageProcessor::ImageProcessor(int patch_size, int nr_labels)
    : patch_size_(patch_size),
      nr_labels_(nr_labels) {

}

std::vector<cv::Mat>& ImageProcessor::raw_images() {
  return raw_images_;
}

std::vector<cv::Mat>& ImageProcessor::label_images() {
  return label_images_;
}

std::vector<int>& ImageProcessor::image_number() {
  return image_number_;
}

void ImageProcessor::SetCropParams(int image_crop, int label_crop) {
  image_crop_ = image_crop;
  label_crop_ = label_crop;
}

void ImageProcessor::SetNormalizationParams(bool apply) {
  apply_normalization_ = apply;
}

void ImageProcessor::ClearImages() {
  raw_images_.clear();
  label_images_.clear();
  image_number_.clear();
}

void ImageProcessor::SetLabelConsolidateParams(bool apply, std::vector<int> labels) {
  label_consolidate_ = apply;
  label_consolidate_labels_ = labels;
}

void ImageProcessor::SubmitImage(cv::Mat raw, int img_id,
                                 std::vector<cv::Mat> labels) {

  std::vector<cv::Mat> rawsplit;
  cv::split(raw, rawsplit);

  if (apply_clahe_) {
    for (unsigned int i = 0; i < rawsplit.size(); ++i) {
      cv::Mat dst;
      clahe_->apply(rawsplit[i], dst);
      rawsplit[i] = dst;
    }
  }

  cv::Mat src;
  cv::merge(rawsplit, src);
  src.convertTo(src, CV_32FC(3), 1.0 / 255.0);

  if (apply_normalization_) {
    cv::Mat dst;
    cv::normalize(src, dst, -1.0, 1.0, cv::NORM_MINMAX);
    src = dst;
  }

  if (apply_border_reflect_) {
    cv::Mat dst;
    cv::copyMakeBorder(src, dst, border_size_, border_size_, border_size_,
                       border_size_, IPL_BORDER_REFLECT, cv::Scalar::all(0.0));
    src = dst;
  }

  raw_images_.push_back(src);
  image_number_.push_back(img_id);
  label_stack_.push_back(labels);

}

int ImageProcessor::Init() {

  if (label_stack_[0].size() > 1) {
    for (unsigned int j = 0; j < label_stack_.size(); ++j) {

      cv::Mat dst_label(label_stack_[j][0].rows, label_stack_[j][0].cols,
      CV_32FC1);
      dst_label.setTo(cv::Scalar(0.0));

      for (unsigned int i = 0; i < label_stack_[j].size(); ++i) {
#pragma omp parallel for
        for (int y = 0; y < label_stack_[j][i].rows; ++y) {
          for (int x = 0; x < label_stack_[j][i].cols; ++x) {
            // Multiple images with 1 label defined per image
            int ks = label_stack_[j][i].at<unsigned char>(y, x);
            if (ks > 0) {
              (dst_label.at<float>(y, x)) = i;
            }
          }
        }
      }
      label_images_.push_back(dst_label);
    }
  } else {

    std::set<int> labelset;

    for (unsigned int j = 0; j < label_stack_.size(); ++j) {
      for (int y = 0; y < label_stack_[j][0].rows; ++y) {
        for (int x = 0; x < label_stack_[j][0].cols; ++x) {
          int ks = label_stack_[j][0].at<unsigned char>(y, x);
          // Label not yet registered
          if (labelset.find(ks) == labelset.end()) {
            labelset.insert(ks);
          }
        }
      }
    }

    for (unsigned int j = 0; j < label_stack_.size(); ++j) {
      cv::Mat dst_label(label_stack_[j][0].rows, label_stack_[j][0].cols,
      CV_32FC1);
      dst_label.setTo(cv::Scalar(0.0));
#pragma omp parallel for
      for (int y = 0; y < label_stack_[j][0].rows; ++y) {
        for (int x = 0; x < label_stack_[j][0].cols; ++x) {
          // Single image with many labels defined per image
          int ks = label_stack_[j][0].at<unsigned char>(y, x);
          (dst_label.at<float>(y, x)) = (float) std::distance(
              labelset.begin(), labelset.find(ks));
        }
      }
      label_images_.push_back(dst_label);
    }
  }

  label_stack_.clear();

  if (raw_images_.size() == 0 || label_images_.size() == 0
      || raw_images_.size() != label_images_.size()) {
    return -1;
  }

  image_size_x_ = raw_images_[0].cols - 2 * border_size_;
  image_size_y_ = raw_images_[0].rows - 2 * border_size_;

  int off_size_x = (image_size_x_ - patch_size_) + 1;
  int off_size_y = (image_size_y_ - patch_size_) + 1;

  offset_selector_ = GetRandomUniform<double>(
      0, label_images_.size() * off_size_x * off_size_y);

  if (apply_label_hist_eq_) {

    std::vector<long> label_count(nr_labels_);
    std::vector<double> label_freq(nr_labels_);

    long total_count = 0;
    for (unsigned int k = 0; k < label_images_.size(); ++k) {
      cv::Mat label_image = label_images_[k];
      for (int y = 0; y < image_size_y_; ++y) {
        for (int x = 0; x < image_size_x_; ++x) {
          // Label counting should be biased towards the borders, as less batches cover those parts
          long mult = std::min(std::min(x, image_size_x_ - x), patch_size_)
              * std::min(std::min(y, image_size_y_ - y), patch_size_);
          label_count[label_image.at<float>(y, x)] += mult;
          total_count += mult;
        }
      }
    }

    for (int l = 0; l < nr_labels_; ++l) {
      label_freq[l] = (double) (label_count[l]) / (double) (total_count);
      LOG(INFO) << "Label " << l << ": " << label_freq[l];
    }

    if (apply_label_patch_prior_) {

      std::vector<double> weighted_label_count(nr_labels_);

      label_running_probability_.resize(
          label_images_.size() * off_size_x * off_size_y);


      // Loop over all images
#pragma omp parallel for
      for (unsigned int k = 0; k < label_images_.size(); ++k) {
        cv::Mat label_image = label_images_[k];
        std::vector<long> patch_label_count(nr_labels_);
        // Loop over all possible patches
        for (int y = 0; y < off_size_y; ++y) {
          for (int x = 0; x < off_size_x; ++x) {

            if(x == 0) {
              // Fully compute the patches at the beginning of the row
              for(int l = 0; l < nr_labels_; ++l) {
                patch_label_count[l] = 0;
              }
              for (int py = y; py < y + patch_size_; ++py) {
                for (int px = x; px < x + patch_size_; ++px) {
                  patch_label_count[label_image.at<float>(py, px)]++;
                }
              }
            } else {
              // Only compute difference for further patches in the row (more efficient)
              for (int py = y; py < y + patch_size_; ++py) {
                patch_label_count[label_image.at<float>(py, x - 1)]--;
                patch_label_count[label_image.at<float>(py, x + patch_size_ - 1)]++;
              }
            }

            // Compute the weight of the patch
            double patch_weight = 0;
            for (int l = 0; l < nr_labels_; ++l) {
              patch_weight += (((double) (patch_label_count[l]))
                  / ((double) (patch_size_ * patch_size_))) / (label_freq[l]);
            }
            for (int l = 0; l < nr_labels_; ++l) {
              weighted_label_count[l] += patch_weight * patch_label_count[l];
            }
            label_running_probability_[k * off_size_x * off_size_y
                + y * off_size_x + x] = patch_weight;
          }
        }
      }

      for (unsigned int k = 1; k < label_running_probability_.size(); ++k) {
        label_running_probability_[k] += label_running_probability_[k - 1];
      }

      double freq_divisor = 0;
      for (int l = 0; l < nr_labels_; ++l) {
        freq_divisor += weighted_label_count[l];
      }
      for (int l = 0; l < nr_labels_; ++l) {
        label_freq[l] = weighted_label_count[l] / freq_divisor;
        LOG(INFO) << "Label " << l << ": " << label_freq[l];
      }

      offset_selector_ = GetRandomUniform<double>(
          0, label_running_probability_[label_running_probability_.size() - 1]);
    }

    if (apply_label_pixel_mask_) {

      double boost_divisor = 0;

      for (int l = 0; l < nr_labels_; ++l) {
        label_freq[l] *= 1.0 / label_boost_[l];
        boost_divisor += label_freq[l];
      }

      for (int l = 0; l < nr_labels_; ++l) {
        label_freq[l] *= 1.0 / boost_divisor;
      }

      label_mask_probability_.resize(nr_labels_);
      float mask_divisor = 0;
      for (int l = 0; l < nr_labels_; ++l) {
        label_mask_probability_[l] = 1.0 / label_freq[l];
        mask_divisor = std::max(mask_divisor, label_mask_probability_[l]);
      }
      for (int l = 0; l < nr_labels_; ++l) {
        label_mask_probability_[l] /= mask_divisor;
        LOG(INFO) << "Label " << l << ", mask probability: "
                  << label_mask_probability_[l];
      }
    }
  }

  return 0;
}

void ImageProcessor::SetBlurParams(bool apply, float mean, float std,
                                   int blur_size) {
  apply_blur_ = apply;
  blur_mean_ = mean;
  blur_std_ = std;
  blur_size_ = blur_size;
  blur_random_selector_ = GetRandomNormal<float>(mean, std);
}

void ImageProcessor::SetBorderParams(bool apply, int border_size) {
  apply_border_reflect_ = apply;
  border_size_ = border_size;
}

void ImageProcessor::SetClaheParams(bool apply, float clip_limit) {
  apply_clahe_ = apply;
  clahe_ = cv::createCLAHE();
  clahe_->setClipLimit(clip_limit);
}

void ImageProcessor::SetRotationParams(bool apply) {
  apply_rotation_ = apply;
  rotation_rand_ = GetRandomOffset(0, 3);
}
void ImageProcessor::SetPatchMirrorParams(bool apply) {
  apply_patch_mirroring_ = apply;
  patch_mirror_rand_ = GetRandomOffset(0, 2);
}

void ImageProcessor::SetLabelHistEqParams(bool apply, bool patch_prior,
                                          bool mask_prob,
                                          std::vector<float> label_boost) {
  apply_label_hist_eq_ = apply;
  apply_label_patch_prior_ = patch_prior;
  apply_label_pixel_mask_ = mask_prob;
  label_mask_prob_rand_.resize(omp_get_max_threads());
  for (int i = 0; i < omp_get_max_threads(); ++i) {
    label_mask_prob_rand_[i] = GetRandomUniform<float>(0.0, 1.0);
  }
  label_boost_ = label_boost;
}

long ImageProcessor::BinarySearchPatch(double offset) {
  long mid, left = 0;
  long right = label_running_probability_.size();
  while (left < right) {
    mid = left + (right - left) / 2;
    if (offset > label_running_probability_[mid]) {
      left = mid + 1;
    } else if (offset < label_running_probability_[mid]) {
      right = mid;
    } else {
      return left;
    }
  }
  return left;
}

ProcessImageProcessor::ProcessImageProcessor(int patch_size, int nr_labels)
    : ImageProcessor(patch_size, nr_labels) {
}

TrainImageProcessor::TrainImageProcessor(int patch_size, int nr_labels)
    : ImageProcessor(patch_size, nr_labels) {
}

std::vector<cv::Mat> TrainImageProcessor::DrawPatchRandom() {

  double offset = offset_selector_();

  long abs_id = 0;

  if (apply_label_hist_eq_ && apply_label_patch_prior_) {
    abs_id = BinarySearchPatch(offset);
  } else {
    abs_id = (long) offset;
  }

  int off_size_x = (image_size_x_ - patch_size_) + 1;
  int off_size_y = (image_size_y_ - patch_size_) + 1;

  int img_id = abs_id / (off_size_x * off_size_y);
  int yoff = (abs_id - (img_id * off_size_x * off_size_y)) / off_size_x;
  int xoff = abs_id
      - ((img_id * off_size_x * off_size_y) + (yoff * off_size_x));

  cv::Mat &full_image = raw_images_[img_id];
  cv::Mat &full_label = label_images_[img_id];

  int actual_patch_size = patch_size_ + 2 * border_size_;
  int actual_label_size = patch_size_;

  cv::Rect roi_patch(xoff, yoff, actual_patch_size, actual_patch_size);
  cv::Rect roi_label(xoff, yoff, actual_label_size, actual_label_size);

// Deep copy so that the original image in storage doesn't get messed up
  cv::Mat patch = full_image(roi_patch).clone();
  cv::Mat label = full_label(roi_label).clone();

  if (apply_patch_mirroring_) {
    int flipcode = patch_mirror_rand_() - 1;
    cv::Mat mirror_patch;
    cv::Mat mirror_label;
    cv::flip(patch, mirror_patch, flipcode);
    cv::flip(label, mirror_label, flipcode);
    patch = mirror_patch;
    label = mirror_label;
  }

  if (apply_rotation_) {

    cv::Mat rotate_patch;
    cv::Mat rotate_label;
    cv::Mat tmp_patch;
    cv::Mat tmp_label;

    switch (rotation_rand_()) {
      case 0:
        rotate_patch = patch;
        rotate_label = label;
        break;
      case 1:
        tmp_patch = patch.t();
        tmp_label = label.t();
        cv::flip(tmp_patch, rotate_patch, 1);
        cv::flip(tmp_label, rotate_label, 1);
        break;
      case 2:
        cv::flip(patch, rotate_patch, -1);
        cv::flip(label, rotate_label, -1);
        break;
      case 3:
        tmp_patch = patch.t();
        tmp_label = label.t();
        cv::flip(tmp_patch, rotate_patch, 0);
        cv::flip(tmp_label, rotate_label, 0);
        break;
    }

    patch = rotate_patch;
    label = rotate_label;

  }

  cv::Rect roi_rot_patch(0, 0, actual_patch_size - image_crop_,
                         actual_patch_size - image_crop_);

  cv::Rect roi_rot_label(0, 0, actual_label_size - label_crop_,
                         actual_label_size - label_crop_);

  patch = patch(roi_rot_patch);
  label = label(roi_rot_label);

  if (apply_blur_) {
    cv::Size ksize(blur_size_, blur_size_);
    float sigma = blur_random_selector_();
    cv::GaussianBlur(patch, patch, ksize, sigma);
  }

  if (apply_label_hist_eq_ && apply_label_pixel_mask_) {
#pragma omp parallel
    {
      std::function<float()> &randprob =
          label_mask_prob_rand_[omp_get_thread_num()];
#pragma omp for
      for (int y = 0; y < patch_size_; ++y) {
        for (int x = 0; x < patch_size_; ++x) {
          label.at<float>(y, x) =
              label_mask_probability_[label.at<float>(y, x)] >= randprob() ?
                  label.at<float>(y, x) : -1.0;
        }
      }
    }
  }

  std::vector<cv::Mat> patch_label;

  if (label_consolidate_) {
#pragma omp parallel for
    for (int y = 0; y < label.rows; ++y) {
      for (int x = 0; x < label.cols; ++x) {
        label.at<float>(y, x) = label.at<float>(y, x) < 0 ?
            -1.0 : label_consolidate_labels_[label.at<float>(y, x)];
      }
    }
  }


  patch_label.push_back(patch);
  patch_label.push_back(label);

  return patch_label;
}

}
