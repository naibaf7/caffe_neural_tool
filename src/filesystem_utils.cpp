/*
 * filesystem_utils.cpp
 *
 *  Created on: May 14, 2015
 *      Author: Fabian Tschopp
 */

#include "../include/filesystem_utils.hpp"

namespace caffe_neural {

std::vector<std::vector<bofs::path>> LoadTrainingSetItems(
    std::set<std::string> filetypes, std::string raw_path,
    std::string label_path, int* error) {
  std::vector<std::vector<bofs::path>> set;

  bofs::path rpath(raw_path);

  try {
    if (bofs::exists(rpath) && bofs::is_directory(rpath)) {

      std::vector<bofs::path> pathvec;

      std::copy(bofs::directory_iterator(rpath), bofs::directory_iterator(),
                std::back_inserter(pathvec));

      std::sort(pathvec.begin(), pathvec.end());

      for (std::vector<bofs::path>::const_iterator it(pathvec.begin());
          it != pathvec.end(); ++it) {
        std::string type = bofs::extension(*it);
        std::transform(type.begin(), type.end(), type.begin(), ::tolower);
        if (filetypes.find(type) != filetypes.end()) {
          std::vector<bofs::path> subset;
          subset.push_back(*it);
          set.push_back(subset);
        }
      }

    } else {
      (*error) = FSU_ERR_NO_FOLDER;
    }
  } catch (const bofs::filesystem_error& ex) {
    (*error) = FSU_ERR_EXCEPTION;
  }

  bofs::path lpath(label_path);

  try {
    if (bofs::exists(lpath) && bofs::is_directory(lpath)) {

      std::vector<bofs::path> pathvec;

      std::copy(bofs::directory_iterator(lpath), bofs::directory_iterator(),
                std::back_inserter(pathvec));

      std::sort(pathvec.begin(), pathvec.end());
      int idxi = 0;

      for (std::vector<bofs::path>::const_iterator it(pathvec.begin());
          it != pathvec.end(); ++it, ++idxi) {

        if (bofs::is_directory(*it)) {
          // Case: Multiple subdirectories for labels
          std::vector<bofs::path> subpathvec;

          std::copy(bofs::directory_iterator(*it), bofs::directory_iterator(),
                    std::back_inserter(subpathvec));

          std::sort(subpathvec.begin(), subpathvec.end());
          int idxj = 0;

          for (std::vector<bofs::path>::const_iterator subit(subpathvec.begin());
              subit != subpathvec.end(); ++subit, ++idxj) {
            std::string type = bofs::extension(*subit);
            std::transform(type.begin(), type.end(), type.begin(), ::tolower);
            if (filetypes.find(type) != filetypes.end()) {
              std::vector<bofs::path> &subset = set[idxj];
              subset.push_back(*subit);

            }
          }
        } else {
          // Case: Single image with all labels
          std::string type = bofs::extension(*it);
          std::transform(type.begin(), type.end(), type.begin(), ::tolower);
          if (filetypes.find(type) != filetypes.end()) {
            std::vector<bofs::path> &subset = set[idxi];
            subset.push_back(*it);
          }
        }
      }

    } else {
      (*error) = FSU_ERR_NO_FOLDER;
    }
  } catch (const bofs::filesystem_error& ex) {
    (*error) = FSU_ERR_EXCEPTION;
  }

  return set;
}

std::vector<bofs::path> LoadProcessSetItems(std::set<std::string> filetypes,
                                            std::string raw_path, int* error) {

  std::vector<bofs::path> set;

  bofs::path rpath(raw_path);

  try {
    if (bofs::exists(rpath) && bofs::is_directory(rpath)) {

      std::vector<bofs::path> pathvec;

      std::copy(bofs::directory_iterator(rpath), bofs::directory_iterator(),
                std::back_inserter(pathvec));

      std::sort(pathvec.begin(), pathvec.end());

      for (std::vector<bofs::path>::const_iterator it(pathvec.begin());
          it != pathvec.end(); ++it) {

        std::string type = bofs::extension(*it);
        std::transform(type.begin(), type.end(), type.begin(), ::tolower);
        if (filetypes.find(type) != filetypes.end()) {
          set.push_back(*it);
        }
      }

    } else {
      (*error) = FSU_ERR_NO_FOLDER;
    }
  } catch (const bofs::filesystem_error& ex) {
    (*error) = FSU_ERR_EXCEPTION;
  }

  return set;
}

std::set<std::string> CreateImageTypesSet() {
  std::set<std::string> filetypes;

  filetypes.insert(".jpeg");
  filetypes.insert(".jpg");
  filetypes.insert(".bmp");
  filetypes.insert(".tif");
  filetypes.insert(".tiff");
  filetypes.insert(".png");

  return filetypes;
}

}
