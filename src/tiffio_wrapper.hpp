/*
 * tiffio_wrapper.hpp
 *
 *  Created on: May 8, 2015
 *      Author: fabian
 */

#ifndef TIFFIO_WRAPPER_HPP_
#define TIFFIO_WRAPPER_HPP_

#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace caffe_neural {

void SaveTiff(std::vector<cv::Mat> image_stack, std::string file);
std::vector<cv::Mat> LoadTiff(std::string file, int nr_channels);


}


#endif /* TIFFIO_WRAPPER_HPP_ */
