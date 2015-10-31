/*
 * caffe_neural_tool.hpp
 *
 *  Created on: May 14, 2015
 *      Author: fabian
 */

#ifndef CAFFE_NEURAL_TOOL_HPP_
#define CAFFE_NEURAL_TOOL_HPP_

#include "caffe/definitions.hpp"


#include "image_processor.hpp"
#include "tiffio_wrapper.hpp"
#include <glog/logging.h>
#include "google/protobuf/message.h"
#include "caffetool.pb.h"

#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include "caffe/caffe.hpp"
#include "caffe/common.hpp"
#include "caffe/solver.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

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


#define OCVDBGW "OpenCV Debug Window"


struct CommonSettings
{
  int param_index;
  bool graphic;
  bool debug;
};



#endif /* CAFFE_NEURAL_TOOL_HPP_ */
