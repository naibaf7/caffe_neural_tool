/*
 * tiffio_wrapper.cpp
 *
 *  Created on: May 8, 2015
 *      Author: Fabian Tschopp
 */

#include "tiffio_wrapper.hpp"
#include <tiffio.h>
#include <iostream>

namespace caffe_neural {

void SaveTiff(std::vector<cv::Mat> image_stack, std::string file) {

  const char* filec = file.c_str();

  TIFF* tif = TIFFOpen(filec, "w");

  if (tif) {

    for (unsigned int i = 0; i < image_stack.size(); ++i) {
      int imagewidth = image_stack[i].cols;
      int imageheight = image_stack[i].rows;
      int nr_channels = image_stack[i].channels();
      bool fp32 = (image_stack[i].type() == CV_32FC1);

      unsigned char buf[imagewidth
          * (nr_channels == 3 ?
              sizeof(uint32) : (fp32 ? sizeof(float) : sizeof(uchar)))];
      void* raster = &buf;
      TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, imagewidth);
      TIFFSetField(tif, TIFFTAG_IMAGELENGTH, imageheight);
      TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, fp32 ? 32 : 8);
      TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT,
                   fp32 ? SAMPLEFORMAT_IEEEFP : SAMPLEFORMAT_UINT);
      TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, nr_channels);
      TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
      TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
      TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, nr_channels == 3 ? PHOTOMETRIC_RGB : PHOTOMETRIC_MINISBLACK);
      TIFFSetField(tif, TIFFTAG_SUBFILETYPE, FILETYPE_PAGE);
      TIFFSetField(tif, TIFFTAG_PAGENUMBER, i, image_stack.size());

      cv::Mat image = image_stack[i];

      switch (nr_channels) {
        case 1: {
          if (fp32) {
            for (int y = 0; y < imageheight; ++y) {
#pragma omp parallel for
              for (int x = 0; x < imagewidth; ++x) {
                ((float*) (raster))[x] = image.at<float>(y, x);
              }
              TIFFWriteScanline(tif, raster, y, 0);
            }
          } else {
            for (int y = 0; y < imageheight; ++y) {
#pragma omp parallel for
              for (int x = 0; x < imagewidth; ++x) {
                ((uchar*) (raster))[x] = image.at<uchar>(y, x);
              }
              TIFFWriteScanline(tif, raster, y, 0);
            }
          }
        }
          break;
        case 3:
        default: {
          for (int y = 0; y < imageheight; ++y) {
#pragma omp parallel for
            for (int x = 0; x < imagewidth; ++x) {
              ((uint32*) (raster))[x] = image.at<cv::Vec3b>(y, x)[2] << 16
                  | image.at<cv::Vec3b>(y, x)[1] << 8
                  | image.at<cv::Vec3b>(y, x)[0] << 0;
            }
            TIFFWriteScanline(tif, raster, y, 0);
          }
        }
          break;
      }
      TIFFWriteDirectory(tif);
    }
  }

  TIFFClose(tif);
}

std::vector<cv::Mat> LoadTiff(std::string file, int nr_channels) {

  std::vector<cv::Mat> image_stack;

  const char* filec = file.c_str();

  TIFF* tif = TIFFOpen(filec, "r");

  if (tif) {

    unsigned int imagewidth, imageheight;
    int dirs = 0;

    TIFFSetDirectory(tif, 0);

    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &imagewidth);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imageheight);

    uint32* raster = (uint32 *) _TIFFmalloc(
        imagewidth * imageheight * sizeof(uint32));

    do {
      ++dirs;
    } while (TIFFReadDirectory(tif));

    for (int n = 0; n < dirs; ++n) {
      TIFFSetDirectory(tif, n);

      cv::Mat image(imageheight, imagewidth, CV_8UC(nr_channels));
      TIFFReadRGBAImageOriented(tif, imagewidth, imageheight, raster,
      ORIENTATION_TOPLEFT);

      switch (nr_channels) {
        case 1: {
#pragma omp parallel for
          for (unsigned int y = 0; y < imageheight; ++y) {
            for (unsigned int x = 0; x < imagewidth; ++x) {
              image.at<uchar>(y, x) = raster[x + y * imagewidth] & 0xFF;
            }
          }
        }
          break;
        case 3:
        default: {
#pragma omp parallel for
          for (unsigned int y = 0; y < imageheight; ++y) {
            for (unsigned int x = 0; x < imagewidth; ++x) {
              image.at<cv::Vec3b>(y, x)[0] = (raster[x + y * imagewidth] >> 0)
                  & 0xFF;
              image.at<cv::Vec3b>(y, x)[1] = (raster[x + y * imagewidth] >> 8)
                  & 0xFF;
              image.at<cv::Vec3b>(y, x)[2] = (raster[x + y * imagewidth] >> 16)
                  & 0xFF;
            }
          }
        }
          break;
      }

      image_stack.push_back(image);
    }

    _TIFFfree(raster);

  }

  TIFFClose(tif);
  return image_stack;
}

}
