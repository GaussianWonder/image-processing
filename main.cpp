#include <iostream>
#include "opencv2/opencv.hpp"

#include "logger.h"
#include "paths.h"

using namespace cv;

int main() {
  Logger::init();

  WARN("Opening file {}", PathConcat(ImageFolder, "/saturn.bmp"));

  Mat img = imread(PathConcat(ImageFolder, "/saturn.bmp"), IMREAD_GRAYSCALE);
  Mat outImg(img.rows, img.cols, CV_8UC1);
  INFO("image loaded with {} {}", img.rows, img.cols);
  for(int i = 0; i < img.rows; i++)
  {
    for(int j = 0; j < img.cols; j++)
    {
      outImg.at<uchar>(i,j) = 255 - img.at<uchar>(i,j);
    }
  }
  imshow("input image", img);
  imshow("output image", outImg);
  waitKey(0);

  Logger::destroy();
  return 0;
}
