#include "opencv2/opencv.hpp"
#include "common.h"
#include "slider.h"

#include "spaces.h"

// Processing functions

void bi_level_color_map(const cv::Mat &src, cv::Mat &dst)
{ // black and white
  for(int i = 0; i < src.rows; ++i)
    for(int j = 0; j < src.cols; ++j)
      dst.at<uchar>(i,j) = src.at<uchar>(i,j) > 127 ? 255 : 0;
  imshow("input image", src);
  imshow("black&white", dst);
}

void negative(const cv::Mat &src, cv::Mat &dst)
{
  for(int i = 0; i < src.rows; ++i)
    for(int j = 0; j < src.cols; ++j)
      dst.at<uchar>(i,j) = 255 - src.at<uchar>(i,j);
  imshow("input image", src);
  imshow("negative", dst);
}

int main() {
  Logger::init();

  DEBUG("Opening file {}", IMAGE("saturn.bmp"));
  cv::Mat img = FileUtils::readImage(IMAGE("saturn.bmp"), cv::IMREAD_GRAYSCALE);
  DEBUG("image loaded with {} {}", img.rows, img.cols);
  cv::Mat outImg(img.rows, img.cols, CV_8UC1);
  INFO("Press the arrow keys to cycle through execution slides");

  // Processing functions that the user can slide through
  Slider slider(
    { [&](){ bi_level_color_map(img, outImg); }
    , [&](){ negative(img, outImg); }
    }
  );

  // Loop
  std::string exportName = "";
  KEY operation = KEY::NONE;
  do {
    // Execute the current processing function
    slider.exec();

    // Input handling
    switch (operation)
    {
    case KEY::LEFT_ARROW:
      cv::destroyAllWindows();
      slider.previous();
      break;
    case KEY::RIGHT_ARROW:
      cv::destroyAllWindows();
      slider.next();
      break;
    case KEY::ENTER:
      FileUtils::quickSave(outImg);
      break;
    case KEY::SPACE:
      cv::destroyAllWindows();
      break;
    default:
      break;
    }

    // Handle keypresses
    operation = WaitKey(30);
  } while (operation != KEY::ESC);

  Logger::destroy();
  return 0;
}
