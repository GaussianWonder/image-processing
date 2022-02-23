#include "opencv2/opencv.hpp"
#include "common.h"
#include "slider.h"

// Processing functions

void bi_level_color_map(const cv::Mat &src, cv::Mat &dst)
{ // black and white
  for(int i = 0; i < src.rows; ++i)
    for(int j = 0; j < src.cols; ++j)
      dst.at<uchar>(i,j) = src.at<uchar>(i,j) > 127 ? 255 : 0;
  imshow("input image", src);
  imshow("output image", dst);
}

void negative(const cv::Mat &src, cv::Mat &dst)
{
  for(int i = 0; i < src.rows; ++i)
    for(int j = 0; j < src.cols; ++j)
      dst.at<uchar>(i,j) = 255 - src.at<uchar>(i,j);
  imshow("input image", src);
  imshow("output image", dst);
}

void additive(const cv::Mat &src, cv::Mat &dst)
{
  for(int i = 0; i < src.rows; ++i)
    for(int j = 0; j < src.cols; ++j) {
      auto val = src.at<uchar>(i,j);
      dst.at<uchar>(i,j) = val <= 155 ? val + 100 : 255;
    }
  imshow("input image", src);
  imshow("output image", dst);
}

void multiplicative(const cv::Mat &src, cv::Mat &dst)
{
  for(int i = 0; i < src.rows; ++i)
    for(int j = 0; j < src.cols; ++j) {
      auto val = src.at<uchar>(i,j) * 2.0f;
      dst.at<uchar>(i,j) = val <= 255.0f ? (uchar) val : 255;
    }
  imshow("input image", src);
  imshow("output image", dst);
}

void four_squares(cv::Mat &dst)
{
  auto halfCols = dst.cols / 2;
  auto halfRows = dst.rows / 2;
  cv::Rect rectangles[] = {
    cv::Rect(0, 0, halfCols, halfRows),
    cv::Rect(halfCols, 0, halfCols, halfRows),
    cv::Rect(0, halfRows, halfCols, halfRows),
    cv::Rect(halfCols, halfRows, halfCols, halfRows)
  };
  cv::Scalar scalars[] = {
    cv::Scalar(255, 255, 255),
    cv::Scalar(0, 0, 255),
    cv::Scalar(0, 255, 0),
    cv::Scalar(0, 255, 255)
  };
  for (int i = 0; i < 4; ++i)
    cv::rectangle(dst, rectangles[i], scalars[i], -1);
  imshow("output image", dst);
}

void split_channels(const cv::Mat &src)
{
  cv::Mat red(src.rows, src.cols, CV_8UC3);
  cv::Mat green(src.rows, src.cols, CV_8UC3);
  cv::Mat blue(src.rows, src.cols, CV_8UC3);

  for (int i = 0; i < src.rows; ++i)
    for (int j = 0; j < src.cols; ++j) {
      auto val = src.at<cv::Vec3b>(i,j);
      red.at<cv::Vec3b>(i,j) = cv::Vec3b(0, 0, val[2]);
      green.at<cv::Vec3b>(i,j) = cv::Vec3b(0, val[1], 0);
      blue.at<cv::Vec3b>(i,j) = cv::Vec3b(val[0], 0, 0);
    }

  imshow("input image", src);
  imshow("red", red);
  imshow("green", green);
  imshow("blue", blue);
}

int main() {
  Logger::init();

  DEBUG("Opening file {}", IMAGE("saturn.bmp"));
  cv::Mat img = FileUtils::readImage(IMAGE("saturn.bmp"), cv::IMREAD_GRAYSCALE);
  DEBUG("image loaded with {} {}", img.rows, img.cols);
  cv::Mat outImg(img.rows, img.cols, CV_8UC1);
  cv::Mat genRGB(img.rows, img.cols, CV_8UC3);

  cv::Mat flowers = FileUtils::readImage(IMAGE("flowers_24bits.bmp"), cv::IMREAD_COLOR);

  INFO("Press the arrow keys to cycle through execution slides");

  // Processing functions that the user can slide through
  Slider slider(
    { [&](){ bi_level_color_map(img, outImg); }
    , [&](){ negative(img, outImg); }
    , [&](){ additive(img, outImg); }
    , [&](){ multiplicative(img, outImg); }
    , [&](){ four_squares(genRGB); }
    , [&](){ split_channels(flowers); }
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
      slider.previous();
      break;
    case KEY::RIGHT_ARROW:
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
