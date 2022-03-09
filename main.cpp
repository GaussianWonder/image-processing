#include "opencv2/opencv.hpp"
#include "common.h"
#include "slider.h"
#include <cmath>

#include "spaces.h"

// Processing functions

void bi_level_color_map(const cv::Mat &src, cv::Mat &dst, const uchar threshold = 127)
{ // black and white
  for(int i = 0; i < src.rows; ++i)
    for(int j = 0; j < src.cols; ++j)
      dst.at<uchar>(i,j) = src.at<uchar>(i,j) > threshold ? 255 : 0;
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

void additive(const cv::Mat &src, cv::Mat &dst)
{
  for(int i = 0; i < src.rows; ++i)
    for(int j = 0; j < src.cols; ++j) {
      auto val = src.at<uchar>(i,j);
      dst.at<uchar>(i,j) = val <= 155 ? val + 100 : 255;
    }
  imshow("input image", src);
  imshow("additive grayscale", dst);
}

void multiplicative(const cv::Mat &src, cv::Mat &dst)
{
  for(int i = 0; i < src.rows; ++i)
    for(int j = 0; j < src.cols; ++j) {
      auto val = src.at<uchar>(i,j) * 2.0f;
      dst.at<uchar>(i,j) = val <= 255.0f ? (uchar) val : 255;
    }
  imshow("input image", src);
  imshow("multiplicative grayscale", dst);
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
  imshow("generated cv::Mat", dst);
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

void split_hsv(const cv::Mat &src)
{
  cv::Mat hue(src.rows, src.cols, CV_8UC1);
  cv::Mat saturation(src.rows, src.cols, CV_8UC1);
  cv::Mat value(src.rows, src.cols, CV_8UC1);

  for (int i = 0; i < src.rows; ++i)
    for (int j = 0; j < src.cols; ++j) {
      auto val = src.at<cv::Vec3b>(i,j);
      HSV hsv(val[2], val[1], val[0]);

      hue.at<uchar>(i,j) = hsv.h * 0.7f;
      saturation.at<uchar>(i,j) = hsv.s * 2.55f;
      value.at<uchar>(i,j) = hsv.v * 2.55f;
    }

  imshow("input image", src);
  imshow("hue", hue);
  imshow("saturation", saturation);
  imshow("value", value);
}

void color_to_grayscale(const cv::Mat &src)
{
  cv::Mat dst(src.rows, src.cols, CV_8UC1);
  for (int i = 0; i < src.rows; ++i)
    for (int j = 0; j < src.cols; ++j) {
      auto val = src.at<cv::Vec3b>(i,j);
      dst.at<uchar>(i,j) = (uchar) (val[0] + val[1] + val[2]) / 3;
    }
  imshow("input image", src);
  imshow("grayscale", dst);
}

void compute_historgram(const cv::Mat &src)
{
  int histSize = 361;
  float range[] = { 0, 256 }; //the upper boundary is exclusive
  const float* histRange[] = { range };

  std::vector<cv::Mat> hsvHists = {
    cv::Mat(src.rows, src.cols, CV_8UC1),
    cv::Mat(src.rows, src.cols, CV_8UC1),
    cv::Mat(src.rows, src.cols, CV_8UC1)
  };

  for (int i = 0; i < src.rows; ++i)
    for (int j = 0; j < src.cols; ++j) {
      auto val = src.at<cv::Vec3b>(i,j);
      HSV hsv(val[2], val[1], val[0]);

      hsvHists[0].at<uchar>(i,j) = hsv.h * 0.7f;
      hsvHists[1].at<uchar>(i,j) = hsv.s * 2.55f;
      hsvHists[2].at<uchar>(i,j) = hsv.v * 2.55f;
    }

  cv::Mat hueHist;
  cv::calcHist(hsvHists.data(), 1, 0, cv::Mat(), hueHist, 1, &histSize, histRange, true, false);

  int hist_w = 360, hist_h = 400;
  int bin_w = cvRound((double) hist_w / histSize);

  cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0,0,0));

  cv::normalize(hueHist, hueHist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

  for(int i=0; i<=histSize; ++i)
  {
    RGB rgb = hsv_to_rgb((float) i, 100.0f, 100.0f);
    auto color = cv::Scalar(rgb.b, rgb.g, rgb.r);
    auto currentPoint = cv::Point(
      bin_w * (i),
      hist_h - cvRound(hueHist.at<float>(i))
    );
    auto toBottom = cv::Point(
      bin_w * (i),
      hist_h
    );
    cv::line(
      histImage,
      currentPoint,
      toBottom,
      color,
      2, 8, 0
    );
  }

  imshow("Source", src);
  imshow("Hue Histogram", histImage);

  std::vector<float> maxLocals = {};

  const int WH = 5;
  const int windowSize = 2*WH+1;
  const float TH = 0.0003;
  for (int mid = WH + 1; mid<=histSize - WH - 1; ++mid) {
    const float midVal = hueHist.at<float>(mid);
    int sum = 0;
    bool applicable = true;

    const int start = mid - WH;
    const int stop = mid + WH;
    for (int i=start; i<stop; ++i) {
      float currentVal = hueHist.at<float>(i);
      sum += currentVal;

      if (currentVal > midVal) {
        applicable = false;
        break;
      }
    }

    float avg = (float) sum / (float) windowSize;
    if (applicable && midVal > avg + TH) {
      maxLocals.push_back(midVal);
    }
  }

  // cv::Mat maxLocalsHist(hist_h, hist_w, CV_8UC3, cv::Scalar(0,0,0));
  // for (auto maxLocal: maxLocals) {
  //   RGB rgb = hsv_to_rgb((float) maxLocal, 100.0f, 100.0f);
  //   auto color = cv::Scalar(rgb.b, rgb.g, rgb.r);
  //   auto currentPoint = cv::Point(
  //     bin_w * (maxLocal),
  //     hist_h - cvRound(hueHist.at<float>(maxLocal))
  //   );
  //   auto toBottom = cv::Point(
  //     bin_w * (maxLocal),
  //     hist_h
  //   );
  //   cv::line(
  //     maxLocalsHist,
  //     currentPoint,
  //     toBottom,
  //     color,
  //     2, 8, 0
  //   );
  // }
  // imshow("Max local histogram", maxLocalsHist);

  const float errorDistribution[3][3] = {
    {0, 0, 0},
    {0, 1, 7/16},
    {3/16, 5/16, 1/16}
  };

  float errorAccumulation[(int) src.rows][(int) src.cols] = { 0 };
  cv::Mat reducedColor(src.rows, src.cols, CV_8UC3);
  const int maxLocalSz = maxLocals.size();
  for (int i = 0; i < src.rows; ++i)
    for (int j = 0; j < src.cols; ++j) {
      auto val = src.at<cv::Vec3b>(i,j);
      HSV hsv(val[2], val[1], val[0]);
      const uchar grayHue = hsv.h * 0.7f;

      float minDiff = 500.0f;
      float maxLocalMinDiff = 0.0f;
      for (float maxLocal : maxLocals) {
        float diff = std::abs(grayHue - maxLocal);
        if (minDiff > diff) {
          minDiff = diff;
          maxLocalMinDiff = maxLocal;
        }
      }

      float maxLocalHue = maxLocalMinDiff * 1.41f;
      RGB newColor = hsv_to_rgb(maxLocalHue, hsv.s, hsv.v);

      reducedColor.at<cv::Vec3b>(i, j) = cv::Vec3b(newColor.B(), newColor.G(), newColor.R());

      float error = hsv.h - maxLocalHue;
      for (int ii; ii<3; ++ii) {
        for (int jj; jj<3; ++jj) {
          int errorI = i + ii;
          int errorJ = j + jj;
          if (errorI > 0 && errorI < src.rows && errorJ > 0 && errorJ < src.cols) {
            errorAccumulation[errorI][errorJ] += error * errorDistribution[ii][jj];
          }
        }
      }
    }

  cv::Mat ditheredColor(src.rows, src.cols, CV_8UC3);
  for (int i = 0; i < src.rows; ++i)
    for (int j = 0; j < src.cols; ++j) {
      auto val = reducedColor.at<cv::Vec3b>(i,j);
      HSV hsv(val[2], val[1], val[0]);
      auto error = errorAccumulation[i][j];
      auto newHue = float(hsv.h + error);
      if (newHue > 360.0f)
        newHue -= 360.0f;
      auto replaceWith = hsv_to_rgb(newHue, hsv.s, hsv.v);
      reducedColor.at<cv::Vec3b>(i,j) = cv::Vec3b(replaceWith.B(), replaceWith.G(), replaceWith.R());
    }

  imshow("Thresholded", reducedColor);
  imshow("Dithered", ditheredColor);
}

int main() {
  Logger::init();

  DEBUG("Opening file {}", IMAGE("saturn.bmp"));
  cv::Mat img = FileUtils::readImage(IMAGE("saturn.bmp"), cv::IMREAD_GRAYSCALE);
  DEBUG("image loaded with {} {}", img.rows, img.cols);
  cv::Mat outImg(img.rows, img.cols, CV_8UC1);
  cv::Mat genRGB(img.rows, img.cols, CV_8UC3);

  cv::Mat flowers = FileUtils::readImage(IMAGE("flowers_24bits.bmp"), cv::IMREAD_COLOR);
  cv::Mat hueSpectrum = FileUtils::readImage(IMAGE("huespectrum.png"), cv::IMREAD_COLOR);

  INFO("Press the arrow keys to cycle through execution slides");

  // Processing functions that the user can slide through
  uchar thresh = 127;
  Slider slider(
    { [&](){ bi_level_color_map(img, outImg, thresh); }
    , [&](){ negative(img, outImg); }
    , [&](){ additive(img, outImg); }
    , [&](){ multiplicative(img, outImg); }
    , [&](){ four_squares(genRGB); }
    , [&](){ split_channels(flowers); }
    , [&](){ color_to_grayscale(flowers); }
    , [&](){ split_hsv(flowers); }
    , [&](){ compute_historgram(flowers); }
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
    case KEY::UP_ARROW:
      thresh += 10;
      break;
    case KEY::DOWN_ARROW:
      thresh -= 10;
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
