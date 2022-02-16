#include "opencv2/opencv.hpp"
#include "file_utils.h"
#include "logger.h"

#include <string>
#include <fstream>
#include <sstream>

std::string FileUtils::readFile(const std::string &fileName)
{
  std::ifstream file(fileName);
  if (file.is_open()) {
    std::stringstream fileStream;
    fileStream << file.rdbuf();
    file.close();
    return fileStream.str();
  }
  else {
    ERROR("Failed to open file {}. Returning empty string.", fileName);
    return "";
  }
}

cv::Mat FileUtils::readImage(const std::string &fileName, const cv::ImreadModes mode)
{
  return imread(fileName, mode);
}
