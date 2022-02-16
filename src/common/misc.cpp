#include "misc.h"

KEY resolvedKey(const int key)
{
  switch (key) {
    case KEY_ESC:
    case KEY_SPACE:
    case KEY_ENTER:
    case KEY_DOWN_ARROW:
    case KEY_RIGHT_ARROW:
    case KEY_UP_ARROW:
    case KEY_LEFT_ARROW:
      return static_cast<KEY>(key);
    default:
      return KEY::NONE;
  }
}

#include <chrono>
#include <sys/time.h>
#include <ctime>
#include <sstream>
#include <filesystem>
#include <unistd.h>

using std::chrono::duration_cast;
using std::chrono::seconds;
using std::chrono::system_clock;
using std::filesystem::current_path;

#include "paths.h"

std::string nextImageName()
{
  // Generate a file name bound to be unique
  const auto secSinceEpoch = duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
  std::stringstream fName;
  fName << secSinceEpoch << ".bmp";
  // Make a path to the file
  std::filesystem::path path;
  path += current_path();
  path += "/assets/exports/";
  path += fName.str();

  return path.string();
}