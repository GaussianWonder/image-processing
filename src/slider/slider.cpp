#include "slider.h"

using Executor = std::function<void()>;
using Executors = std::vector<Executor>;

Slider::Slider(Executors executors)
  :executors(executors)
{
  size = executors.size();
}

void Slider::next()
{
  currentIndex = currentIndex == size - 1 ? 0 : currentIndex + 1;
}

void Slider::previous()
{
  currentIndex = currentIndex == 0 ? size - 1 : currentIndex - 1;
}

void Slider::exec()
{
  executors[currentIndex]();
}
