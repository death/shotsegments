#include <iostream>
#include <ostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <getopt.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

const int DefaultThreshold = 50;
const int DefaultMinDuration = 1000;

enum {
  TS_INPUT,
  TS_OUTPUT,
  TS_DURATION
};

void usage();
void save(int frame, cv::Mat image, const std::string & suffix);
std::string timespec(int frame, double fps, int mode);
std::string segmentfile(const std::string & basis, int segment);

int main(int argc, char *argv[])
{
  // Command line options

  if (argc < 2) {
    usage();
    return 0;
  }

  std::string inputFile;
  bool saveImages = false;
  bool ffmpeg = false;
  int threshold = DefaultThreshold;
  int minDuration = DefaultMinDuration;
  int verbose = 0;

  while (1) {
    static struct option longOptions[] = {
      {"in",           required_argument, 0, 'i'},
      {"save-images",  no_argument,       0, 's'},
      {"threshold",    required_argument, 0, 't'},
      {"min-duration", required_argument, 0, 'm'},
      {"ffmpeg",       no_argument,       0, 'f'},
      {"verbose",      optional_argument, 0, 'v'},
      {"help",         no_argument,       0, 'h'},
      {0,              0,                 0,  0 }
    };
    int c = getopt_long(argc, argv, "i:st:m:fv::h", longOptions, NULL);
    if (c < 0) {
      break;
    }
    switch (c) {
    case 'i':
      inputFile = optarg;
      break;
    case 's':
      saveImages = true;
      break;
    case 't':
      threshold = atoi(optarg);
      if (threshold == 0) {
        threshold = DefaultThreshold;
      }
      break;
    case 'm':
      minDuration = atoi(optarg);
      if (minDuration == 0) {
        minDuration = DefaultMinDuration;
      }
      break;
    case 'f':
      ffmpeg = true;
      break;
    case 'v':
      if (optarg) {
        verbose = atoi(optarg);
      } else {
        verbose = 1;
      }
      break;
    case 'h':
      usage();
      return 0;
    default:
      break;
    }
  }

  if (inputFile.empty()) {
    usage();
    return 0;
  }

  // The bee's knees

  cv::VideoCapture video;
  if (!video.open(inputFile)) {
    std::cerr << inputFile << ": can't open video\n";
    return 1;
  }

  cv::Mat previous;
  cv::Mat previousColor;
  cv::Mat currentColor;
  cv::Mat current;
  cv::Mat diffs;
  int frame = 1;
  int numPixels = 0;
  int lastScore = 0;
  std::vector<int> markers;
  double fps = video.get(cv::CAP_PROP_FPS);

  if (verbose) {
    std::cout << "FPS=" << fps << "\n";
  }

  if (!video.read(previousColor)) {
    std::cerr << inputFile << ": need some frames\n";
    return 1;
  }

  cv::cvtColor(previousColor, previous, CV_RGB2GRAY);
  diffs = previous.clone();
  numPixels = previous.rows * previous.cols;
  markers.push_back(0);
  if (saveImages) {
    save(0, previousColor, "in");
  }

  while (video.read(currentColor)) {
    cv::cvtColor(currentColor, current, CV_RGB2GRAY);

    cv::absdiff(current, previous, diffs);
    cv::Scalar s = cv::sum(diffs);
    int score = s[0] / numPixels;
    int scoreDiff = std::abs(score - lastScore);
    if (verbose > 1 || (verbose == 1 && ((frame % 1000) == 0))) {
      std::cout << "Frame=" << frame << " Score=" << score << " Diff=" << scoreDiff << "\n";
    }
    if (score > threshold && scoreDiff > threshold) {
      markers.push_back(frame);
      if (saveImages) {
        save(frame - 1, previousColor, "out");
        save(frame, currentColor, "in");
      }
    }

    lastScore = score;
    current.copyTo(previous);
    currentColor.copyTo(previousColor);
    frame++;
  }

  markers.push_back(frame - 1);
  if (saveImages) {
    save(frame - 1, previousColor, "out");
  }

  int segment = 0;
  for (size_t i = 1; i < markers.size(); i++) {
    int len = markers[i] - markers[i - 1];
    if (len >= minDuration) {
      segment++;
      int start = markers[i - 1];
      int end = markers[i];
      if (ffmpeg) {
        std::cout << "ffmpeg"
                  << " -ss " << timespec(start, fps, TS_INPUT)
                  << " -i \"" << inputFile << "\""
                  << " -ss " << timespec(start, fps, TS_OUTPUT)
                  << " -t " << timespec(end - start, fps, TS_DURATION)
                  << " -c copy"
                  << " -y"
                  << " " << segmentfile(inputFile, segment)
                  << "\n";
      } else {
        std::cout << segment << ": " << markers[i - 1] << " - " << markers[i] << "\n";
      }
    }
  }

  return 0;
}

void usage()
{
  std::cout << "usage: shotsegments --in <video-file>\n"
            << "                    [--save-images]\n"
            << "                    [--threshold t=" << DefaultThreshold << "]\n"
            << "                    [--min-duration d=" << DefaultMinDuration << "]\n"
            << "                    [--ffmpeg]\n"
            << "                    [--verbose[=level]]\n"
            << "                    [--help]\n"
            << "\n";
}

void save(int frame, cv::Mat image, const std::string & suffix)
{
  std::ostringstream ss;
  ss << std::setfill('0') << std::setw(8) << frame << "-" << suffix << ".jpg";
  cv::imwrite(ss.str(), image);
}

std::string timespec(int frame, double fps, int mode)
{
  int totalSeconds = frame / fps;
  switch (mode) {
  case TS_INPUT:
    totalSeconds -= totalSeconds % 60;
    break;
  case TS_OUTPUT:
    totalSeconds = totalSeconds % 60;
    break;
  case TS_DURATION:
    totalSeconds++;
    break;
  default:
    break;
  }
  int hours = totalSeconds / 60 / 60;
  int minutes = (totalSeconds / 60) % 60;
  int seconds = totalSeconds % 60;
  std::ostringstream ss;
  ss << std::setfill('0')
     << std::setw(2) << hours << ":"
     << std::setw(2) << minutes << ":"
     << std::setw(2) << seconds;
  return ss.str();
}

std::string segmentfile(const std::string & basis, int segment)
{
  size_t pos = basis.rfind('.');
  std::ostringstream ss;
  ss << basis.substr(0, pos) << "-" << segment << basis.substr(pos);
  return ss.str();
}
