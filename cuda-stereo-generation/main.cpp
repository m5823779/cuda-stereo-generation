#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "kernel.h"

int main() {
  float depth[10] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
  int depth_index = 0;

  while (true) {
    Mat raw_input = imread("../test1.jpg");
    int input_h = raw_input.rows;
    int input_w = raw_input.cols;
    Mat depth_map(input_h, input_w, CV_32FC1, cv::Scalar(depth[depth_index % 10]));
    Mat dst;
    
    // pixel shifting
    cv::cuda::GpuMat d_left_img(input_h, input_w, CV_8UC3);
    cv::cuda::GpuMat d_depth_map(input_h, input_w, CV_32FC1);
    cv::cuda::GpuMat d_stereo_img(input_h, input_w, CV_8UC3);
    
    d_left_img.upload(raw_input);
    d_depth_map.upload(depth_map);
    d_depth_map.convertTo(d_depth_map, CV_8UC1, 255.0 / 1.0, 0);

    cv::cuda::GpuMat d_right_img(input_h, input_w, CV_16SC3, cv::Scalar(-1, -1, -1));

    PixelShifting(d_left_img, d_depth_map, d_right_img, input_h, input_w,  d_left_img.channels());
    ImagePainting(d_right_img, input_h, input_w, d_left_img.channels());

    d_right_img.convertTo(d_right_img, CV_8UC3);
    ImageConcate(d_left_img, d_right_img, d_stereo_img, input_h, input_w, d_left_img.channels());
    
    d_stereo_img.download(dst);
    cv::imshow("Pixel shifting", dst);
    cv::waitKey(1);
    depth_index += 1;
  }
  return 0;
}


