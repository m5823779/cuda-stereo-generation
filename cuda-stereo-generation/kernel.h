#include <Windows.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace cv::cuda;

void PixelShifting(PtrStep<unsigned char> src, PtrStep<unsigned char> depth, PtrStep<signed short> dst, int height, int width, int channels);
void ImagePainting(PtrStep<signed short> img, int height, int width, int channels);
void ImageConcate(PtrStep<unsigned char> src1, PtrStep<unsigned char> src2, PtrStep<unsigned char> dst, int height, int width, int channels);
