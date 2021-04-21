#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.h"

__global__ void CUDA_PixelShifting(PtrStep<unsigned char> src, PtrStep<unsigned char> depth, PtrStep<signed short> dst,
	int rows, int cols, int channels) {
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	if (col < cols && row < rows) {
		int src_offset = (row * src.step + channels * col);
		int dst_offset = (row * dst.step / sizeof(signed short) + channels * col);
		int depth_offset = (row * depth.step + col);

		int dis = (int)(depth[depth_offset] * 70. / 255.);
		//int dis = (int)(depth[depth_offset] * 70.);

		if (col > dis) {
			dst[dst_offset - (dis * channels) + 0] = src[src_offset + 0];
			dst[dst_offset - (dis * channels) + 1] = src[src_offset + 1];
			dst[dst_offset - (dis * channels) + 2] = src[src_offset + 2];
		}
	}
}

__global__ void CUDA_ImagePainting(PtrStep<signed short> img, int rows, int cols, int channels) {
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	if (col < cols && row < rows) {
		int rgb_offset = (row * img.step / sizeof(signed short) + channels * col);
		if (img[rgb_offset + 0] == -1 && img[rgb_offset + 1] == -1 && img[rgb_offset + 2] == -1) {
			for (int offset = 1; offset < 70; offset++) {

				if (col - offset >= 0 && img[rgb_offset - (offset * channels) + 0] != -1 && img[rgb_offset - (offset * channels) + 1] != -1 && img[rgb_offset - (offset * channels) + 2] != -1) {
					img[rgb_offset + 0] = img[rgb_offset - (offset * channels) + 0];
					img[rgb_offset + 1] = img[rgb_offset - (offset * channels) + 1];
					img[rgb_offset + 2] = img[rgb_offset - (offset * channels) + 2];
					break;
				}
				if (col + offset <= cols && img[rgb_offset + (offset * channels) + 0] != -1 && img[rgb_offset + (offset * channels) + 1] != -1 && img[rgb_offset + (offset * channels) + 2] != -1) {
					img[rgb_offset + 0] = img[rgb_offset + (offset * channels) + 0];
					img[rgb_offset + 1] = img[rgb_offset + (offset * channels) + 1];
					img[rgb_offset + 2] = img[rgb_offset + (offset * channels) + 2];
					break;
				}
			}
		}
	}
}

__global__ void CUDA_Concate(PtrStep<unsigned char> src1, PtrStep<unsigned char> src2, PtrStep<unsigned char> dst,
	int rows, int cols, int channels) {
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	if (col < cols && row < rows && col % 2 == 0) {
		int src_rgb_offset = (row * src1.step + channels * col);

		if (col <= cols) {
			int dst_rgb_offset = (row * src1.step + channels * col / 2);
			dst[dst_rgb_offset + 0] = src1[src_rgb_offset + 0];
			dst[dst_rgb_offset + 1] = src1[src_rgb_offset + 1];
			dst[dst_rgb_offset + 2] = src1[src_rgb_offset + 2];

			dst_rgb_offset += channels * (int)ceil(cols / 2.);
			dst[dst_rgb_offset + 0] = src2[src_rgb_offset + 0];
			dst[dst_rgb_offset + 1] = src2[src_rgb_offset + 1];
			dst[dst_rgb_offset + 2] = src2[src_rgb_offset + 2];
		}
	}
}

void PixelShifting(PtrStep<unsigned char> src, PtrStep<unsigned char> depth, PtrStep<signed short> dst,
	int height, int width, int channels) {
	const dim3 dimGrid((int)ceil(width / 16.), (int)ceil(height / 16.));
	const dim3 dimBlock(16, 16);
	CUDA_PixelShifting << <dimGrid, dimBlock >> > (src, depth, dst, height, width, channels);
}

void ImagePainting(PtrStep<signed short> img, int height, int width, int channels) {
	const dim3 dimGrid((int)ceil(width / 16.), (int)ceil(height / 16.));
	const dim3 dimBlock(16, 16);
	CUDA_ImagePainting << <dimGrid, dimBlock >> > (img, height, width, channels);
}

void ImageConcate(PtrStep<unsigned char> src1, PtrStep<unsigned char> src2, PtrStep<unsigned char> dst, int height, int width,
	int channels) {
	const dim3 dimGrid((int)ceil(width / 16.), (int)ceil(height / 16.));
	const dim3 dimBlock(16, 16);
	CUDA_Concate << <dimGrid, dimBlock >> > (src1, src2, dst, height, width, channels);
}
