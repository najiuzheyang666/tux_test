#include "pch.h"
#include <iostream>
#include <graphics.h>
#include <time.h> 
#include <math.h>
#include<opencv.hpp>
#include <opencv2/opencv.hpp>
#define Pi 3.14
#include <amp_math.h>
using namespace std;
using namespace cv;



void main___()
{
	

	Mat matSrc, matDst1, matDst2;
	matSrc = imread("C:\\Users\\qianyewu\\Desktop\\千夜舞\\33.png", 2 | 4);
	//matSrc = cv::imread("lena.jpg", 2 | 4);
	if (matSrc.empty()) return;

	int height = matSrc.rows, width = matSrc.cols;
	matDst1 = Mat(height,width, matSrc.type(), cv::Scalar::all(0));
	//matDst1 = cv::Mat(cv::Size(width, height), matSrc.type(), cv::Scalar::all(0));
	//matDst1 = cv::Mat(cv::Size(800, 1000), matSrc.type(), cv::Scalar::all(0));
	matDst2 = Mat(matSrc.size(), matSrc.type(), Scalar::all(0));

	double scale_x = (double)matSrc.cols / matDst1.cols;
	double scale_y = (double)matSrc.rows / matDst1.rows;


	//最近邻
	for (int i = 0; i < matDst1.cols; ++i)
	{
		int sx = cvFloor(i * scale_x);
		sx = min(sx, matSrc.cols - 1);
		for (int j = 0; j < matDst1.rows; ++j)
		{
			int sy = cvFloor(j * scale_y);
			sy = min(sy, matSrc.rows - 1);
			//多通道的图像可以直接赋值，不必每个通道赋值。但是要注意其类型是Vec3b，如果写成uchar，最后的copy图像只会显示源图像的1/3
			//matDst1.at<uchar>(j, i) = matSrc.at<uchar>(sy, sx);
			matDst1.at<cv::Vec3b>(j, i) = matSrc.at<cv::Vec3b>(sy, sx);
		}
	}
	//cv::imwrite("nearest_1.jpg", matDst1);
	cv::resize(matSrc, matDst2, matDst1.size(), 0, 0, 0);
	//cv::imwrite("nearest_2.jpg", matDst2);
	imshow("matSrc", matSrc);
	imshow("matDst1", matDst1);
	imshow("matDst2", matDst2);
	waitKey(0);
	system("pause");

	

	/*
	//双线性：由相邻的四像素(2*2)计算得出
	uchar* dataDst = matDst1.data;
	int stepDst = matDst1.step;
	uchar* dataSrc = matSrc.data;
	int stepSrc = matSrc.step;
	int iWidthSrc = matSrc.cols;
	int iHiehgtSrc = matSrc.rows;

	for (int j = 0; j < matDst1.rows; ++j)
	{
		float fy = (float)((j + 0.5) * scale_y - 0.5);
		int sy = cvFloor(fy);
		fy -= sy;
		sy = std::min(sy, iHiehgtSrc - 2);
		sy = std::max(0, sy);

		short cbufy[2];
		cbufy[0] = cv::saturate_cast<short>((1.f - fy) * 2048);
		cbufy[1] = 2048 - cbufy[0];

		for (int i = 0; i < matDst1.cols; ++i)
		{
			float fx = (float)((i + 0.5) * scale_x - 0.5);
			int sx = cvFloor(fx);
			fx -= sx;

			if (sx < 0) {
				fx = 0, sx = 0;
			}
			if (sx >= iWidthSrc - 1) {
				fx = 0, sx = iWidthSrc - 2;
			}

			short cbufx[2];
			cbufx[0] = cv::saturate_cast<short>((1.f - fx) * 2048);
			cbufx[1] = 2048 - cbufx[0];

			for (int k = 0; k < matSrc.channels(); ++k)
			{
				*(dataDst + j * stepDst + 3 * i + k) = (*(dataSrc + sy * stepSrc + 3 * sx + k) * cbufx[0] * cbufy[0] +
					*(dataSrc + (sy + 1)*stepSrc + 3 * sx + k) * cbufx[0] * cbufy[1] +
					*(dataSrc + sy * stepSrc + 3 * (sx + 1) + k) * cbufx[1] * cbufy[0] +
					*(dataSrc + (sy + 1)*stepSrc + 3 * (sx + 1) + k) * cbufx[1] * cbufy[1]) >> 22;
			}
		}
	}
	//cv::imwrite("linear_1.jpg", matDst1);
	cv::resize(matSrc, matDst2, matDst1.size(), 0, 0, 1);
	//cv::imwrite("linear_2.jpg", matDst2);

	imshow("matSrc", matSrc);
	imshow("matDst1", matDst1);
	imshow("matDst2", matDst2);
	waitKey(0);
	system("pause");
	*/
	}