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

//����1 ����ڲ�ֵ��˫���Բ�ֵ��ʵ��
/*
% INTERPOLATE ��ֵ����
% f ԭͼ szͼ���С
% m ԭͼy��������
% n ԭͼx��������
% ex x���������������
% ey x���������������
% way 1Ϊ����ڲ�ֵ��2Ϊ˫���Բ�ֵ����
*/
inline int Interpolate_qyw(Mat f, int sz[], int m, int n, float ex, float ey, int way) {
	int gray = 0;
	float fr1, fr2, fr3;
	//1. ���ͳһ��0��1֮��
	if (ex < 0) {
		ex = 1 + ex;
		n--;
	}
	if (ey < 0) {
		ey = 1 + ey;
		m--;
	}
	if (m <= 0 || n <= 0)
	{
		//cout << "**gray1**" << endl;
		return gray;
	}
	//2. ���ڽ���ֵ
	if (way == 1) {
		//cout << "���ڽ���ֵ" << endl;
		if (ex > 0.5)
			n++;
		if (ey > 0.5)
			m++;
		if (m > sz[0] || n > sz[1])
		{
			//cout << m << " "<<sz[0] <<" "<< n << " " << sz[1] << endl;
			cout << "**gray2**" << endl;
			return gray;
		}
		//gray = f[sz[1] * m + n];
		gray = int(f.ptr<uchar>(m)[n]);
		//cout << "*****" << endl;
		return gray;
	}

	// 3.˫���Բ�ֵ
/*
	if (((m + 1) > sz[0]) || ((n + 1) > sz[1]))
		return gray;
	if (way == 2) {
		cout << "˫���Բ�ֵ" << endl;
		fr1 = (1 - ex)*float(f[sz[1] * m + n]) + ex * float(f[sz[1] * m + n + 1]);
		fr2 = (1 - ex)*float(f[sz[1] * (m + 1) + n]) + ex * float(f[sz[1] * (m + 1) + n + 1]);
		fr3 = (1 - ey)*fr1 + ey * fr2;
		gray = BYTE(fr3);
	}
	return gray;
	*/
}
int getPixel_qyw(int x, int y, Mat mat)
{
	return int(mat.ptr<uchar>(x)[y]);
}
//����2 ԭʼ�ķ���ӳ���㷨
/*
Mat normalRoate_qyw(BYTE img[], int w, int h, double theta, int *neww, int *newh) {
	float fsin, fcos, c1, c2, fx, fy, ex, ey;
	int w1, h1, xx, yy;
	int sz[2] = { h,w };
	//1. �����������
	fsin = sin(theta);
	fcos = cos(theta);
	*newh = h1 = ceilf(abs(h*fcos) + abs(w*fsin));
	*neww = w1 = ceilf(abs(w*fcos) + abs(h*fsin));
	Mat I1 = new(std::c) BYTE[w1*h1];
	if (!I1)
		return NULL;
	memset(I1, 0, w1*h1);
	c1 = (w - w1 * fcos - h1 * fsin) / 2;
	c2 = (h + w1 * fsin - h1 * fcos) / 2;
	//2. ���㷴�����겢�����ֵ
	for (int y = 0; y < h1; y++) {
		for (int x = 0; x < w1; x++) {
			//�������ӳ���ľ�ȷλ�� ÿ���㶼ʹ��ԭʼ��ʽ����
			fx = x * fcos + y * fsin + c1; //�Ĵθ���˷����Ĵθ���ӷ�
			fy = y * fcos - x * fsin + c2;
			xx = roundf(fx);
			yy = roundf(fy);
			ex = fx - float(xx);
			ey = fy - float(yy);
			//I1[w1*y + x] = Interpolate(img, sz, yy, xx, ex, ey, 2);//˫���Բ�ֵ
			//I1[w1*y + x] = Interpolate_qyw(img, sz, yy, xx, ex, ey, 1);//���ڽ���ֵ
		}
	}
	return I1;
}
*/
//����3 ����ֱ���㷨��ͼ����ת
Mat DDARoateFast_qyw(Mat img, int w, int h, double theta, int *neww, int *newh) {
	float fsin, fcos, c1, c2, fx, fy, ex, ey;
	int w1, h1, xx, yy;
	int sz[2] = { h, w };
	//1. ������ת����
	fsin = sin(theta);
	fcos = cos(theta);
	*newh = h1 = ceilf(abs(h*fcos) + abs(w*fsin));
	*neww = w1 = ceilf(abs(w*fcos) + abs(h*fsin));
	//auto I1 = new(std::nothrow) BYTE[w1*h1];
	Mat dst(h1, w1, CV_8UC1);//����һ����height����width�ĻҶ�ͼ��Mat����

	c1 = (w - w1 * fcos - h1 * fsin) / 2; //������[1]�Ĺ�ʽ
	c2 = (h + w1 * fsin - h1 * fcos) / 2;
	fx = c1 - fsin;
	fy = c2 - fcos;
	//2. ���㷴�����겢�����ֵ
	for (int y = 0; y < h1; y++) {//��������ѭ���м��������ʱ��û�и���˷�����
		//�����һ������ӳ���ľ�ȷλ��
		fx = fx + fsin;
		fy = fy + fcos;
		// �����һ�����Ӧ��դ��λ��
		xx = roundf(fx);
		//ԭ����
		//yy = roundF(fy);
		yy = roundf(fy);

		ex = fx - float(xx);//���
		ey = fy - float(yy);
		for (int x = 0; x < w1; x++) {
			//ԭ����
			//I1[w1*y + x] = InterpolateFast(img, sz, yy, xx, ex, ey, 2);
			//I1[w1*y + x] = Interpolate(img, sz, yy, xx, ex, ey, 2);
			//I1[w1*y + x] = Interpolate_qyw(img, sz, yy, xx, ex, ey, 1);//���ڽ�
			ex = ex + fcos;
			ey = ey - fsin;
			if (ex > 0.5) {
				xx++;
				ex = ex - 1;
			}
			if (ey < -0.5) {
				yy--;
				ey = ey + 1;
			}
		}
	}
	return dst;
}

//����4 ǰ��ӳ���opencvʵ��
void FastRotateImage_qyw(Mat &srcImg, Mat &roateImg, float degree) {
	imshow("src", srcImg);
	cout << "ǰ��ӳ���opencvʵ��" << endl;
	assert((srcImg.cols > 0) && (srcImg.rows > 0));
	float fsin, fcos, c1, c2, fx, fy, xx, yy;
	int w1, h1, w, h;
	w = srcImg.cols;
	h = srcImg.rows;
	int sz[2] = { srcImg.rows, srcImg.cols };
	Mat map1_x, map2_y, m1, m2, m3, sPoint, newMap; //m1 m2 m3 Ϊǰ���Ƽ������еĻ�������
	//1. ������ת����
	fsin = sin(degree);
	fcos = cos(degree);
	h1 = ceilf(abs(h*fcos) + abs(w*fsin));
	w1 = ceilf(abs(w*fcos) + abs(h*fsin));
	roateImg.create(h1, w1, CV_8UC1); //srcImg.type
	map1_x.create(srcImg.size(), CV_32FC1);
	map2_y.create(srcImg.size(), CV_32FC1);
	c1 = (w - w1 * fcos - h1 * fsin) / 2;
	c2 = (h + w1 * fsin - h1 * fcos) / 2;
	//2. ����ǰ��ĵ�һ��������
	m1 = Mat::eye(3, 3, CV_32FC1);
	m1.at<float>(2, 0) = -w / 2;
	m1.at<float>(2, 1) = h / 2;
	m1.at<float>(1, 1) = -1;
	m2 = Mat::eye(3, 3, CV_32FC1);
	m2.at<float>(0, 0) = fcos;
	m2.at<float>(0, 1) = -fsin;
	m2.at<float>(1, 0) = fsin;
	m2.at<float>(1, 1) = fcos;
	m3 = Mat::eye(3, 3, CV_32FC1);
	m3.at<float>(2, 0) = w1 / 2;
	m3.at<float>(2, 1) = h1 / 2;
	m3.at<float>(1, 1) = -1;
	sPoint = Mat::zeros(1, 3, CV_32FC1);
	sPoint.at<float>(0, 2) = 1;
	Mat snPoint = sPoint * m3*m2*m1;
	//cout << snPoint << endl;
	fx = snPoint.at<float>(0, 0) - fsin;
	fy = snPoint.at<float>(0, 1) - fcos;
	//3. ��ֱ�߻�������ʣ��������ӳ�������
	for (int y = 0; y < h; y++) {
		//�����һ��ǰ��ӳ���ľ�ȷλ��
		fx = fx + fsin;
		fy = fy + fcos;
		xx = fx - fcos;
		yy = fy + fsin;
		float *ptrx = map1_x.ptr<float>(y);
		float *ptry = map2_y.ptr<float>(y);
		for (int x = 0; x < w; x++) {
			xx = xx + fcos;
			yy = yy - fsin;
			*(ptrx++) = xx;
			*(ptry++) = yy;
		}
	}

	//3.����opencv����ӳ�亯�����ǰ��ӳ��Ĳ�ֵ����
	remap(srcImg, roateImg, map1_x, map2_y, INTER_CUBIC, BORDER_CONSTANT, Scalar(0, 0, 0));
	imshow("src1", srcImg);
	imshow("roateImg", roateImg);
}

//����5 opencv����תʵ��
//��ʱ����תͼ��degree�Ƕȣ�ԭ�ߴ磩  
void rotateImage_qyw(Mat &img, Mat &img_rotate, int degree)
{
	//��ת����Ϊͼ������  
	CvPoint2D32f center;
	center.x = float(img.cols / 2.0 + 0.5);
	center.y = float(img.rows / 2.0 + 0.5);
	//�����ά��ת�ķ���任����  
	Mat M(2, 3, CV_32FC1);
	M = getRotationMatrix2D(center, degree, 1);
	clock_t t1 = clock();
	//�任ͼ�񣬲��ú�ɫ�������ֵ  
	//warpAffine(img, img_rotate, M, img.size(), INTER_CUBIC);
	warpAffine(img, img_rotate, M, img.size(), INTER_NEAREST);//���ڽ�
	clock_t t2 = clock();
	cout << (t2 - t1)*1.0 / CLOCKS_PER_SEC / 10;
}


void main()
{
	Mat src_img;
	src_img = imread(R"(F:\C\tux_test\tux_test\gray_image_big.jpg)", 2 | 4);//��ͨ��
	//src_img = imread(R"(F:\C\tux_test\tux_test\gray_image_big.jpg)");//��ͨ��
	//cout << "tongdao:" << int (src_img.channels()) << endl;
	//imshow("matSrc", src_img);
	int height = src_img.rows, width = src_img.cols;
	int sz[2] = { height ,width };
	int i = Interpolate_qyw(src_img, sz, 100, 123, 0.1, 0.1, 1);
	int j = getPixel_qyw(100, 123, src_img);
	cout << j << "   " << i << endl;
	Mat mat1(height, width, CV_8UC1);
	Mat mat2(height, width, CV_8UC1);
	for (int i = 0; i < src_img.rows; i++)        //����ÿһ��ÿһ�в�����������ֵ
	{
		for (int j = 0; j < src_img.cols; j++)
		{

			//mat1.ptr<uchar>(i)[j] = int(src_img.ptr<uchar>(i)[j]);
			mat1.ptr<uchar>(i)[j] = getPixel_qyw(i,j, src_img);
			//mat2.ptr<uchar>(i)[j] = Interpolate_qyw(src_img, sz, i, j, 0.1, 0.1, 1);
		}
	}
	for (int i = 0; i < src_img.rows; i++)        //����ÿһ��ÿһ�в�����������ֵ
	{
		for (int j = 0; j < src_img.cols; j++)
		{

			//mat1.ptr<uchar>(i)[j] = int(src_img.ptr<uchar>(i)[j]);
			//mat1.ptr<uchar>(i)[j] = getPixel_qyw(i,j, src_img);
			mat2.ptr<uchar>(i)[j] = Interpolate_qyw(src_img, sz, i, j, 0.1, 0.5, 1);
		}
	}
	imshow("mat1", mat1);
	imshow("mat2", mat2);
	/*Mat matSrc, matDst1, matDst2;
	matSrc = imread("C:\\Users\\qianyewu\\Desktop\\ǧҹ��\\33.png", 2 | 4);
	//matSrc = cv::imread("lena.jpg", 2 | 4);
	if (matSrc.empty()) return;

	int height = matSrc.rows, width = matSrc.cols;
	matDst1 = Mat(height/2,width/2, matSrc.type(), cv::Scalar::all(0));
	//matDst1 = cv::Mat(cv::Size(width, height), matSrc.type(), cv::Scalar::all(0));
	//matDst1 = cv::Mat(cv::Size(800, 1000), matSrc.type(), cv::Scalar::all(0));
	matDst2 = Mat(matSrc.size(), matSrc.type(), Scalar::all(0));

	double scale_x = (double)matSrc.cols / matDst1.cols;
	double scale_y = (double)matSrc.rows / matDst1.rows;


	//�����
	for (int i = 0; i < matDst1.cols; ++i)
	{
		int sx = cvFloor(i * scale_x);
		sx = min(sx, matSrc.cols - 1);
		for (int j = 0; j < matDst1.rows; ++j)
		{
			int sy = cvFloor(j * scale_y);
			sy = min(sy, matSrc.rows - 1);
			//��ͨ����ͼ�����ֱ�Ӹ�ֵ������ÿ��ͨ����ֵ������Ҫע����������Vec3b�����д��uchar������copyͼ��ֻ����ʾԴͼ���1/3
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
	*/


	/*
	//˫���ԣ������ڵ�������(2*2)����ó�
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
	waitKey(0);
	system("pause");
}