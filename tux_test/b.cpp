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

//blog.csdn.net/xiaoheiblack/article/details/79026239 
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
inline int Interpolate_qyw(BYTE f[], int sz[], int m, int n, float ex, float ey, char way) {
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
		return gray;

	//2. ���ڽ���ֵ
	if (way == 1) {
		cout << "���ڽ���ֵ" << endl;
		if (ex > 0.5)
			n++;
		if (ey > 0.5)
			m++;
		if (m > sz[0] || n > sz[1])
			return gray;
		gray = f[sz[1] * m + n];
		return gray;
	}

	// 3.˫���Բ�ֵ

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
}
inline BYTE Interpolate(BYTE f[], int sz[], int m, int n, float ex, float ey, char way) {
	BYTE gray = 0;
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
		return gray;

	//2. ���ڽ���ֵ
	if (way == 1) {
		cout << "���ڽ���ֵ" << endl;
		if (ex > 0.5)
			n++;
		if (ey > 0.5)
			m++;
		if (m > sz[0] || n > sz[1])
			return gray;
		gray = f[sz[1] * m + n];
		return gray;
	}

	// 3.˫���Բ�ֵ
	
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
}
//����2 ԭʼ�ķ���ӳ���㷨

BYTE* normalRoate(BYTE img[], int w, int h, double theta, int *neww, int *newh) {
	float fsin, fcos, c1, c2, fx, fy, ex, ey;
	int w1, h1, xx, yy;
	int sz[2] = { h,w };
	//1. �����������
	fsin = sin(theta);
	fcos = cos(theta);
	*newh = h1 = ceilf(abs(h*fcos) + abs(w*fsin));
	*neww = w1 = ceilf(abs(w*fcos) + abs(h*fsin));
	auto I1 = new(std::nothrow) BYTE[w1*h1];
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
			I1[w1*y + x] = Interpolate(img, sz, yy, xx, ex, ey, 1);//���ڽ���ֵ
		}
	}
	return I1;
}

//����3 ����ֱ���㷨��ͼ����ת
BYTE* DDARoateFast(BYTE img[], int w, int h, double theta, int *neww, int *newh) {
	float fsin, fcos, c1, c2, fx, fy, ex, ey;
	int w1, h1, xx, yy;
	int sz[2] = { h, w };
	//1. ������ת����
	fsin = sin(theta);
	fcos = cos(theta);
	*newh = h1 = ceilf(abs(h*fcos) + abs(w*fsin));
	*neww = w1 = ceilf(abs(w*fcos) + abs(h*fsin));
	auto I1 = new(std::nothrow) BYTE[w1*h1];
	if (!I1)
		return NULL;
	memset(I1, 0, w1*h1);
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
			I1[w1*y + x] = Interpolate(img, sz, yy, xx, ex, ey, 1);//���ڽ�
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
	return I1;
}

//����4 ǰ��ӳ���opencvʵ��
void FastRotateImage(Mat &srcImg, Mat &roateImg, float degree) {
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
void rotateImage(Mat &img, Mat &img_rotate, int degree)
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

int main33()
{

	Mat img = imread(R"(C:\Users\qianyewu\Desktop\ǧҹ��\33.png)");
	if (img.empty())
	{
		cerr << "can not load image" << endl;
		return 0;
	}

	// �������任�ģ����ģ��Ƕȣ��߶�
	Point2f center;
	center = Point2f(0, 0);
	double degree = 30;
	double scale = 1;

	// ��ת 30 �ȵ�����
	Mat rot = getRotationMatrix2D(center, degree, scale);
	/*��Ҫ���ڻ��ͼ������ ĳһ�����ת���� 
Mat getRotationMatrix2D(Point2f center, double angle, double scale)
������⣺
Point2f center����ʾ��ת�����ĵ�
double angle����ʾ��ת�ĽǶ�
double scale��ͼ����������*/
	Mat rimg;
	warpAffine(img, rimg, rot, img.size());
	imshow("30", rimg);

	// �������任�ģ����ģ��Ƕȣ��߶�
	// ������ ע�⣬����任֮ǰ�����ı���Ϊ ԭͼ������ ������
	center = Point2f(img.cols / 2.0, img.rows / 2.0);
	degree = 10;
	scale = 1;

	// ��ȡ�任����
	rot = getRotationMatrix2D(center, degree, scale);
	rimg;
	warpAffine(img, rimg, rot, img.size());
	imshow("img", img);
	imshow("rimg", rimg);

	// ��ȡ�任֮��� �����������Ҫ����Ȼ�Ļ����任֮���ͼ����ʾ��ȫ
	Rect bbox;
	bbox = RotatedRect(center, Size(scale*img.cols, scale*img.rows), degree).boundingRect();

	// �Ա任��������һ�����޸ģ����¶���任�� ����

	rot.at<double>(0, 2) += bbox.width / 2 - center.x;
	rot.at<double>(1, 2) += bbox.height / 2 - center.y;


	Mat dst;
	warpAffine(img, dst, rot, bbox.size());
	imshow("dst", dst);

	waitKey();
	system("pause");
	return 0;
}

int main()
{
	Mat img, mat, mat_rotate_a;
	//resize;
	Mat src_img, dst, gray_img;
	src_img = imread(R"(F:\C\tux_test\tux_test\gray_image_big.jpg)");

	normalRoate((BYTE)src_img, 10, 10, 10, &src_img, &src_img)

	//FastRotateImage(src_img, src_img, 30.0);

	//rotateImage(src_img,dst,30);
	
	imshow("src", src_img);
	imshow("dst", dst);
	waitKey(0);
	system("pause");
	return 0;
}

void main0000()
{
	cv::Mat matSrc = cv::imread("C:\\Users\\qianyewu\\Desktop\\ǧҹ��\\33.png", 2 | 4);

	if (matSrc.empty()) return;

	const double degree = 45;
	double angle = degree * CV_PI / 180.;
	double alpha = cos(angle);
	double beta = sin(angle);
	int iWidth = matSrc.cols;
	int iHeight = matSrc.rows;
	int iNewWidth = cvRound(iWidth * fabs(alpha) + iHeight * fabs(beta));
	int iNewHeight = cvRound(iHeight * fabs(alpha) + iWidth * fabs(beta));

	double m[6];
	m[0] = alpha;
	m[1] = beta;
	m[2] = (1 - alpha) * iWidth / 2. - beta * iHeight / 2.;
	m[3] = -m[1];
	m[4] = m[0];
	m[5] = beta * iWidth / 2. + (1 - alpha) * iHeight / 2.;

	cv::Mat M = cv::Mat(2, 3, CV_64F, m);
	cv::Mat matDst1 = cv::Mat(cv::Size(iNewWidth, iNewHeight), matSrc.type(), cv::Scalar::all(0));

	double D = m[0] * m[4] - m[1] * m[3];
	D = D != 0 ? 1. / D : 0;
	double A11 = m[4] * D, A22 = m[0] * D;
	m[0] = A11; m[1] *= -D;
	m[3] *= -D; m[4] = A22;
	double b1 = -m[0] * m[2] - m[1] * m[5];
	double b2 = -m[3] * m[2] - m[4] * m[5];
	m[2] = b1; m[5] = b2;

	int round_delta = 512;//nearest
	for (int y = 0; y < iNewHeight; ++y)
	{
		for (int x = 0; x < iNewWidth; ++x)
		{
			//int tmpx = cvFloor(m[0] * x + m[1] * y + m[2]);
			//int tmpy = cvFloor(m[3] * x + m[4] * y + m[5]);
			int adelta = cv::saturate_cast<int>(m[0] * x * 1024);
			int bdelta = cv::saturate_cast<int>(m[3] * x * 1024);
			int X0 = cv::saturate_cast<int>((m[1] * y + m[2]) * 1024) + round_delta;
			int Y0 = cv::saturate_cast<int>((m[4] * y + m[5]) * 1024) + round_delta;
			int X = (X0 + adelta) >> 10;
			int Y = (Y0 + bdelta) >> 10;

			if ((unsigned)X < iWidth && (unsigned)Y < iHeight)
			{
				matDst1.at<cv::Vec3b>(y, x) = matSrc.at<cv::Vec3b>(Y, X);
			}
		}
	}
	cv::imwrite("rotate_nearest_1.jpg", matDst1);

	M = cv::getRotationMatrix2D(cv::Point2f(iWidth / 2., iHeight / 2.), degree, 1);

	cv::Mat matDst2;
	cv::warpAffine(matSrc, matDst2, M, cv::Size(iNewWidth, iNewHeight), 0, 0, 0);
	cv::imwrite("rotate_nearest_2.jpg", matDst2);
	
	/*
	cv::Mat matSrc = cv::imread("C:\\Users\\qianyewu\\Desktop\\ǧҹ��\\33.png", 2 | 4);
	if (matSrc.empty()) return;

	const double degree = 45;
	double angle = degree * CV_PI / 180.;
	double alpha = cos(angle);
	double beta = sin(angle);
	int iWidth = matSrc.cols;
	int iHeight = matSrc.rows;
	int iNewWidth = cvRound(iWidth * fabs(alpha) + iHeight * fabs(beta));
	int iNewHeight = cvRound(iHeight * fabs(alpha) + iWidth * fabs(beta));

	double m[6];
	m[0] = alpha;
	m[1] = beta;
	m[2] = (1 - alpha) * iWidth / 2. - beta * iHeight / 2.;
	m[3] = -m[1];
	m[4] = m[0];
	m[5] = beta * iWidth / 2. + (1 - alpha) * iHeight / 2.;

	cv::Mat M = cv::Mat(2, 3, CV_64F, m);
	cv::Mat matDst1 = cv::Mat(cv::Size(iNewWidth, iNewHeight), matSrc.type(), cv::Scalar::all(0));

	double D = m[0] * m[4] - m[1] * m[3];
	D = D != 0 ? 1. / D : 0;
	double A11 = m[4] * D, A22 = m[0] * D;
	m[0] = A11; m[1] *= -D;
	m[3] *= -D; m[4] = A22;
	double b1 = -m[0] * m[2] - m[1] * m[5];
	double b2 = -m[3] * m[2] - m[4] * m[5];
	m[2] = b1; m[5] = b2;

	for (int y = 0; y < iNewHeight; ++y)
	{
		for (int x = 0; x < iNewWidth; ++x)
		{
			//int tmpx = cvFloor(m[0] * x + m[1] * y + m[2]);
			//int tmpy = cvFloor(m[3] * x + m[4] * y + m[5]);
			float fx = m[0] * x + m[1] * y + m[2];
			float fy = m[3] * x + m[4] * y + m[5];

			int sy = cvFloor(fy);
			fy -= sy;
			//sy = std::min(sy, iHeight-2);
			//sy = std::max(0, sy);
			if (sy < 0 || sy >= iHeight) continue;

			short cbufy[2];
			cbufy[0] = cv::saturate_cast<short>((1.f - fy) * 2048);
			cbufy[1] = 2048 - cbufy[0];

			int sx = cvFloor(fx);
			fx -= sx;
			//if (sx < 0) {
			//	fx = 0, sx = 0;
			//}
			//if (sx >= iWidth - 1) {
			//	fx = 0, sx = iWidth - 2;
			//}
			if (sx < 0 || sx >= iWidth) continue;

			short cbufx[2];
			cbufx[0] = cv::saturate_cast<short>((1.f - fx) * 2048);
			cbufx[1] = 2048 - cbufx[0];

			for (int k = 0; k < matSrc.channels(); ++k)
			{
				if (sy == iHeight - 1 || sx == iWidth - 1) {
					continue;
				}
				else {
					matDst1.at<cv::Vec3b>(y, x)[k] = (matSrc.at<cv::Vec3b>(sy, sx)[k] * cbufx[0] * cbufy[0] +
						matSrc.at<cv::Vec3b>(sy + 1, sx)[k] * cbufx[0] * cbufy[1] +
						matSrc.at<cv::Vec3b>(sy, sx + 1)[k] * cbufx[1] * cbufy[0] +
						matSrc.at<cv::Vec3b>(sy + 1, sx + 1)[k] * cbufx[1] * cbufy[1]) >> 22;
				}
			}
		}
	}
	cv::imwrite("rotate_bilinear_11.jpg", matDst1);

	M = cv::getRotationMatrix2D(cv::Point2f(iWidth / 2., iHeight / 2.), degree, 1);

	cv::Mat matDst2;
	cv::warpAffine(matSrc, matDst2, M, cv::Size(iNewWidth, iNewHeight), 1, 0, 0);
	cv::imwrite("rotate_bilinear_22.jpg", matDst2);
	*/
}