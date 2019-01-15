#include "pch.h"

using namespace std;
using namespace cv;

//��ʱ����ת��pdst�����ģ�������0���
//pSrc,srcW,srcHԭͼ����ߴ�
//pDst,dstW,dstH��ת��ͼ����ߴ�
//��ת�Ƕ�
//ͨ����
void myImgRotate(unsigned char* pSrc, int srcW, int srcH,
	unsigned char* pDst, int dstW, int dstH,
	double degree, int nchannel)
{

	int k;
	double angle = degree * 3.1415926 / 180.;	//��ת�Ƕ�
	double co = cos(angle);	//����
	double si = sin(angle);	//����
	int rotateW, rotateH;	//��ת��ͼ��ĸ߿�
	int srcWidthStep = srcW * nchannel;//��Ȳ���
	int dstWisthStep = dstW * nchannel;
	int x, y;
	int xMin, xMax, yMin, yMax;
	int xOff, yOff;	//ƫ��
	double xSrc = 0.;
	double ySrc = 0.;	//�任��ͼ���������ԭͼ�е�����

	//��ʱ����
	float valueTemp = 0.;
	float a1, a2, a3, a4;

	memset(pDst, 0, dstWisthStep*dstH * sizeof(unsigned char));
	//������ת������귶Χ
	rotateH = srcW * fabs(si) + srcH * fabs(co);
	rotateW = srcW * fabs(co) + srcH * fabs(si);

	//����ƫ��
	xOff = dstW / 2;
	yOff = dstH / 2;

	yMin = (dstH - rotateH) / 2.;
	yMax = yMin + rotateH + 1;	//��1
	xMin = (dstW - rotateW) / 2.;
	xMax = xMin + rotateW + 1;

	for (y = yMin; y <= yMax; y++)
	{
		for (x = xMin; x <= xMax; x++)
		{
			//��ȡ��ԭͼ�е�����
			ySrc = si * double(x - xOff) + co * double(y - yOff) + double(int(srcH / 2));
			xSrc = co * double(x - xOff) - si * double(y - yOff) + double(int(srcW / 2));

			//�����ԭͼ��Χ��
			if (ySrc >= 0. && ySrc < srcH - 0.5 && xSrc >= 0. && xSrc < srcW - 0.5)
			{
				//��ֵ
				int xSmall = floor(xSrc);
				int xBig = ceil(xSrc);
				int ySmall = floor(ySrc);
				int yBig = ceil(ySrc);

				for (k = 0; k < nchannel; k++)
				{
					a1 = (xSmall >= 0 && ySmall >= 0 ? pSrc[ySmall*srcWidthStep + xSmall * nchannel + k] : 0);
					a2 = (xBig < srcW && ySmall >= 0 ? pSrc[ySmall*srcWidthStep + xBig * nchannel + k] : 0);
					a3 = (xSmall >= 0 && yBig < srcH ? pSrc[yBig*srcWidthStep + xSmall * nchannel + k] : 0);
					a4 = (xBig < srcW && yBig < srcH ? pSrc[yBig*srcWidthStep + xBig * nchannel + k] : 0);
					double ux = xSrc - xSmall;
					double uy = ySrc - ySmall;
					//˫���Բ�ֵ
					valueTemp = (1 - ux)*(1 - uy)*a1 + (1 - ux)*uy*a3 + (1 - uy)*ux*a2 + ux * uy*a4;
					pDst[y*dstWisthStep + x * nchannel + k] = floor(valueTemp);
				}

			}
		}
	}
}
/*
void main(int argc, char** argv)
{
	IplImage* iplOrg = cvLoadImage(argv[1]);	//����ͼ��
	unsigned char* pColorImg = NULL;
	int width = iplOrg->width;
	int height = iplOrg->height;
	pColorImg = (unsigned char*)malloc(width*height * 3 * sizeof(unsigned char));

	cvCvtColor(iplOrg, iplOrg, CV_BGR2RGB);
	IplToUchar(iplOrg, pColorImg);	//�������ʾͼ��
	cvReleaseImage(&iplOrg);

	double degree = 15;	//��ʱ����ת�Ƕ�0~180

	int tempLength = sqrt((double)width * width + (double)height *height) + 10;//��֤ԭͼ��������Ƕ���ת����С�ߴ�
	unsigned char* pTemp = (unsigned char*)malloc(tempLength*tempLength * 3 * sizeof(unsigned char));


	//��ת
	myImgRotate(pColorImg, width, height, pTemp, tempLength, tempLength, degree, 3);
	DisplayPicture(tempLength, tempLength, pTemp, "rotate.bmp", 3);	//����ͼ��

	free(pTemp);
	pTemp = NULL;
	free(pColorImg);
	pColorImg = NULL;

}
*/


//��תͼ�����ݲ��䣬�ߴ���Ӧ���
IplImage* rotateImage1(IplImage* img, int degree) {
	double angle = degree * CV_PI / 180.; // ����  
	double a = sin(angle), b = cos(angle);
	int width = img->width;
	int height = img->height;
	int width_rotate = int(height * fabs(a) + width * fabs(b));
	int height_rotate = int(width * fabs(a) + height * fabs(b));
	//��ת����map
	// [ m0  m1  m2 ] ===>  [ A11  A12   b1 ]
	// [ m3  m4  m5 ] ===>  [ A21  A22   b2 ]
	float map[6];
	CvMat map_matrix = cvMat(2, 3, CV_32F, map);
	// ��ת����
	CvPoint2D32f center = cvPoint2D32f(width / 2, height / 2);
	cv2DRotationMatrix(center, degree, 1.0, &map_matrix);
	map[2] += (width_rotate - width) / 2;
	map[5] += (height_rotate - height) / 2;
	IplImage* img_rotate = cvCreateImage(cvSize(width_rotate, height_rotate), 8, 3);
	//��ͼ��������任
	//CV_WARP_FILL_OUTLIERS - ����������ͼ������ء�
	//�������������������ͼ��ı߽��⣬��ô���ǵ�ֵ�趨Ϊ fillval.
	//CV_WARP_INVERSE_MAP - ָ�� map_matrix �����ͼ������ͼ��ķ��任��
	cvWarpAffine(img, img_rotate, &map_matrix, CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS, cvScalarAll(0));
	return img_rotate;
}


IplImage* rotateImage2(IplImage* img, int degree)
{
	double angle = degree * CV_PI / 180.;
	double a = sin(angle), b = cos(angle);
	int width = img->width, height = img->height;
	//��ת�����ͼ�ߴ�
	int width_rotate = int(height * fabs(a) + width * fabs(b));
	int height_rotate = int(width * fabs(a) + height * fabs(b));
	IplImage* img_rotate = cvCreateImage(cvSize(width_rotate, height_rotate), img->depth, img->nChannels);
	cvZero(img_rotate);
	//��֤ԭͼ��������Ƕ���ת����С�ߴ�
	int tempLength = sqrt((double)width * width + (double)height *height) + 10;
	int tempX = (tempLength + 1) / 2 - width / 2;
	int tempY = (tempLength + 1) / 2 - height / 2;
	IplImage* temp = cvCreateImage(cvSize(tempLength, tempLength), img->depth, img->nChannels);
	cvZero(temp);
	//��ԭͼ���Ƶ���ʱͼ��tmp����
	cvSetImageROI(temp, cvRect(tempX, tempY, width, height));
	cvCopy(img, temp, NULL);
	cvResetImageROI(temp);
	//��ת����map
	// [ m0  m1  m2 ] ===>  [ A11  A12   b1 ]
	// [ m3  m4  m5 ] ===>  [ A21  A22   b2 ]
	float m[6];
	int w = temp->width;
	int h = temp->height;
	m[0] = b;
	m[1] = a;
	m[3] = -m[1];
	m[4] = m[0];
	// ����ת��������ͼ���м�  
	m[2] = w * 0.5f;
	m[5] = h * 0.5f;
	CvMat M = cvMat(2, 3, CV_32F, m);
	cvGetQuadrangleSubPix(temp, img_rotate, &M);
	cvReleaseImage(&temp);
	return img_rotate;
}

void Bresenham_1(int x0, int y0, int x1, int y1, Mat mat) {

	int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
	int dy = abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
	int err = (dx > dy ? dx : -dy) / 2, e2;

	for (;;) {
		//putpixel(x0, y0, COLOR);
		mat.ptr<uchar>(x0)[y0] = 0;

		if (x0 == x1 && y0 == y1) break;
		e2 = err;
		if (e2 > -dx) { err -= dy; x0 += sx; }
		if (e2 < dy) { err += dx; y0 += sy; }
	}
}
int sign(int  x)
{
	/*if (x < 0) return -1;
	else if (x == 0)return 0;
	else return 1;*/
	return x < 0 ? -1 : x == 0 ? 0 : 1;
}
/*
��ͼƬ�Ŀ�С������x2ʱ����Ҫ���¿���
*/
void Bresenham(int x1, int y1, int x2, int y2, Mat mat)
{
	int x, y, s1, s2, increx, increy, temp, interchange, e, i;
	x = x1;
	y = y1;
	s1 = sign(x2 - x1);
	s2 = sign(y2 - y1);
	increx = abs(x2 - x1);
	increy = abs(y2 - y1);
	if (increy > increx)
	{
		temp = increx;
		increx = increy;
		increy = temp;
		interchange = 1;
	}
	else
		interchange = 0;
	e = 2 * increy - increx;
	for (i = 1; i <= increx; i++)
	{
		//putpixel(x, y, COLOR);
		mat.ptr<uchar>(x)[y] = 33;
		//cout << x << " . " << y<<endl;
		if (e >= 0)
		{
			if (interchange = 1)
				x = x + s1;
			else
				y = y + s2;
			e = e - 2 * increx;
		}
		if (interchange = 1)
			y = y + s2;
		else
			x = x + s1;
		e = e + 2 * increy;
	}
}

int getPixel(int x, int y, Mat mat)
{
	return int(mat.ptr<uchar>(x)[y]);
}
Mat creatImage(int height, int width)
{
	Mat mat(height, width, CV_8UC1);//����һ����height����width�ĻҶ�ͼ��Mat����
	return mat;
}

//˳ʱ������ʱ�벻һ��
int getX(int x, int y, double sina, double cosa)
{
	//return (x * cosa + y * sina);
	double xx = x * cosa - y * sina;
	//cout << "getX:" << xx << "\n" << endl;
	if (xx >= 0)
		return int(xx + 0.5);
	else
		return int(xx - 0.5);
}
int getY(int x, int y, double sina, double cosa)
{
	//return (y*cosa - x * sina);
	double yy = y * cosa + x * sina;
	//cout << "getY:" << yy << "\n" << endl;
	if (yy >= 0)
		return int(yy + 0.5);
	else
		return int(yy - 0.5);
}
//ͼ�����ꡪ��>һ������ϵ
int im2maX(int x, int width)
{
	return int(x - 0.5*width + 0.5);
}
int im2maY(int y, int height)
{
	return  int(-y + 0.5*height - 0.5);
}
//һ������ϵ����>ͼ������
int ma2imX(int x, int width)
{
	return int(x + 0.5*width - 0.5);
}
int ma2imY(int y, int height)
{
	return  int(-y + 0.5*height - 0.5);
}


//��ȡ��תǰ(����)
double getX_b(int x, int y, double sina, double cosa, double c1)
{
	double xx = x * cosa - y * sina + c1;
	return xx;
}
double getY_b(int x, int y, double sina, double cosa, double c2)
{
	double yy = y * cosa + x * sina + c2;
	return yy;
}

//��ȡ��ת��(ǰ��)
double getX_a(int x, int y, double sina, double cosa, double c1)
{
	double xx = x * cosa + y * sina + c1;
	return xx;
}
double getY_a(int x, int y, double sina, double cosa, double c2)
{
	double yy = y * cosa - x * sina + c2;
	return yy;
}


//ǰ�򣨰汾1��
int main_a1()
{
	Mat img, mat, mat_rotate_a;

	Mat src_img, dst, gray_img;
	//putText();
	//src_img = imread(R"(F:/C/image2txt/resource/33.png)");
	src_img = imread(R"(C:\Users\qianyewu\Desktop\ǧҹ��\33.png)");

	//imshow("src_img", src_img);
	cvtColor(src_img, gray_img, CV_RGB2GRAY);

	//imshow("gray_img", gray_img);
	//cout << "��ȡ����" << getPixel(10, 300, gray_img) <<"   "<< endl;




	int height = gray_img.rows, width = gray_img.cols;
	int degree = -30;
	double angle = degree * Pi / 180;//���� = �Ƕ� * Pi / 180;  //CV_PI 
	double sina = sin(angle), cosa = cos(angle);

	/*img = imread("C:/Users/qianyewu/Desktop/1/3.jpg", 1);
	IplImage imgTmp = img;
	IplImage *input = cvCloneImage(&imgTmp);

	IplImage *output1, *output2;
	output1=rotateImage1(input,30);
	output2 = rotateImage1(input, 30);

	imshow("input", img);
	cvShowImage("output1", output1);
	cvShowImage("output2", output2);
	*/

	int width_rotate = int(height * fabs(sina) + width * fabs(cosa));
	int height_rotate = int(width * fabs(sina) + height * fabs(cosa));
	mat = creatImage(height, width);

	//cout << width_rotate << " ..��ת��.." << height_rotate << "\n\n" << endl;
	mat_rotate_a = creatImage(height_rotate, width_rotate);

	//Bresenham_1(100, 1, 333, 233, m);
	/*Bresenham(10, 10, 80, 90, mat);
	Bresenham(80, 10, 90, 80, mat);
	Bresenham(90, 80, 90, 80, mat);
	Bresenham(90, 80, 110, 10, mat);
	*/
	//cout << m.rows << "."<<endl;
	for (int i = 0; i < mat.rows; i++)        //����ÿһ��ÿһ�в�����������ֵ
	{
		for (int j = 0; j < mat.cols; j++)
		{
			if (i > 10 && i < 80 && j>20 && j < 53)
			{
				//m.at<uchar>(i, j) = 0;
				//cout << i << " . " <<j<< endl;

				//mat.ptr<uchar>(i)[j] = 33;

				//mat.ptr<uchar>(i+1)[j] = 0;
				//mat.ptr<uchar>(i+2)[j] = 0;
			}
		}
	}
	int rx, ry;//��ת������
	int mx, my;//��ѧ����
	int ix, iy;//��ת���ͼ������
	for (int i = 0; i < mat.cols; i++)
	{
		for (int j = 0; j < mat.rows; j++)
		{
			mx = im2maX(i, width);
			my = im2maY(j, height);
			rx = getX(mx, my, sina, cosa);
			ry = getY(mx, my, sina, cosa);
			ix = ma2imX(rx, width_rotate);
			iy = ma2imY(ry, height_rotate);
			mat_rotate_a.ptr<uchar>(iy)[ix] = getPixel(j, i, gray_img);
			//cout << "��ȡ����" << getPixel(j, i, mat) << endl;
			//cout <<i<<"  "<<j << " ...  " <<mx<<"  "<< my  << " ... "<<rx<<"   "<<ry<<  " ... " << ix <<"  "<< iy<<endl;

			//cout << int(mat.ptr<uchar>(i)[j]) << endl;
		}
	}
	//cout <<"��ȡ����"<< getPixel(10,111,mat) << endl;
	/*uchar y1=mat.ptr<uchar>(0)[1];
	uchar y2 = mat.ptr<uchar>(0)[0];
	cout << "y1:" << (int)y1<<" ,y2: "<<(int)y2<<endl;
	*/
	/**
	IplImage m1 = mat, *output3;
	IplImage *m2 = cvCloneImage(&m1);
	output3 = rotateImage2(m2, 30);
	cvShowImage("output3", output3);
	*/

	int x1, y1;
	int x0, y0;
	x0 = 0;
	y0 = 2;
	x1 = x0 - 0.5*width;// width - 1;
	y1 = -y0 + 0.5*height;
	int(fabs(sina*height));
	cout << " x1: " << x1 << " ,y1: " << y1 << endl;
	cout << getX(x1, y1, sina, cosa) << " ..����.  " << getY(x1, y1, sina, cosa) << "\n" << endl;
	//cout << getX(x1, y1, sina, cosa) << " ...  " << getY(x1, y1, sina, cosa) << "\n\n" << endl;
	//cout << mat << "\n\n"<< endl;
	//cout << mat_rotate_a << endl;
	//imwrite("gray_imgage.jpg", gray_img);
	imwrite("mat_rotate_a.jpg", mat_rotate_a);
	imshow("mat_rotate_a", mat_rotate_a);
	imshow("gray_img", gray_img);
	waitKey(0);
	system("pause");
	return 0;
}

//ǰ�򣨰汾2��
int main_a2()
{
	Mat img, mat, mat_rotate_a;
	Mat src_img, dst, gray_img;

	src_img = imread(R"(C:\Users\qianyewu\Desktop\ǧҹ��\33.png)");
	cvtColor(src_img, gray_img, CV_RGB2GRAY);

	int height = gray_img.rows, width = gray_img.cols;
	int degree = -30;
	double angle = degree * Pi / 180;//���� = �Ƕ� * Pi / 180;  //CV_PI 
	double sina = sin(angle), cosa = cos(angle);

	int width_rotate = int(height * fabs(sina) + width * fabs(cosa));
	int height_rotate = int(width * fabs(sina) + height * fabs(cosa));

	mat_rotate_a = creatImage(height_rotate, width_rotate);

	double c1 = -width / 2 * cosa - height / 2 * sina + width_rotate / 2 ;
	double c2 = width / 2 * sina - height / 2 * cosa + height_rotate / 2;

	int  xx = 0;
	int yy = 0;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			xx = int(getX_a(j, i, sina, cosa, c1) + 0.5);
			yy = int(getY_a(j, i, sina, cosa, c2) + 0.5);
			if (xx >= 0 && yy >= 0 && xx < width_rotate&&yy < height_rotate)
			{
				mat_rotate_a.ptr<uchar>(yy)[xx] = getPixel(i, j, gray_img);
				//cout << xx<<" , " <<yy<< " ��ȡ " << getPixel(yy, xx, gray_img) <<endl;
			}
			else
				continue;
		}
	}

	//imwrite("gray_imgage.jpg", gray_img);
	//imwrite("mat_rotate_a2.jpg", mat_rotate_a);
	imshow("mat_rotate_a", mat_rotate_a);
	imshow("gray_img", gray_img);
	waitKey(0);
	system("pause");
	return 0;
}

//����
int main_b()
{
	Mat img, mat, mat_rotate_b;
	Mat src_img, dst, gray_img;
	src_img = imread(R"(C:\Users\qianyewu\Desktop\ǧҹ��\33.png)");
	cvtColor(src_img, gray_img, CV_RGB2GRAY);

	int height = gray_img.rows, width = gray_img.cols;
	//int height = 3, width = 3;
	int degree = -30;
	double angle = degree * Pi / 180;//���� = �Ƕ� * Pi / 180;  //CV_PI 
	double sina = sin(angle), cosa = cos(angle);

	int width_rotate = int(height * fabs(sina) + width * fabs(cosa));
	int height_rotate = int(width * fabs(sina) + height * fabs(cosa));

	//cout << "qian" << width << " ," << height << "hou " << width_rotate << ", " << height_rotate << endl;
	mat_rotate_b = creatImage(height_rotate, width_rotate);

	//uble c1 = width / 2 - width_rotate / 2 * cosa - height_rotate / 2 * sina;
	//uble c2 = height / 2 + width_rotate / 2 * sina - height_rotate / 2 * cosa;
	double c1 = width / 2 - width_rotate / 2 * cosa + height_rotate / 2 * sina;
	double c2 = height / 2 - width_rotate / 2 * sina - height_rotate / 2 * cosa;

	int  xx = 0;
	int yy = 0;
	for (int i = 0; i < mat_rotate_b.rows; i++)
	{
		for (int j = 0; j < mat_rotate_b.cols; j++)
		{
			xx = int(getX_b(j, i, sina, cosa, c1)+0.5);
			yy = int(getY_b(j, i, sina, cosa, c2)+0.5);
			if (xx >= 0 && yy>=0&&xx<width&&yy<height)
			{
				mat_rotate_b.ptr<uchar>(i)[j] = getPixel(yy, xx, gray_img);
				//cout << xx<<" , " <<yy<< " ��ȡ " << getPixel(yy, xx, gray_img) <<endl;
			}
			else
				continue;
		}
	}
	
	imshow("mat_rotate_b", mat_rotate_b);
	imshow("gray_img", gray_img);
	//imwrite("gray_image_big.jpg", gray_img);
	//imwrite("mat_rotate_b_big.jpg", mat_rotate_b);
	waitKey(0);
	system("pause");
	return 0;
}
