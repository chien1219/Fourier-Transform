#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <highgui.h>
#include <math.h>
#include "cv.h"
using namespace std;
using namespace cv;

#define  PI     3.1415926535897932384626433832795028  
#define	 FFT_N	1024
int ques = 1;
string img_name, img_output1;

struct complex_ { float real, imag; };                                    //定義一個復數結構
struct complex_ s[FFT_N*FFT_N];                                           

struct complex_ EE(struct complex_ a, struct complex_ b)    //對兩複數做乘積
{
	struct complex_ c;

	c.real = a.real*b.real - a.imag*b.imag;
	c.imag = a.real*b.imag + a.imag*b.real;

	return(c);
}
void FFT(struct complex_ *xin, int fft_size)
{
	int f, m, nv2, nm1, i, k, l, j = 0;
	struct complex_ u, w, t;

	nv2 = fft_size / 2;                   //變址運算
	nm1 = fft_size - 1;

	for (i = 0; i < nm1; i++)
	{
		if (i < j)                      //如果i<j,即進行變址
		{
			t = xin[j];
			xin[j] = xin[i];
			xin[i] = t;
		}
		k = nv2;                   
		while (k <= j)             
		{
			j = j - k;             
			k = k / 2;             
		}
		j = j + k;                 
	}

	{
		int le, lei, ip;                            //FFT運算

		f = fft_size;
		for (l = 1; (f = f / 2) != 1; l++)    
			;
		for (m = 1; m <= l; m++)              
		{                                     
			le = 2 << (m - 1);                
			lei = le / 2;                     
			u.real = 1.0;                     
			u.imag = 0.0;
			w.real = cos(PI / lei);           
			w.imag = -sin(PI / lei);
			for (j = 0; j <= lei - 1; j++)                  
			{
				for (i = j; i <= FFT_N*FFT_N - 1; i = i + le)
				{
					ip = i + lei;                           
					t = EE(xin[ip], u);                    
					xin[ip].real = xin[i].real - t.real;
					xin[ip].imag = xin[i].imag - t.imag;
					xin[i].real = xin[i].real + t.real;
					xin[i].imag = xin[i].imag + t.imag;
				}
				u = EE(u, w);                           //改變系數，進行下一個蝶形運算
			}
		}
	}
}
int main(){

	cout << "Input Question Number = ";
	cin >> ques;

	switch (ques){
	case 1:
		img_name = "Q1.tif";		//fig 4.29
		img_output1 = "Q1_output.tif";
		break;
	case 2:
		img_name = "Q2_new.tif";	//fig 4.36
		img_output1 = "Q2_output.tif";
		break;
	case 3:
		img_name = "Q3.tif";		//fig 4.38
		img_output1 = "Q3_output.tif";
		break;
	case 4:
		img_name = "Q4.tif";		//fig 4.41
		img_output1 = "Q4_output.tif";
		break;
	default:
		break;
	}
	// Read input images
	Mat SrcImg = imread(img_name, CV_LOAD_IMAGE_GRAYSCALE);

	Mat padded;                            //expand input image to optimal size
	int m = FFT_N;
	cout << SrcImg.rows << " " << m << endl;
	int n = FFT_N; // on the border add zero values
	cout << SrcImg.cols << " " << n << endl;

	copyMakeBorder(SrcImg, padded, 0, m - SrcImg.rows, 0, n - SrcImg.cols, BORDER_CONSTANT, Scalar::all(0));

	// Create a grayscale output image matrix
	Mat DstImg = Mat(padded.rows, padded.cols, CV_8UC1);

	//Mat planes[] = { Mat_<float>(SrcImg), Mat::zeros(SrcImg.size(), CV_32F) };
	//Mat complex__Img;
	//merge(planes, 2, complex_Img);

	//struct complex_ s[FFT_N * FFT_N];

	for (int i = 0; i < FFT_N * FFT_N; i++)                                        //to prevent null in array
	{
		s[i].real = 1 + 2 * sin(2 * PI*i / FFT_N*FFT_N);			  //實部為正弦波FFT_N
		s[i].imag = 0;                                                //虛部為0
	}


	for (int i = 0; i < padded.rows; ++i)
	for (int j = 0; j < padded.cols; ++j){
		s[i*FFT_N + j].real = (double)padded.at<uchar>(i, j);
	}

	FFT(s, FFT_N * FFT_N);
	
	for (int i = 0; i < FFT_N * FFT_N; i++)                             //求變換後結果的模值，存入復數的實部部分
		s[i].real =log10( sqrt(s[i].real*s[i].real + s[i].imag * s[i].imag));

	for (int i = 0; i < padded.rows; ++i)
	for (int j = 0; j < padded.cols; ++j){
		DstImg.at<uchar>(i, j) = (uchar)s[i * FFT_N + j].real;
	}
	/*
	for (int i = 0; i < padded.rows; ++i){
		for (int j = 0; j < padded.cols; ++j){
			cout << (double)DstImg.at<uchar>(i, j) << " ";
		}
		cout << endl;
	}
	*/
	//dft(complex_Img, complex_Img);
	
	//split(complex_Img, planes);                  //分離通道，planes[0]為實數部分，planes[1]為虛數部分 
	//magnitude(planes[0], planes[1], planes[0]); //planes[0] = sqrt((planes[0])^2 + (planes[1])^2
	//Mat magI = planes[0];
	//magI += Scalar::all(1);                     //magI = log(1+planes[0])
	//log(magI, magI);

	//magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));  //令邊長為偶數

	//將區塊重排，讓原點在影像的中央
	int cx = DstImg.cols / 2;
	int cy = DstImg.rows / 2;

	Mat q0(DstImg, Rect(0, 0, cx, cy));
	Mat q1(DstImg, Rect(cx, 0, cx, cy));
	Mat q2(DstImg, Rect(0, cy, cx, cy));
	Mat q3(DstImg, Rect(cx, cy, cx, cy));

	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	normalize(DstImg, DstImg, 0, 255, CV_MINMAX);

	imshow("輸入圖", SrcImg);
	imshow("頻譜", DstImg);
	imwrite(img_output1, DstImg);


	waitKey();

	return 0;
}

/*
void cv::dft(InputArray _src0, OutputArray _dst, int flags, int nonzero_rows)
{
CV_INSTRUMENT_REGION()

#ifdef HAVE_CLAMDFFT
CV_OCL_RUN(ocl::haveAmdFft() && ocl::Device::getDefault().type() != ocl::Device::TYPE_CPU &&
_dst.isUMat() && _src0.dims() <= 2 && nonzero_rows == 0,
ocl_dft_amdfft(_src0, _dst, flags))
#endif

#ifdef HAVE_OPENCL
CV_OCL_RUN(_dst.isUMat() && _src0.dims() <= 2,
ocl_dft(_src0, _dst, flags, nonzero_rows))
#endif

Mat src0 = _src0.getMat(), src = src0;
bool inv = (flags & DFT_INVERSE) != 0;
int type = src.type();
int depth = src.depth();

CV_Assert(type == CV_32FC1 || type == CV_32FC2 || type == CV_64FC1 || type == CV_64FC2);

// Fail if DFT_complex__INPUT is specified, but src is not 2 channels.
CV_Assert(!((flags & DFT_complex__INPUT) && src.channels() != 2));

if (!inv && src.channels() == 1 && (flags & DFT_complex__OUTPUT))
_dst.create(src.size(), CV_MAKETYPE(depth, 2));
else if (inv && src.channels() == 2 && (flags & DFT_REAL_OUTPUT))
_dst.create(src.size(), depth);
else
_dst.create(src.size(), type);

Mat dst = _dst.getMat();

int f = 0;
if (src.isContinuous() && dst.isContinuous())
f |= CV_HAL_DFT_IS_CONTINUOUS;
if (inv)
f |= CV_HAL_DFT_INVERSE;
if (flags & DFT_ROWS)
f |= CV_HAL_DFT_ROWS;
if (flags & DFT_SCALE)
f |= CV_HAL_DFT_SCALE;
if (src.data == dst.data)
f |= CV_HAL_DFT_IS_INPLACE;
Ptr<hal::DFT2D> c = hal::DFT2D::create(src.cols, src.rows, depth, src.channels(), dst.channels(), f, nonzero_rows);
c->apply(src.data, src.step, dst.data, dst.step);
}
*/