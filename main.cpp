#include <iostream>
#include <opencv2/opencv.hpp>
#include "include/kalman_filter.h"
#include "include/utils/get_fps.h"

using namespace std;
using namespace cv;

int g_fps;
GetFps get_fps;

void motionDetection(Mat pre_frame, Mat frame)
{
	Mat origin = frame.clone();
	// convert the original image to grayscale image
	cvtColor(pre_frame, pre_frame, CV_BGR2GRAY);
	cvtColor(frame, frame, CV_BGR2GRAY);

	// calculate the absolute difference between two images
	Mat diff_frame;
	absdiff(pre_frame, frame, diff_frame);
	imshow("diff_frame", diff_frame);

	// binarization
	threshold(diff_frame, diff_frame, 25, 255, CV_THRESH_BINARY);
	imshow("diff_threshold", diff_frame);
	
	// opening operation
	Mat dilate_element = getStructuringElement(MORPH_RECT, Size(30,30));
	Mat erode_element = getStructuringElement(MORPH_RECT, Size(3,3));
	erode(diff_frame, diff_frame, erode_element);
	dilate(diff_frame, diff_frame, dilate_element);
	imshow("diff_erode_dilate", diff_frame);

	// draw bounding box
	vector<vector<Point> > contours;
	findContours(diff_frame, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	vector<Point> pts;
	for(int i = 0; i < contours.size(); i++){
		for (int j = 0; j < contours[i].size(); j++){
			pts.push_back(contours[i][j]);
		}
	}
	Rect rect = boundingRect(pts);
	//cout << "top left: " << rect.tl() << endl;
	//cout << "bottom right" << rect.br() << endl << endl;
	rectangle(origin, rect, Scalar(0,0,255), 5);

	// draw fps
	get_fps.getFps(g_fps);
	string s_fps = to_string(g_fps);
	putText(origin, "fps = "+s_fps, Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 1, Scalar(100,100,255), 2, 8);	
	imshow("result", origin);
}

int main()
{
	VideoCapture capture(0);
	if(!capture.isOpened())
	{
		cout << "video open error" << endl;
		return 0;
	}
	// previous frame	
	Mat pre_frame;
	while(1)
	{
		Mat frame;
		capture >> frame;
		if(frame.empty()){
			break;
		}	

		if(pre_frame.empty()){
			motionDetection(frame, frame);	
		}
		else{
			motionDetection(pre_frame, frame);
		}
		// update previous frame
		pre_frame = frame.clone();

		if(waitKey(30) == 27){
			break;
		}
	}
	return 0;
}
