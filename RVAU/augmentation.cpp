#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <windows.h>

using namespace std;
using namespace cv;


vector<Mat> markers;



//load markers
void loadMarkers() {
	Mat image;
	string dir = "../data/markers/";

	bool x = true;
	int i = 0;
	wchar_t* file = L"../data/markers/*.jpg";
	WIN32_FIND_DATA FindFileData;
	HANDLE hFind;
	hFind = FindFirstFile(file, &FindFileData);

	if (hFind != INVALID_HANDLE_VALUE) {
		i++;
		while ((x = FindNextFile(hFind, &FindFileData)) == TRUE)
			i++;
	}

	for (int j = 0; j < i; j++) {

		image = imread(dir+"mk"+to_string(j+1)+".jpg", CV_LOAD_IMAGE_GRAYSCALE);
		markers.push_back(image);
	}
}

void showMarkers(){
	for (int i = 0; i < markers.size(); i++) {
		imshow("Marker" + to_string(i + 1), markers[i]);
		waitKey();
		destroyWindow("Marker" + to_string(i + 1));
	}
	waitKey();
}

void augmentation() {

	loadMarkers();
	showMarkers();
}

int main(int argc, char* argv[]) {
	//augmentation();
	
	
	Mat img2;
	VideoCapture cap(0);

	while (true)
	{
		cap >> img2;

		Mat tHold;
		cvtColor(img2, tHold, CV_BGR2GRAY);
		threshold(tHold, tHold, 50, 255, CV_THRESH_OTSU);


		imshow("Thresholded Image", tHold);
		waitKey(1);
		
		Mat canny_output;
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;

		Canny(tHold, canny_output, 50, 150, 3);
		//cvFindCornerSubPix();

		/// Find contours
		findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		imshow("canny", canny_output);

		vector<RotatedRect> minRect(contours.size());
		for (int i = 0; i < contours.size(); i++)
			//ajustar parameteros
			if((abs(boundingRect(contours[i]).height - boundingRect(contours[i]).width) <= 5) && (contourArea(contours[i]) > 30 && contourArea(contours[i]) < 100) )
				minRect[i] = minAreaRect(Mat(contours[i]));

		/// Draw contours
		Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
		RNG rng;
		for (int i = 0; i< contours.size(); i++)
		{
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());

			Point2f rect_points[4]; minRect[i].points(rect_points);
			for (int j = 0; j < 4; j++)
				line(img2, rect_points[j], rect_points[(j + 1) % 4], color, 1, 8);
		}

		imshow("Contours", img2);

	}

	return 0;
	
}