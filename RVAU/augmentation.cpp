#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <windows.h>

using namespace std;
using namespace cv;


vector<Mat> markers;


// helper function :
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
static double angle(Point pt1, Point pt2, Point pt0)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}


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

		Mat tHold, img3;
		cvtColor(img2, tHold, CV_BGR2GRAY);
		cvtColor(img2, img3, CV_BGR2GRAY);
		threshold(tHold, tHold, 50, 255, CV_THRESH_OTSU);

		imshow("Thresholded Image", tHold);
		waitKey(1);
		
		Mat canny_output;
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;

		Canny(tHold, canny_output, 100, 200);
		/// Find contours
		findContours(canny_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));
		imshow("canny", canny_output);


		// you could also reuse img1 here
		Mat mask = Mat::zeros(canny_output.rows, canny_output.cols, CV_8UC1);	
		// CV_FILLED fills the connected components found
		drawContours(mask, contours, -1, Scalar(255), CV_FILLED);

		int erosion_size = 10;
		Mat element = getStructuringElement(cv::MORPH_RECT,
			cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
			cv::Point(erosion_size, erosion_size));

		// Apply erosion or dilation on the image - closing
		erode(mask, mask, element);
		dilate(mask, mask, element);

		Mat crop(img2.rows, img2.cols, CV_8UC3);
		// set background to green
		crop.setTo(Scalar(0, 255, 0));
		// and copy the magic apple
		img2.copyTo(crop, mask);
		normalize(mask.clone(), mask, 0.0, 255.0, CV_MINMAX, CV_8UC1);

		imshow("mask", mask);
		imshow("cropped", crop);


		// anotehr canny
		Mat canny_output2;
		vector<vector<Point> > contours2;
		vector<Vec4i> hierarchy2;
		Canny(crop, canny_output2, 100, 200);
		/// Find contours
		findContours(canny_output2, contours2, hierarchy2, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));
		imshow("canny2", canny_output2);

		vector<RotatedRect> minRect(contours2.size());
		for (int i = 0; i < contours2.size(); i++)
			//ajustar parameteros
			//if ((abs(boundingRect(contours2[i]).height - boundingRect(contours2[i]).width) <= 5) && (contourArea(contours2[i]) > 50 && contourArea(contours2[i]) < 500000))
				minRect[i] = minAreaRect(Mat(contours2[i]));

		/// Draw contours
		Mat drawing = Mat::zeros(canny_output2.size(), CV_8UC3);
		RNG rng;
		for (int i = 0; i< contours2.size(); i++)
		{
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			drawContours(drawing, contours2, i, color, 2, 8, hierarchy2, 0, Point());

			Point2f rect_points[4]; minRect[i].points(rect_points);
			for (int j = 0; j < 4; j++)
				line(img2, rect_points[j], rect_points[(j + 1) % 4], color, 1, 8);
		}

		imshow("Contours", img2);

		/*
		//hough
		vector<Vec4i> lines;
		HoughLinesP(canny_output2, lines, 1, CV_PI / 180, 100, 70, 10);
		for (size_t i = 0; i < lines.size(); i++)
		{
			line(img2, Point(lines[i][0], lines[i][1]),
				Point(lines[i][2], lines[i][3]), Scalar(0, 0, 255), 3, 8);
		}
		imshow("detected lines", img2);
		*/


		/*
		// MARIA
		vector< Point2f > corners;
		Mat mask;
		double qualityLevel = 0.01;
		double minDistance = 100.0;
		int maxCorners = 20;
		int blockSize = 2;
		double k = 0.04;
		bool useHarrisDetector = true;

		goodFeaturesToTrack(tHold, corners, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k);

		for (size_t i = 0; i < corners.size(); i++)	{
			cout << corners[i] << endl;
			cv::circle(tHold, corners[i], 10, cv::Scalar(255.), -1);
		}

		imshow("good_window", tHold);
		*/

		/*
		//************************* CONTOURS *************************
		vector<RotatedRect> minRect(contours.size());
		for (int i = 0; i < contours.size(); i++)
			//ajustar parameteros
			if((abs(boundingRect(contours[i]).height - boundingRect(contours[i]).width) <= 5) && (contourArea(contours[i]) > 50 && contourArea(contours[i]) < 500000 ))
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

		*/

	}

	return 0;
	
}