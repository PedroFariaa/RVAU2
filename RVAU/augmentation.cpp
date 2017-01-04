#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/features2d.hpp"

#include <iostream>
#include <windows.h>

using namespace std;
using namespace cv;


vector<Mat> markers;
vector<string> obj;
string object;


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

int tutorial_mode(){
	
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
		waitKey();
		destroyWindow("Thresholded Image");
		
		Mat canny_output;
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;

		Canny(tHold, canny_output, 100, 200);
		/// Find contours
		findContours(canny_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));
		imshow("canny", canny_output);
		waitKey();
		destroyWindow("canny");

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

		imshow("detect marker using mask", mask);
		waitKey();
		destroyWindow("detect marker using mask");

		imshow("crop marker", crop);
		waitKey();
		destroyWindow("crop marker");


		// anotehr canny
		Mat canny_output2;
		vector<vector<Point> > contours2;
		vector<Vec4i> hierarchy2;
		Canny(crop, canny_output2, 100, 200);
		/// Find contours
		findContours(canny_output2, contours2, hierarchy2, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));
		imshow("find marker contours", canny_output2);
		waitKey();
		destroyWindow("find marker contours");

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

		imshow("display marker contours", img2);
		waitKey();
		destroyWindow("display marker contours");

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

//draw pyramid function
void drawPyramid(InputOutputArray _image, InputArray _cameraMatrix, InputArray _distCoeffs,
	InputArray _rvec, InputArray _tvec, float length, Scalar color) {

	float x2 = length * 2;

	// project axis points
	vector< Point3f > pyrPoints;
	pyrPoints.push_back(Point3f(-length, -length, 0));
	pyrPoints.push_back(Point3f(-length, length, 0));
	pyrPoints.push_back(Point3f(length, length, 0));
	pyrPoints.push_back(Point3f(length, -length, 0));
	pyrPoints.push_back(Point3f(-length, length, 0));
	pyrPoints.push_back(Point3f(length, length, 0));
	pyrPoints.push_back(Point3f(length, -length, 0));
	pyrPoints.push_back(Point3f(-length, -length, 0));

	pyrPoints.push_back(Point3f(-length, -length, 0));
	pyrPoints.push_back(Point3f(0, 0, x2));
	pyrPoints.push_back(Point3f(-length, length, 0));
	pyrPoints.push_back(Point3f(0, 0, x2));
	pyrPoints.push_back(Point3f(length, length, 0));
	pyrPoints.push_back(Point3f(0, 0, x2));
	pyrPoints.push_back(Point3f(length, -length, 0));
	pyrPoints.push_back(Point3f(0, 0, x2));

	pyrPoints.push_back(Point3f(0, 1, 0));
	pyrPoints.push_back(Point3f(0, 0, 1));
	vector< Point2f > imagePoints;
	projectPoints(pyrPoints, _rvec, _tvec, _cameraMatrix, _distCoeffs, imagePoints);

	// draw axis lines

	for (int i = 0; i < imagePoints.size() - 2; i += 2)
		line(_image, imagePoints[i], imagePoints[i + 1], color, 3);

}




// draw cube function
void drawCube(InputOutputArray _image, InputArray _cameraMatrix, InputArray _distCoeffs,
	InputArray _rvec, InputArray _tvec, float length, Scalar color) {

	float x2 = length * 2;

	// project axis points
	vector< Point3f > cubePoints;
	cubePoints.push_back(Point3f(-length, -length, 0));
	cubePoints.push_back(Point3f(-length, length, 0));
	cubePoints.push_back(Point3f(length, length, 0));
	cubePoints.push_back(Point3f(length, -length, 0));
	cubePoints.push_back(Point3f(-length, length, 0));
	cubePoints.push_back(Point3f(length, length, 0));
	cubePoints.push_back(Point3f(length, -length, 0));
	cubePoints.push_back(Point3f(-length, -length, 0));

	cubePoints.push_back(Point3f(-length, -length, x2));
	cubePoints.push_back(Point3f(-length, length, x2));
	cubePoints.push_back(Point3f(length, length, x2));
	cubePoints.push_back(Point3f(length, -length, x2));
	cubePoints.push_back(Point3f(-length, length, x2));
	cubePoints.push_back(Point3f(length, length, x2));
	cubePoints.push_back(Point3f(length, -length, x2));
	cubePoints.push_back(Point3f(-length, -length, x2));

	cubePoints.push_back(Point3f(-length, -length, 0));
	cubePoints.push_back(Point3f(-length, -length, x2));
	cubePoints.push_back(Point3f(-length, length, 0));
	cubePoints.push_back(Point3f(-length, length, x2));
	cubePoints.push_back(Point3f(length, length, 0));
	cubePoints.push_back(Point3f(length, length, x2));
	cubePoints.push_back(Point3f(length, -length, 0));
	cubePoints.push_back(Point3f(length, -length, x2));

	cubePoints.push_back(Point3f(0, 1, 0));
	cubePoints.push_back(Point3f(0, 0, 1));
	vector< Point2f > imagePoints;
	projectPoints(cubePoints, _rvec, _tvec, _cameraMatrix, _distCoeffs, imagePoints);

	// draw axis lines

	for (int i = 0; i < imagePoints.size() - 2; i += 2)
		line(_image, imagePoints[i], imagePoints[i + 1], color, 3);

}

int compareMarkers(Mat found) {
	if (markers.size() == 0)
		loadMarkers();
	
	Ptr<FeatureDetector> detector = FastFeatureDetector::create();
	vector<KeyPoint> keypoints1, keypoints2;
	vector< vector<KeyPoint>> markers_kp;
	Mat descriptors1, descriptors2;
	vector<Mat> markers_desc;
	FlannBasedMatcher matcher;
	vector< DMatch > matches;
	vector<int> good_matches;
	int gdm ;
	double max_dist = 0; double min_dist = 100;

	detector->detect(found, keypoints1);
	detector->compute(found, keypoints1, descriptors1);

	for (int i = 0; i < markers.size(); i++) {
		int gdm = 0;
		//detect keypoints
		detector->detect(markers[i], keypoints2);
		markers_kp.push_back(keypoints2);
		keypoints2.clear();
		//compute descriptors
		detector->compute(markers[i], keypoints2, descriptors2);
		markers_desc.push_back(descriptors2);
		//matcher
		matcher.match(descriptors1, markers_desc[i], matches);
		max_dist = 0;
		min_dist = 100;
		//-- Quick calculation of max and min distances between keypoints
		for (int i = 0; i < descriptors1.rows; i++) {
			double dist = matches[i].distance;
			if (dist < min_dist)
				min_dist = dist;
			if (dist > max_dist)
				max_dist = dist;
		}

		for (int i = 0; i < descriptors1.rows; i++)
			if (matches[i].distance < 3 * min_dist)
				gdm++;

		good_matches.push_back(gdm);

	}

	int ind;
	int ind_m = 0;

	for(int m = 0; m < good_matches.size(); m++)
		if (good_matches[m] > ind_m) {
			ind = m;
			ind_m = good_matches[m];
		}

	return ind;
}


int augmentation_mode() {
	Scalar red(255, 0, 0);
	Scalar green(0, 255, 0);
	Scalar blue(0, 0, 255);

	FileStorage fs("../RVAU/out_camera_data.xml", FileStorage::READ);
	Mat intrinsics, distortion;
	fs["Camera_Matrix"] >> intrinsics;
	fs["Distortion_Coefficients"] >> distortion;

	if (intrinsics.rows != 3 || intrinsics.cols != 3 || distortion.rows != 5 || distortion.cols != 1) {
		cout << "Run calibration (in ../RVAU/) first!" << endl;
		return 1;
	}

	VideoCapture cap(0);
	if (!cap.isOpened())  // check if we succeeded
		return -1;

	Mat image;

	for (;;) {
		cap >> image;
		Mat grayImage;
		cvtColor(image, grayImage, CV_RGB2GRAY);
		Mat blurredImage;
		blur(grayImage, blurredImage, Size(5, 5));
		Mat threshImage;
		threshold(blurredImage, threshImage, 128.0, 255.0, THRESH_OTSU);
		vector<vector<Point> > contours;
		findContours(threshImage, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
		//drawContours(image, contours, -1, red);

		vector<Mat> squares;
		for (auto contour : contours) {
			vector<Point> approx;
			approxPolyDP(contour, approx, arcLength(Mat(contour), true)*0.02, true);
			if (approx.size() == 4 &&
				fabs(contourArea(Mat(approx))) > 1000 &&
				isContourConvex(Mat(approx)))
			{
				Mat squareMat;
				Mat(approx).convertTo(squareMat, CV_32FC3);
				squares.push_back(squareMat);
			}
		}

		if (squares.size() > 0) {
			vector<Point3f> objectPoints = { Point3f(-1, -1, 0), Point3f(-1, 1, 0), Point3f(1, 1, 0), Point3f(1, -1, 0) };
			Mat objectPointsMat(objectPoints);
			cout << "objectPointsMat: " << objectPointsMat.rows << ", " << objectPointsMat.cols << endl;
			cout << "squares[0]: " << squares[0] << endl;
			Mat rvec;
			Mat tvec;
			solvePnP(objectPointsMat, squares[0], intrinsics, distortion, rvec, tvec);
			
			cout << "rvec = " << rvec << endl;
			cout << "tvec = " << tvec << endl;

			if(object == "cube")
				drawCube(image, intrinsics, distortion, rvec, tvec, 1, Scalar(0, 0, 255));
			if (object == "pyramid")
				drawPyramid(image, intrinsics, distortion, rvec, tvec, 1, Scalar(0, 0, 255));

			vector<Point3f> line3d = { { 0, 0, 0 },{ 0, 0, 1 } };
			vector<Point2f> line2d;
			projectPoints(line3d, rvec, tvec, intrinsics, distortion, line2d);
			cout << "line2d = " << line2d << endl;
			line(image, line2d[0], line2d[1], red);
		}

		cv::imshow("image", image);
		cv::waitKey(1);
	}

	return 0;
}

void choseObj() {
	char ans;
	cout << "[C]ube" << endl;
	cout << "[P]yramid" << endl;
	cin >> ans;

	if (ans == 'c' || ans == 'C')
		object = "cube";
	else if (ans == 'p' || ans == 'P')
		object = "pyramid";
	else {
		cout << "Invalid input !" << endl << endl;
		choseObj();
	}
}

void menu() {
	// menu
	int ans;
	cout << "[1] Display markers" << endl;
	cout << "[2] Augmentation mode" << endl;
	cout << "[3] Tutorial mode" << endl;
	cin >> ans;

	if (ans == 1) {
		loadMarkers();
		showMarkers();
		menu();
	}
	else if (ans == 2) {
		choseObj();
		augmentation_mode();
		menu();
	}
	else if (ans == 3) {
		tutorial_mode();
		menu();
	}
	else {
		cout << "Invalid input !" << endl;
		menu();
	}
}

int main(int argc, char** argv) {

	menu();
	return 0;

}

