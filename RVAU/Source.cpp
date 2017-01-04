

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "Marker.h"
#include "opencv"

using namespace cv;
using namespace std;

RNG rng(12345);

/// Global variables
Mat src, src_gray;
int thresh = 127;
int max_thresh = 255;
vector<Point3f> m_markerCorners3d;
vector<Point2f> m_markerCorners2d;
Mat canonicalMarkerImage;
Size markerSize = Size(100, 100);;
float m_minContourLengthAllowed = 100;

inline void show(string name, const Mat& m)
{
	cv::imshow(name, m);
	waitKey(25);
}

template <typename T>
string ToString(const T& value)
{
	ostringstream stream;
	stream << value;
	return stream.str();
}


void greyScaleImg(const Mat& img, Mat& greyscale)
{
	cvtColor(img, greyscale, CV_BGR2GRAY);
	//show("Greyscale", greyscale);
}

void thresholdImg(const Mat& greyscale, Mat& thresholdImg)
{
	threshold(greyscale, thresholdImg, thresh, max_thresh, cv::THRESH_OTSU);
	//show("ThresHold", thresholdImg);
}

void findContoursImg(const Mat& thresholdImg, vector<vector<Point>>& contours, int minContourPointsAllowed)
{
	vector<Vec4i> hierarchy;
	vector< vector<Point> > allContours;

	findContours(thresholdImg, allContours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	contours.clear();
	for (size_t i = 0; i<allContours.size(); i++)
	{
		size_t contourSize = allContours[i].size();
		if (contourSize > minContourPointsAllowed)
		{
			contours.push_back(allContours[i]);
		}
	}
	Mat contoursImage(thresholdImg.size(), CV_8UC1);
	contoursImage = Scalar(0);
	drawContours(contoursImage, contours, -1, cv::Scalar(255), 2, CV_AA);
	show("Contours", contoursImage);
}

float perimeter(const vector<Point2f> &a)
{
	float sum = 0, dx, dy;

	for (size_t i = 0; i<a.size(); i++)
	{
		size_t i2 = (i + 1) % a.size();

		dx = a[i].x - a[i2].x;
		dy = a[i].y - a[i2].y;

		sum += sqrt(dx*dx + dy*dy);
	}

	return sum;
}


bool isInto(Mat &contour, vector<Point2f> &b)
{
	for (size_t i = 0; i<b.size(); i++)
	{
		if (pointPolygonTest(contour, b[i], false)>0) return true;
	}
	return false;
}

void findCandidates(const vector<vector<Point>>& contours, vector<Marker>& detectedMarkers)

{
	vector<Point>  approxCurve;
	vector<Marker>     possibleMarkers;
	float m_minContourLengthAllowed = 100;
	float perimeter(const vector<Point2f> &a);

	// For each contour, analyze if it is a parallelepiped likely to be the marker
	for (size_t i = 0; i<contours.size(); i++)
	{
		// Approximate to a polygon
		double eps = contours[i].size() * 0.05;
		approxPolyDP(contours[i], approxCurve, eps, true);

		// We interested only in polygons that contains only four points
		if (approxCurve.size() != 4)
			continue;

		// And they have to be convex
		if (!isContourConvex(approxCurve))
			continue;

		// Ensure that the distance between consecutive points is large enough
		float minDist = numeric_limits<float>::max();

		for (int i = 0; i < 4; i++)
		{
			Point side = approxCurve[i] - approxCurve[(i + 1) % 4];
			float squaredSideLength = side.dot(side);
			minDist = min(minDist, squaredSideLength);
		}

		// Check that distance is not very small
		if (minDist < m_minContourLengthAllowed)
			continue;

		// All tests are passed. Save marker candidate:
		Marker m;

		for (int i = 0; i<4; i++)
			m.points.push_back(Point2f(approxCurve[i].x, approxCurve[i].y));

		// Sort the points in anti-clockwise order
		// Trace a line between the first and second point.
		// If the third point is at the right side, then the points are anti-clockwise
		Point v1 = m.points[1] - m.points[0];
		Point v2 = m.points[2] - m.points[0];

		double o = (v1.x * v2.y) - (v1.y * v2.x);

		if (o < 0.0)		 //if the third point is in the left side, then sort in anti-clockwise order
			swap(m.points[1], m.points[3]);

		possibleMarkers.push_back(m);
	}


	// Remove these elements which corners are too close to each other.
	// First detect candidates for removal:
	vector< pair<int, int> > tooNearCandidates;
	for (size_t i = 0; i<possibleMarkers.size(); i++)
	{
		const Marker& m1 = possibleMarkers[i];

		//calculate the average distance of each corner to the nearest corner of the other marker candidate
		for (size_t j = i + 1; j<possibleMarkers.size(); j++)
		{
			const Marker& m2 = possibleMarkers[j];

			float distSquared = 0;

			for (int c = 0; c < 4; c++)
			{
				Point v = m1.points[c] - m2.points[c];
				distSquared += v.dot(v);
			}

			distSquared /= 4;

			if (distSquared < 100)
			{
				tooNearCandidates.push_back(pair<int, int>(i, j));
			}
		}
	}

	// Mark for removal the element of the pair with smaller perimeter
	vector<bool> removalMask(possibleMarkers.size(), false);

	for (size_t i = 0; i<tooNearCandidates.size(); i++)
	{
		float p1 = perimeter(possibleMarkers[tooNearCandidates[i].first].points);
		float p2 = perimeter(possibleMarkers[tooNearCandidates[i].second].points);

		size_t removalIndex;
		if (p1 > p2)
			removalIndex = tooNearCandidates[i].second;
		else
			removalIndex = tooNearCandidates[i].first;

		removalMask[removalIndex] = true;
	}

	// Return candidates
	detectedMarkers.clear();
	for (size_t i = 0; i<possibleMarkers.size(); i++)
	{
		if (!removalMask[i])
			detectedMarkers.push_back(possibleMarkers[i]);
	}

}

void recognizeMarkers(const Mat& grayscale, vector<Marker>& detectedMarkers)
{
	vector<Marker> goodMarkers;

	m_markerCorners2d.push_back(cv::Point2f(0, 0));
	m_markerCorners2d.push_back(cv::Point2f(99, 0));

	// Identify the markers
	for (size_t i = 0; i<detectedMarkers.size(); i++)
	{
		Marker& marker = detectedMarkers[i];

		// Find the perspective transformation that brings current marker to rectangular form
		Mat markerTransform = getPerspectiveTransform(marker.points, m_markerCorners2d);

		// Transform image to get a canonical marker image
		warpPerspective(grayscale, canonicalMarkerImage, markerTransform, markerSize);

		Mat markerImage = grayscale.clone();
		marker.drawContour(markerImage);
		Mat markerSubImage = markerImage(cv::boundingRect(marker.points));

		show("Source marker" + ToString(i), markerSubImage);
		show("Marker " + ToString(i) + " after warp", canonicalMarkerImage);

		int nRotations;
		int id = Marker::getMarkerId(canonicalMarkerImage, nRotations);
		if (id != -1)
		{
			marker.id = id;
			//sort the points so that they are always in the same order no matter the camera orientation
			std::rotate(marker.points.begin(), marker.points.begin() + 4 - nRotations, marker.points.end());

			goodMarkers.push_back(marker);
		}
	}

	// Refine marker corners using sub pixel accuracy
	if (goodMarkers.size() > 0)
	{
		std::vector<cv::Point2f> preciseCorners(4 * goodMarkers.size());

		for (size_t i = 0; i<goodMarkers.size(); i++)
		{
			const Marker& marker = goodMarkers[i];

			for (int c = 0; c <4; c++)
			{
				preciseCorners[i * 4 + c] = marker.points[c];
			}
		}

		cv::TermCriteria termCriteria = cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 30, 0.01);
		cv::cornerSubPix(grayscale, preciseCorners, cvSize(5, 5), cvSize(-1, -1), termCriteria);

		// Copy refined corners position back to markers
		for (size_t i = 0; i<goodMarkers.size(); i++)
		{
			Marker& marker = goodMarkers[i];

			for (int c = 0; c<4; c++)
			{
				marker.points[c] = preciseCorners[i * 4 + c];
			}
		}
	}

	cv::Mat markerCornersMat(grayscale.size(), grayscale.type());
	markerCornersMat = cv::Scalar(0);

	for (size_t i = 0; i<goodMarkers.size(); i++)
	{
		goodMarkers[i].drawContour(markerCornersMat, cv::Scalar(255));
	}

	show("Markers refined edges", grayscale * 0.5 + markerCornersMat);

	detectedMarkers = goodMarkers;
}


/** @function main */
int main()
{

	Mat img;
	VideoCapture cap(0);

	while (true)
	{

		cap >> img;

		Mat greyScale, tHold;
		vector<Marker> detectedMarkers;

		//Convert to greyscale
		greyScaleImg(img, greyScale);

		//Threshold
		thresholdImg(greyScale, tHold);

		vector<vector<Point> > contours;
		findContoursImg(tHold, contours, greyScale.cols / 5);

		findCandidates(contours, detectedMarkers);

		recognizeMarkers(greyScale, detectedMarkers);

		waitKey(33);
	}

}
