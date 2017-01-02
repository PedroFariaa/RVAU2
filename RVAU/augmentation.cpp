#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <windows.h>

using namespace std;
using namespace cv;





//show markers
void showMarkers() {
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
		imread(dir+"mk"+to_string(j)+".jpg", CV_LOAD_IMAGE_GRAYSCALE);
		imshow("Marker "+to_string(j), image);
	}
}