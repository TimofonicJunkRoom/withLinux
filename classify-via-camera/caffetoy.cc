// CaffeToy
// (C) 2015 lumin <cdluminate@163.com>
// BSD-2-Clause
#include <iostream>

#include <glog/logging.h>
#include <opencv/cv.h>  
#include <opencv/cxcore.h>  
#include <opencv/highgui.h>  

#define WINDOW "CaffeToy"
using namespace std;

int main (int argc, char** argv)  
{
	cout << "\x1b[31m\x1b[m";
	// Init google-glog
	FLAGS_logtostderr = 1;
	google::InitGoogleLogging (argv[0]);
	LOG(INFO) << "Welcome to CaffeToy";

	IplImage* pFrame = NULL;  


	LOG(INFO) << "Create Camera Capture";
	CvCapture* pCapture = cvCreateCameraCapture(-1);  
	CHECK (pCapture != NULL) << "cvCreateCameraCapture [failed]";

	cvSetCaptureProperty (pCapture, CV_CAP_PROP_FPS, 1);

	LOG(INFO) << "Creating named window";
	cvNamedWindow(WINDOW, 1);  
	
	LOG(INFO) << "Showing Images";
	while( NULL != (pFrame = cvQueryFrame (pCapture)) )  
	//while (1)
	{
		//cvGrabFrame (pCapture);
		//pFrame = cvRetrieveFrame (pCapture);
		if (!pFrame) break;  
		LOG(INFO) << "cvQueryFrame [OK]";

		cvShowImage (WINDOW, pFrame);  
		cvSaveImage ("Frame.jpg", pFrame);

		LOG(INFO) << "Classify frame ...";
		// the program is from caffe source / examples / cpp-classification
		system ("classification m/deploy.prototxt m/bvlc_reference_caffenet.caffemodel m/imagenet_mean.binaryproto m/synset_words.txt Frame.jpg");

		// reset capture
		LOG(INFO) << "Releasing capture";
		cvReleaseCapture (&pCapture);  
		LOG(INFO) << "Create Camera Capture";
		pCapture = cvCreateCameraCapture(-1);  
		CHECK (pCapture != NULL) << "cvCreateCameraCapture [failed]";

		// wait for any key to trigger next time classification
		// ESC for exit.
		if ( 27 == (char)cvWaitKey(0)) {
			LOG(INFO) << "ESC arrived. Exiting ...";
			break;
		}
		cout << "\x1b[31m----------------------------------------------\x1b[m" << endl;
	}  
	LOG(INFO) << "Releasing Memory...";
	cvReleaseCapture (&pCapture);  
	cvDestroyWindow (WINDOW);  
	LOG(INFO) << "Exit.";
}
