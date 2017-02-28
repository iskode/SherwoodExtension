 #include<opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"
 using namespace cv;
using namespace std;
   
    int main()
    {
    
         VideoCapture cap;
    // open the default camera, use something different from 0 otherwise;
    // Check VideoCapture documentation.
    if(!cap.open(0))
        return 0;
    for(;;)
    {
          Mat frame;
          cap >> frame;
          if( frame.empty() ) break; // end of video stream
          imshow("my webcaml in action !!! Youpi !!!", frame);
          if( waitKey(10) == 27 ) break; // stop capturing by pressing ESC 
    }
    // the camera will be closed automatically upon exit
    // cap.close();
    return 0;
   
       
   }
