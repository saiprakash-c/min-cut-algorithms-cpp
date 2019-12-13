#include "image.hpp"
#include "segmentation.hpp"

int main(int argc, char* argv[])
{
    Image img_object("/home/vulcan/01_courses/eecs_477/01_Project/src/messi.jpg");
    cv::Mat dest;
    cv::resize(img_object.image_data,dest,cv::Size(100,100),0,0,cv::INTER_CUBIC);
    img_object.image_data = dest;
    // img_object.ShowImageToSelectReferencePixels();
    img_object.source_cell[0] = 50;
    img_object.source_cell[1] = 50;
    img_object.sink_cell[0] = 0;
    img_object.sink_cell[0] = 0;
    img_object.ConvertToFlowNetwork();
    Segmentation segmentation_object(img_object.network);
    // segmentation_object.RunFordFulkersonAlgorithm(img_object.binary_mask);
    segmentation_object.RunPushRelabelAlgorithm(img_object.binary_mask);
    img_object.ShowSegmentedImage();
    return 0;
}