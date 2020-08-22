#include "image.hpp"
#include "segmentation.hpp"

enum Mode
{
    edmond_karp=0,
    push_relabel,
};

int main(int argc, char* argv[])
{
    // mode that stores the algorithm that the user wants
    Mode mode;
    // if number of arguments are less, exit 
    if(argc !=2 )
    {
        std::cout << "Provide correct number of arugments(2)" << "\n";
        return -1;
    }

    if(argv[1] == "edmond-karp")
    {
        mode = edmond_karp;
    }
    else if(argv[1] == "push-relabel")
    {
        mode = push_relabel;
    }
    else
    {
        std::cout << "Second argument needs to be either edmond-karp or push-relabel" << "\n";
        return -1;
    }

    // copy the image data
    Image img_object(argv[0]);

    // resize the image and store it in dest
    cv::Mat dest;
    cv::resize(img_object.image_data,dest,cv::Size(50,50),0,0,cv::INTER_CUBIC);
    img_object.image_data = dest;

    // show the image so that user selects reference background and foreground pixels 
    img_object.ShowImageToSelectReferencePixels();

    // convert the image to a flow network
    img_object.ConvertToFlowNetwork();
    Segmentation segmentation_object(img_object.network);

    // run the desired algorithm
    if(mode = edmond_karp)
    {
        segmentation_object.RunFordFulkersonAlgorithm(img_object.binary_mask);
    }
    else if(mode = push_relabel)
    {
        segmentation_object.RunPushRelabelAlgorithm(img_object.binary_mask);
    }

    // show the output
    img_object.ShowSegmentedImage();
    return 0;
}