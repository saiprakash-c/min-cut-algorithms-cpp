#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "image.hpp"

void MouseHandler(int event, int x, int y, int flags, void* user_data)
{
    Image* object = (Image*)user_data;
    if(event == 1)
    {
        object->source_cell[0] = y; object->source_cell[1] = x;
    }
    if(event == 2)
    {
        object->sink_cell[0] = y; object->sink_cell[1] = x;
    }
}


Image::Image(std::string file_name):
image_data(cv::imread(file_name,0))
{

}

void Image::LoadImage(std::string file_name)
{
   image_data = cv::imread(file_name, 0); 
}


void Image::ShowImageToSelectReferencePixels()
{
    cv::namedWindow("Original Image", 1);
    cv::setMouseCallback("Original Image", MouseHandler, this);
    cv::imshow("Original Image", image_data);
    int k = cv::waitKey();
    if(k == 27)
    {
        cv::destroyAllWindows();
    }

}


void Image::ConvertToFlowNetwork()
{
    int num_of_nodes = image_data.rows * image_data.cols + 2 ;
    network.graph.adjacency_list.resize(num_of_nodes);
    // source and sink will be last two elements
    // connect source to all nodes
    network.source = num_of_nodes-2;
    network.sink = num_of_nodes-1;
    for(int i=0;i<image_data.rows;++i)
    {
        for(int j=0;j<image_data.cols;++j)
        {
            float weight = CalculateWeight(i,j,source_cell[0],source_cell[1]);
            network.AddEdge(network.source, CellCoordsToNodeNum(i,j), weight);

            weight = CalculateWeight(i,j,sink_cell[0],sink_cell[1]);
            network.AddEdge(CellCoordsToNodeNum(i,j),network.sink, weight);

            AddEdgesToNeighbouringNodes(i,j);
        }
    }
}



void Image::AddEdgesToNeighbouringNodes(int i, int j)
{
   AddEdge(i,j,i+1,j);
   AddEdge(i,j,i,j-1);
   AddEdge(i,j,i-1,j);
   AddEdge(i,j,i,j+1);
}

void Image::AddEdge(int i,int j,int u,int v)
{
    if(IsWithinImage(i,j) && IsWithinImage(u,v))
    {
        float weight = CalculateWeight(i,j,u,v);
        int node_1 = CellCoordsToNodeNum(i,j);
        int node_2 = CellCoordsToNodeNum(u,v);
        network.AddEdge(node_1,node_2, weight);
    }
}

inline int Image::CellCoordsToNodeNum(int u, int v)
{
    return u*image_data.cols+v; 
}

inline float Image::CalculateWeight(int i, int j, int u, int v)
{
    int diff = std::abs(image_data.at<type>(i,j)-image_data.at<type>(u,v));
    int sigma = 30;
    return 100*exp(-pow(diff,2)/(2*pow(sigma,2)));
}

bool Image::IsWithinImage(int i,int j)
{
    return (i<=image_data.rows-1) && (i>=0) && (j<=image_data.cols-1) && (j>=0);
}

void Image::ShowSegmentedImage()
{
    // convert binary_mask to cv::mat
    cv::Mat mask(image_data.rows, image_data.cols, CV_8UC1);
    for(int i=0;i<binary_mask.mask.size()-2;++i)
    {
        int u = floor(i/image_data.cols);
        int v = i%image_data.cols;
        if(binary_mask.mask[i]==true)
        {
            mask.at<uint8_t>(u,v) = 255;
        }
        else if(binary_mask.mask[i]==false)
        {
            mask.at<uint8_t>(u,v) = 0;
        }
        
    }
    cv::imwrite("output.png",mask);
}