#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <opencv2/opencv.hpp>
#include "flow_network.hpp"
#include <string>

using type = uint8_t;

class Image
{
    public :
        cv::Mat image_data;
        FlowNetwork network;
        Mask binary_mask;
        std::array<int,2> source_cell;
        std::array<int,2> sink_cell;

        void ConvertToFlowNetwork();
        void LoadImage(std::string);
        void ShowImageToSelectReferencePixels();
        void ShowSegmentedImage();
        Image(std::string);
        Image() = default;
    private : 
        void AddEdgesToNeighbouringNodes(int,int);
        void AddEdge(int i,int j,int u,int v);
        inline int CellCoordsToNodeNum(int u, int v);
        inline float CalculateWeight(int i, int j, int u, int v);
        bool IsWithinImage(int,int);
};

#endif