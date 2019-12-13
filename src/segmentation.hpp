#ifndef SEGMENTATION_HPP
#define SEGMENTATION_HPP

#include "flow_network.hpp"
#include <vector>


class Segmentation{
public:
    Segmentation(FlowNetwork& graph);
    FlowNetwork graph;
    void RunFordFulkersonAlgorithm(Mask&);
    void RunPushRelabelAlgorithm(Mask&);

    std::vector<capacity_type> excess;
    std::vector<int> height;
private:
    void UpdateGraph(Path& path, Flow& flow);
    void FindMaskOfForeground(const std::set<int>& reachable_nodes, Mask&);

    void InitializePreFlow();
    void ReturnNodesHavingExcess(std::vector<int>& excess_nodes);
    int FindNodeWithMaxHeight(const std::vector<int>& excess_nodes);
    int ReturnAnyDownHill(int node);
    int Push(int u, int v);
    void IncrementHeightByOne(int u);
};

#endif //SEGMENTATION_HPP