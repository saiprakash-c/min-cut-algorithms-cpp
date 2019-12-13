#ifndef FLOW_NETWORK_HPP
#define FLOW_NETWORK_HPP


#include <map>
#include <set>
#include <vector>
enum edge_variables_for_original_graph{capacity,flow};
using  list = std::array<int,2>;
using capacity_type = int;
using edge_values = std::array<capacity_type,2>;


struct Path
{
    std::vector<int> path;
};

struct Flow
{
    capacity_type flow;
};

struct Mask
{
    std::vector<bool> mask;
    Mask(int, bool); 
    Mask() = default;
};

struct AdjacencyList
{
    std::vector<std::map<int, edge_values>> adjacency_list;
    AdjacencyList(int);
    AdjacencyList() = default;
};

class FlowNetwork
{
    public :
        AdjacencyList graph;
        int source;
        int sink;
        void AddEdge(int,int,capacity_type);
        void RemoveEdge(int,int);
        void AugmentFlow(Path&, Flow&);
        void FindAugmentingPath(Path&, std::set<int>&);
        void FindTheFlowInAugmentingPath(const Path&,Flow&);
        uint NumOfVertices();
        FlowNetwork(int);
        FlowNetwork() = default;


    private :
        bool CreateEdgeIfDoesntExist(int,int);
        void ReturnBFSPath(Path& p,const std::map<int,int>& visited);

};

#endif