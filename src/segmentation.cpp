#include "flow_network.hpp"
#include "segmentation.hpp"

#include <set>



//constructor
Segmentation::Segmentation(FlowNetwork& graph_input):
graph(graph_input)
{

}

//ford-fulckerson algorithm
void Segmentation::RunFordFulkersonAlgorithm(Mask& binary_mask)
{
    Flow f_p;
    f_p.flow = 0;
    while(true)
    {
        Path p;
        std::set<int> reachable_nodes;
        graph.FindAugmentingPath(p, reachable_nodes);
        if(p.path.size() == 0)
        {
            FindMaskOfForeground(reachable_nodes, binary_mask);
            break;
        }
        else
        {
            graph.FindTheFlowInAugmentingPath(p,f_p);
            UpdateGraph(p, f_p);
        }
        
    }
     
}

//update original graph
void Segmentation::UpdateGraph(Path& p, Flow& f)
{
    //update the original graph
    graph.AugmentFlow(p ,f);
}

void Segmentation::FindMaskOfForeground(const std::set<int>& reachable_nodes, Mask& binary_mask)
{
    // find all the reachable vertices from source
    binary_mask.mask.resize(graph.NumOfVertices());
    std::fill(binary_mask.mask.begin(), binary_mask.mask.end(), false);
    for(auto node:reachable_nodes)
    {
        binary_mask.mask[node] = true;
    }
}

void Segmentation::RunPushRelabelAlgorithm(Mask& binary_mask)
{
    // define height and excess values
    int n= graph.NumOfVertices();
    excess.resize(n);
    height.resize(n);
    std::fill(excess.begin(),excess.end(),0);
    std::fill(height.begin(),height.end(),0);

    // initialization
    height[graph.source] = n;
    InitializePreFlow();
    bool check_max_height_node = true;
    std::vector<int> excess_nodes;
    int max_height_node;
    int next_node;
    // while loop 
    while(true)
    {
        if(check_max_height_node)
        {
            excess_nodes.clear();
            ReturnNodesHavingExcess(excess_nodes);
            max_height_node = FindNodeWithMaxHeight(excess_nodes);
        }
       
        if(excess_nodes.size() == 0)
        {
            std::set<int> reachable_nodes;
            Path p;
            graph.FindAugmentingPath(p, reachable_nodes);
            FindMaskOfForeground(reachable_nodes, binary_mask);
            break;
        }
        
        //push
        next_node = ReturnAnyDownHill(max_height_node);
        if(next_node != max_height_node)
        {
            Push(max_height_node, next_node);
            check_max_height_node = true;
        }
        //relabel
        else
        {
            IncrementHeightByOne(max_height_node);
            check_max_height_node = false;
        }
        
    }
}

void Segmentation::InitializePreFlow()
{
    for(auto& edge_info:graph.graph.adjacency_list[graph.source])
    {
        Push(graph.source, edge_info.first);
    }
}

void Segmentation::ReturnNodesHavingExcess(std::vector<int>& excess_nodes)
{
    for(int i=0;i<excess.size()-2;++i)
    {
        if(excess[i]>0)
        {
            excess_nodes.emplace_back(i);
        }
    }
}

int Segmentation::FindNodeWithMaxHeight(const std::vector<int>& excess_nodes)
{
    int max_height_node = excess_nodes[0];
    int max_height = height[max_height_node];
    for(auto node:excess_nodes)
    {
        if(height[node]>max_height)
        {
            max_height = height[node];
            max_height_node = node;
        }
    }
    return max_height_node;
}

int Segmentation::ReturnAnyDownHill(int node)
{
    for(auto edge_info:graph.graph.adjacency_list[node])
    {
        if(((edge_info.second[capacity]-edge_info.second[flow]) > 0) && (height[node] == height[edge_info.first]+1))
        {
            return edge_info.first;
        }
    }
    return node;
}

int Segmentation::Push(int u, int v)
{
    capacity_type residual_capacity = graph.graph.adjacency_list[u][v][capacity] - graph.graph.adjacency_list[u][v][flow];
    capacity_type delta = (u == graph.source) ? residual_capacity : std::min(excess[u],residual_capacity);
    graph.graph.adjacency_list[u][v][flow] += delta;
    graph.graph.adjacency_list[v][u][flow] -= delta;
    excess[v] += delta;
    excess[u] -= delta;
}

void Segmentation::IncrementHeightByOne(int u)
{
    height[u] +=1;
}
