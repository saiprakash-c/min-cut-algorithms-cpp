#include "flow_network.hpp"
#include "segmentation.hpp"

#include <set>

#include <iostream>
#include <chrono>

using namespace std::chrono;

//constructor
Segmentation::Segmentation(FlowNetwork& graph_input):
graph(graph_input)
{

}

//ford-fulckerson algorithm
void Segmentation::RunFordFulkersonAlgorithm(Mask& binary_mask)
{
    Flow f;
    f.flow = 0;
    Flow f_p;
    f_p.flow = 0;
    int64_t num_of_iterations = 0;
    auto start = std::chrono::system_clock::now();
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
            f.flow += f_p.flow;
        }
        
        num_of_iterations ++; 
    }
    auto end = std::chrono::system_clock::now();
    std::cout << "Flow:" << f.flow << std::endl;
    std::cout << "Number of iterations :" << num_of_iterations << std::endl;
    std::cout << "Average ms per iteration :" << (double)duration_cast<milliseconds>(end-start).count()/num_of_iterations << std::endl;
    std::cout << "Total time :" << duration_cast<milliseconds>(end-start).count() << std::endl;     
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
    std::unordered_set<int> excess_nodes;
    InitializePreFlow(excess_nodes);
    bool check_max_height_node = true;
    int max_height_node;
    int next_node;

    auto start = std::chrono::system_clock::now();
    int64_t num_of_iterations = 0;
    // int64_t height_to_increase = INT64_MAX;
    // while loop 
    while(true)
    {
        if(check_max_height_node && excess_nodes.size()!=0)
        {
            // excess_nodes.clear();
            // ReturnNodesHavingExcess(excess_nodes);
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
            Push(max_height_node, next_node, excess_nodes);
            check_max_height_node = true;
        }
        else
        {
            IncrementHeightByOne(max_height_node);
            check_max_height_node = false;
        }  
        num_of_iterations ++;
        // if(height[max_height_node] > 2*graph.NumOfVertices()-1){
        //     std::cout << "Something is wrong" << std::endl;
        // }
    }
    auto end = std::chrono::system_clock::now();
    std::cout << "Flow :" << excess[graph.sink] << std::endl;
    std::cout << "Number of iterations :" << num_of_iterations << std::endl;
    std::cout << "Average ms per iteration :" << (double)duration_cast<milliseconds>(end-start).count()/num_of_iterations << std::endl;
    std::cout << "Total time :" << duration_cast<milliseconds>(end-start).count() << std::endl;  


}

void Segmentation::InitializePreFlow(std::unordered_set<int>& excess_nodes)
{
    for(auto& edge_info:graph.graph.adjacency_list[graph.source])
    {
        Push(graph.source, edge_info.first, excess_nodes);
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

int Segmentation::FindNodeWithMaxHeight(const std::unordered_set<int>& excess_nodes)
{
    int max_height_node;
    int max_height = -1;
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
    // height_to_increase = INT64_MAX;
    for(auto edge_info:graph.graph.adjacency_list[node])
    {
        if(((edge_info.second[capacity]-edge_info.second[flow]) > 0) && (height[node] == height[edge_info.first]+1))
        {
            // height_to_increase = std::min(height_to_increase, height[edge_info.first]-height[node]);
            return edge_info.first;   
        }
    }
    return node;
}

bool Segmentation::Push(int u, int v, std::unordered_set<int>& excess_nodes)
{
    // if(u == v)
    // {
    //     return false;
    // }
    capacity_type residual_capacity = graph.graph.adjacency_list[u][v][capacity] - graph.graph.adjacency_list[u][v][flow];
    // if(residual_capacity == 0){
    //     return false;
    // }
    capacity_type delta = (u == graph.source) ? residual_capacity : std::min(excess[u],residual_capacity);
    if((v!= graph.sink) && (v!=graph.source)){excess_nodes.insert(v);}
    if(residual_capacity >=excess[u])
    {
        if((u!=graph.source) && (u!= graph.sink)){excess_nodes.erase(u);}
    }
    graph.graph.adjacency_list[u][v][flow] += delta;
    graph.graph.adjacency_list[v][u][flow] -= delta;
    excess[v] += delta;
    excess[u] -= delta;


    return true;
}

void Segmentation::IncrementHeightByOne(int u)
{
    height[u]  += 1;
}
