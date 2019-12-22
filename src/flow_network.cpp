#include "flow_network.hpp"

#include <queue>
#include <stack>
#include <set>
#include <map>
#include <vector>
#include <assert.h>
#include <algorithm>





Mask::Mask(int size, bool value):
mask(std::vector<bool>(size, value))
{
    
}

AdjacencyList::AdjacencyList(int num_of_nodes)
{
    adjacency_list.resize(num_of_nodes);
}


FlowNetwork::FlowNetwork(int num_of_nodes)
{
    graph.adjacency_list.resize(num_of_nodes);
}

void FlowNetwork::AddEdge(int u, int v, capacity_type w)
{
    // first edge needs to be added
    graph.adjacency_list[u][v]= edge_values({static_cast<capacity_type>(w),static_cast<capacity_type>(0)});
    if(graph.adjacency_list[v].find(u) == graph.adjacency_list[v].end())
    {
         graph.adjacency_list[v][u]= edge_values({static_cast<capacity_type>(0),static_cast<capacity_type>(0)});
    }
}

void FlowNetwork::RemoveEdge(int u, int v)
{
    graph.adjacency_list[u].erase(v);
}



void FlowNetwork::AugmentFlow(Path& p, Flow& f)
{
    int i = p.path.size()-1;
    while(p.path[i] != sink)
    {
        int u = p.path[i];
        int v = p.path[i-1];
        graph.adjacency_list[u][v][flow] += f.flow;
        graph.adjacency_list[v][u][flow] -= f.flow;
        i -=1;
    }
}


void FlowNetwork::FindAugmentingPath(Path& augmenting_path, std::set<int>& reachable_nodes)
{
    // find the path from source to sink using BFS
    std::stack<int> bfs_queue;
    std::map<int,int> visited;

    bfs_queue.push(source);
    visited.insert({source,source});
    while(!bfs_queue.empty())
    {
        int present_node = bfs_queue.top();
        bfs_queue.pop();
        if(present_node == sink)
        {
            augmenting_path.path.reserve(visited.size());
            ReturnBFSPath(augmenting_path, visited);
            break;
        }
        for(auto& next_node_values:graph.adjacency_list[present_node])
        {
            if(visited.find(next_node_values.first) == visited.end())
            {
                if((next_node_values.second[capacity]-next_node_values.second[flow]) > 0)
                {
                    bfs_queue.push(next_node_values.first);
                    visited.emplace(next_node_values.first, present_node);
                }    
            }
        }
    }

    for(auto& visited_node:visited)
    {   
        reachable_nodes.insert(visited_node.first);
    }

}

void FlowNetwork::ReturnBFSPath(Path& p,const std::map<int,int>& visited)
{
    int next_node = sink;
    p.path.emplace_back(next_node);
    while(next_node != source)
    {  
        auto it = visited.find(next_node);
        next_node = it->second;
        p.path.emplace_back(next_node);
    }
    //path is in reversed order

}

void FlowNetwork::FindTheFlowInAugmentingPath(const Path& p,Flow& f_p)
{
    assert(p.path.front()==sink && p.path.back()==source);
    f_p.flow = graph.adjacency_list[p.path[p.path.size()-1]][p.path[p.path.size()-2]][capacity];
    for(int i=p.path.size()-1;i>0;--i)
    { 
        int u = p.path[i];
        int v = p.path[i-1];
        f_p.flow = std::min(f_p.flow, graph.adjacency_list[u][v][capacity]-graph.adjacency_list[u][v][flow]);
    }
}

uint FlowNetwork::NumOfVertices()
{
    return graph.adjacency_list.size();
}


