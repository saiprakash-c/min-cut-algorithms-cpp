# min-cut algorithms for image segmentation 

The image segmentation is modelled as a min-cut problem and algorithms such as Edmond-Karps and Push Relabel are implemented to segment an image into foreground and background.

The image is first converted to a flow network.

## Conversion to flow network

Yuri et al [1] describes how to convert an image to a flow graph. For simplicity, we convert the image to a grayscale image first. 
If there are *n* pixels in the image, there would be *n+2* vertices. The additional two vertices correspond to source and sink. 
There are two types of edges in this graph - *n-links* and *t-links*. *n-links* connect neighboring pixel vertices. *t-links* connect
each pixel vertex to source and sink. The weight of a *n-link* should be big for similar pixels and small if they are different. It is given by 

$$ w(p,q) = 100exp(-\frac{I_p-I_q}{2{\sigma}^2}) $$

where *p*,*q* are neighbouring pixels. *I_x* is the intensity of pixel *x*. $\sigma$ is a parameter, taken as 30. 

For convenience, the user is asked to select one background pixel and one foreground pixel from the image. The intensities of source and sink are made equal to those of the background and foreground pixels respectively. The weight of a *t-link* then, is given by the same formula as above.  

## Edmond-Karp
1. Initialize the flow *f* to be 0
2. While true :
   * Perform Breadth First Search in residual network to find any
augmenting path
   * If there exists an augmenting path:
     * Find the max flow *f’* that can be sent through this path
     * *f = f + f’*
     * Update the residual network
   * else :
     * Find all the nodes reachable from source
     * break

The overall complexity is $O(VE^2)$

## Push Relabel
1. Initialize the heights and excess of all nodes(except source) to be zero
2. Initialize the height of source to be *n*
3. Saturate all the edges going from source
4. While true :
    * Find the minimum height node that has excess
    * Push the minimum of excess and residual capacity
        * If push is not possible, relabel the node

The overall complexity is $O(V^2\sqrt(E))$

## Prerequisites

Before you begin, ensure you have the following libraries installed.
<!--- These are just example requirements. Add, duplicate or remove as required --->

* [opencv](https://github.com/Itseez/opencv.git) 

## Installing

To install the project, follow these steps:

```
git clone https://github.com/saiprakash-c/min-cut-algorithms-cpp.git
cd min-cut-algorithms-cpp
mkdir {build,bin,lib}
cd build
cmake ..
make
```
## Usage

To use the project, follow these steps:

Run `graph_cut_main` from the bin directory which takes two arguments. 
```
./bin/graph_cut_main <address of the image> <edmond-karp/push-relabel>
```

For example :

```
./bin/graph_cut_main /home/saip/flower.jpeg push-relabel
```

The above code will take the image `/home/saip/flower.jpeg` and implement push-relabel algorithm to segment it. The output is stored is `output.png`.

## References

1. Boykov, Yuri, and Gareth Funka-Lea. "Graph cuts and efficient ND image segmentation." International journal of computer vision 70.2 (2006): 109-131.

## Contact

If you want to contact me, you can reach me at saip@umich.edu
