# include Cmake test
include(CTest)


# set names of variables
set(GRAPH_CUT_MAIN graph_cut_main)

# find libraries 
# find_library(LIB_OPENCV
#            NAMES opencv)

# find package
find_package(OpenCV 4 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

#link_directories(${OpenCV_LIBRARY_DIRS})

message(${OpenCV_LIBS})
message(${OpenCV_INCLUDE_DIRS})

# create libraries
add_library(graph_cut_library flow_network.cpp image.cpp segmentation.cpp)

# create executables
add_executable(${GRAPH_CUT_MAIN} main.cpp)

# link library
target_link_libraries(${GRAPH_CUT_MAIN} 
                        ${OpenCV_LIBS}
                        graph_cut_library
                        )
