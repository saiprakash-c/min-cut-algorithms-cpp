# project name
project(graph_cut)

# required cmake and c++ version
cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 11)

# set cmake variables

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

set(CMAKE_CXX_FLAGS "-g")

# set where to keep executables and libs
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)


# add subdirectory
add_subdirectory(src)