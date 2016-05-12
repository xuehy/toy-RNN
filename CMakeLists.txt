cmake_minimum_required(VERSION 2.6)

project(rnn)

# Flags
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3 -msse4.2 -std=c++11")

find_package( OpenMP REQUIRED)
find_package( CBLAS REQUIRED)
find_package(CUDA REQUIRED)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} ${CBLAS_CXX_FLAGS}")

set(CMAKE_BUILD_TYPE Release)


file(GLOB PRE_SRC_FILES "*.cpp")
add_executable(rnn ${PRE_SRC_FILES})

target_link_libraries(rnn cblas)


