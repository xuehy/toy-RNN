cmake_minimum_required(VERSION 2.6)

project(rnn)

# Flags
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3 -std=c++11")
set(CUDA_NVCC_FLAGS_RELEASE  -O3 -gencode arch=compute_30,code=sm_30 --expt-extended-lambda -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__)

include_directories(/opt/cuda/include)
link_directories(/opt/cuda/lib64)
find_package( OpenMP REQUIRED)
find_package( CBLAS REQUIRED)
find_package(CUDA REQUIRED)
find_package( Protobuf REQUIRED )
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} ${CBLAS_CXX_FLAGS}")

set(CMAKE_BUILD_TYPE Release)

file(GLOB CUDA_SRC_FILES "*.cu")
file(GLOB PRE_SRC_FILES "*.cpp" "*.cc" "*.hpp")
list(REMOVE_ITEM PRE_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/rnn.cpp)
cuda_compile(CUDA_TARGETS ${CUDA_SRC_FILES})
add_executable(rnn ${PRE_SRC_FILES} ${CUDA_TARGETS})

target_link_libraries(rnn cblas protobuf cublas cudart)


