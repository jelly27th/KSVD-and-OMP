cmake_minimum_required(VERSION 3.8)

project(KSVD)

find_package(OpenCV 4.5.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(OpenMP REQUIRED)

include_directories(_omp)
file(GLOB _OMP CONFIGURE_DEPENDS _omp/*.cpp)
include_directories(ksvd)   
file(GLOB KSVD CONFIGURE_DEPENDS ksvd/*.cpp)
include_directories(utils)
file(GLOB UTILS CONFIGURE_DEPENDS utils/*.cpp)

add_executable(KSVD main.cpp ${_OMP} ${KSVD} ${UTILS})

target_link_libraries(KSVD PUBLIC ${OpenCV_LIBS} OpenMP::OpenMP_CXX)