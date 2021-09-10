cmake_minimum_required(VERSION 2.8)
#set(CMAKE_C_COMPILER "/usr/local/bin/gcc-7")
#set(CMAKE_CXX_COMPILER "/usr/local/bin/g++-7")
project(efanna2e)
include_directories(${PROJECT_SOURCE_DIR}/include)
#OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "no OpenMP supprot")
endif()

#OpenBLAS
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/module)
find_package(OpenBLAS REQUIRED)
include_directories(${OpenBLAS_INCLUDE_DIR})
set(BLAS_LIB ${OpenBLAS_LIB})
#include_directories(/usr/local/opt/openblas/include)
#set(/usr/local/opt/openblas/lib)

#external libraries like faiss
#include_directories(${PROJECT_SOURCE_DIR}/extern_libraries)
#link_directories(${PROJECT_SOURCE_DIR}/extern_libraries/faiss/lib)

#set(BOOST_ROOT /usr/local/Cellar/boost/1.74.0/)
find_package(BOOST)
if(Boost_FOUND)
    #message(Boost_INCLUDE_DIRS "${Boost_INCLUDE_DIRS}")
    #message(Boost_LIBRARY_DIRS "${Boost_LIBRARY_DIRS}")
    include_directories(${Boost_INCLUDE_DIRS})
    link_directories(${Boost_LIBRARY_DIRS})
endif()

add_definitions (-std=c++11 -O3 -lboost -march=native -Wall -DINFO)

add_subdirectory(src)
add_subdirectory(tests)
