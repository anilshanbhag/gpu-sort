# 
# cd build
# cmake -DCMAKE_CUDA_FLAGS="-arch=sm_61" ../
#
# OR ON MAC OS
# cmake -DCMAKE_CUDA_FLAGS="-arch=sm_61" -DOPENSSL_ROOT_DIR=/usr/local/opt/openssl ../

# Requires CMake 3.8
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# Project "cuda-lib" requires languages C++ and CUDA
project(cuda-lib LANGUAGES CXX CUDA)

#{txt,jpg,png} ${CMAKE_SOURCE_DIR}
#file(GLOB_RECURSE SOURCES RELATIVE "/home/elias/tmp" "src/*" "tests/*"  "external/benchmark/*")
file(GLOB_RECURSE SOURCES "src/*" "tests/*"  "external/benchmark/*")


###################
## INCLUDE PATHS ##
###################
include_directories(src)
include_directories(external/)

###################
## INCLUDE PATHS ##
###################
# If available, use pkg-config to find openssl
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
	pkg_search_module(OPENSSL REQUIRED openssl)
else()
	find_package(openssl REQUIRED)
endif(PkgConfig_FOUND)

include_directories(${OPENSSL_INCLUDE_DIR})
message(STATUS "Using OpenSSL ${OPENSSL_VERSION} ${OPENSSL_INCLUDE_DIR}")


####################################
## IF SETTING PREPROCESSOR MACROs ##
####################################
#set (VERSION_MAJOR 0)
#set (VERSION_MINOR 2)
#set (VERSION_PATCH 0)
## configure a header file to pass some of the CMake settings
## to the source code
#configure_file (
#  "${PROJECT_SOURCE_DIR}/TutorialConfig.h.in"
#  "${PROJECT_BINARY_DIR}/TutorialConfig.h"
#  )
# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
#include_directories("${PROJECT_BINARY_DIR}")
# // the configured options and settings for Tutorial. This would be placed in the TutorialConfig.h.in
##define Tutorial_VERSION_MAJOR @Tutorial_VERSION_MAJOR@
##define Tutorial_VERSION_MINOR @Tutorial_VERSION_MINOR@

# Enforce more verbose make 
option(CMAKE_VERBOSE_MAKEFILE ON)

########################
# GTEST 
########################
add_subdirectory(external/googletest)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

 
# Request that project is built with --std=c++11
# As this is a public compile feature anything that links to 
# particles will also build with -std=c++11
#target_compile_features(cuda-lib PUBLIC cxx_std_11)

#set (CMAKE_CXX_STANDARD 11)
 
# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
#set_target_properties( particles
#						PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
 
add_executable(cuda-lib ${SOURCES})
target_compile_features(cuda-lib PUBLIC cxx_std_11)
target_link_libraries(cuda-lib ${OPENSSL_LIBRARIES})
target_link_libraries(cuda-lib gtest)
target_link_libraries(cuda-lib curand)
if(NOT APPLE)
target_link_libraries(cuda-lib gomp)
endif()
 
 
#if(APPLE)
#  # We need to add the path to the driver (libcuda.dylib) as an rpath, 
#  # so that the static cuda runtime can find it at runtime.
#  set_property(TARGET particle_test 
#               PROPERTY
#               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
#endif()
