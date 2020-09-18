cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(cuda-lib LANGUAGES CXX CUDA)

file(GLOB_RECURSE SOURCES "src/*" "tests/*"  "external/benchmark/*")
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/test.cu") # it defines the main which conflicts with the main in tests

# INCLUDE PATHS
include_directories(src)
include_directories(external/)

# openssl
find_package(OpenSSL REQUIRED)
include_directories(${OPENSSL_INCLUDE_DIR})
message(STATUS "Using OpenSSL ${OPENSSL_VERSION} ${OPENSSL_INCLUDE_DIR}")

# gtest
add_subdirectory(external/googletest)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

# Enforce more verbose make 
option(CMAKE_VERBOSE_MAKEFILE ON)

# c++ 14
set (CMAKE_CXX_STANDARD 14)
 
# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
#set_target_properties( particles
#						PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
 
add_executable(cuda-lib ${SOURCES})
target_link_libraries(cuda-lib ${OPENSSL_LIBRARIES})
target_link_libraries(cuda-lib gtest)
target_link_libraries(cuda-lib curand)
target_link_libraries(cuda-lib gomp)
