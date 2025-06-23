cmake_minimum_required(VERSION 3.10)
project(KalmanProject)

set(CMAKE_CXX_STANDARD 17)

# Find Eigen (assumes it's installed in a standard location)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Add include directory for your headers (adjust as needed)
include_directories(../include)

# Main Kalman library
add_library(kalman Kalman.cpp)

# Demo executable
add_executable(DemoKalman DemoKalman.cpp)
target_link_libraries(DemoKalman kalman)

# Unit test executable
add_executable(UnitTest UnitTest/UnitTest.cpp)
target_link_libraries(UnitTest kalman)

# If using doctest as a header-only library, just include it in your test source