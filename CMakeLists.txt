cmake_minimum_required(VERSION 3.30)
project(ConnectedComponents)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

find_package(OpenCV REQUIRED)

add_executable(cc_label
    src/main.cpp
    src/VisualizationUtils.cpp
    src/ConnectedComponentsBFS.h
    src/ConnectedComponentsBFS.cpp
    src/ConnectedComponentsUF.h
    src/ConnectedComponentsUF.cpp
    src/IComponentDetector.h
    src/ComponentEvaluator.h
    src/ComponentEvaluator.cpp
)
target_link_libraries(cc_label PRIVATE ${OpenCV_LIBS})

message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
