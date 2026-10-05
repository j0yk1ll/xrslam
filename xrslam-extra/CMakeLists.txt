cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
project(xrslam-extra)

add_compile_options(-fPIC)

include(SuperBuildDepends)
superbuild_extern(opencv)
superbuild_depend(yaml-cpp)

# XFeat is an optional experimental local-feature backend. Keep ONNX Runtime
# completely optional so the default XRSLAM build has no learned-feature
# dependency.
if(NOT XRSLAM_IOS)
  find_package(onnxruntime CONFIG QUIET)
endif()

set(XRSLAM_EXTRA_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(XRSLAM_EXTRA_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(xrslam-extra-common INTERFACE IMPORTED GLOBAL)
target_include_directories(xrslam-extra-common
  INTERFACE
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>
    $<BUILD_INTERFACE:${XRSLAM_EXTRA_INCLUDE_DIR}>
)
target_link_libraries(xrslam-extra-common
  INTERFACE
    xrslam-core
)

if(TARGET depends::opencv)
  add_library(xrslam-extra-opencv-image
    ${XRSLAM_EXTRA_INCLUDE_DIR}/xrslam/extra/image_undistorter.h
    ${XRSLAM_EXTRA_INCLUDE_DIR}/xrslam/extra/opencv_image.h
    ${XRSLAM_EXTRA_SOURCE_DIR}/xrslam/extra/opencv_image.cpp
  )
  target_link_libraries(xrslam-extra-opencv-image
    PUBLIC
      xrslam-extra-common
      depends::opencv
  )
endif()

if(TARGET depends::yaml-cpp)
  add_library(xrslam-extra-yaml-config
    ${XRSLAM_EXTRA_INCLUDE_DIR}/xrslam/extra/yaml_config.h
    ${XRSLAM_EXTRA_SOURCE_DIR}/xrslam/extra/yaml_config.cpp
  )
  target_link_libraries(xrslam-extra-yaml-config
    PUBLIC
      xrslam-extra-common
      depends::yaml-cpp
  )
endif()

set(XRSLAM_ONNXRUNTIME_TARGET "")
if(TARGET onnxruntime::onnxruntime)
  set(XRSLAM_ONNXRUNTIME_TARGET onnxruntime::onnxruntime)
elseif(TARGET onnxruntime)
  set(XRSLAM_ONNXRUNTIME_TARGET onnxruntime)
endif()

if(XRSLAM_ONNXRUNTIME_TARGET AND TARGET xrslam-extra-opencv-image)
  add_library(xrslam-extra-xfeat
    ${XRSLAM_EXTRA_INCLUDE_DIR}/xrslam/extra/xfeat_local_feature_backend.h
    ${XRSLAM_EXTRA_SOURCE_DIR}/xrslam/extra/xfeat_local_feature_backend.cpp
  )
  target_link_libraries(xrslam-extra-xfeat
    PUBLIC
      xrslam-extra-opencv-image
    PRIVATE
      ${XRSLAM_ONNXRUNTIME_TARGET}
  )
endif()

set(XRSLAM_EXTRA_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE INTERNAL "")
