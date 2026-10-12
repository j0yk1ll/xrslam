cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
project(xrslam-extra)

add_compile_options(-fPIC)

include(SuperBuildDepends)
superbuild_extern(opencv)

# Optional USearch-backed place database. Keep the vector-search dependency
# outside xrslam-core and disabled in the default build.
option(XRSLAM_ENABLE_USEARCH "Enable USearch place database" OFF)
if(XRSLAM_ENABLE_USEARCH)
  superbuild_depend(usearch)
endif()

superbuild_depend(yaml-cpp)

# TorchScript inference backends are optional and depend on the same LibTorch.
# No Torch dependency is introduced into xrslam-core or the default build.
option(XRSLAM_ENABLE_PIDNET "Build the PIDNet semantic segmentation backend" OFF)
if(XRSLAM_ENABLE_PIDNET AND XRSLAM_IOS)
  message(FATAL_ERROR "PIDNet LibTorch backend is currently desktop-only")
endif()
if(NOT XRSLAM_IOS AND (XRSLAM_ENABLE_EIGENPLACES OR XRSLAM_ENABLE_PIDNET))
  find_package(Torch CONFIG REQUIRED)
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

if(XRSLAM_ENABLE_USEARCH AND TARGET depends::usearch)
  add_library(xrslam-extra-usearch-place-database
    ${XRSLAM_EXTRA_INCLUDE_DIR}/xrslam/extra/usearch_place_database.h
    ${XRSLAM_EXTRA_SOURCE_DIR}/xrslam/extra/usearch_place_database.cpp
  )
  target_compile_features(xrslam-extra-usearch-place-database PRIVATE cxx_std_17)
  target_link_libraries(xrslam-extra-usearch-place-database
    PUBLIC
      xrslam-extra-common
    PRIVATE
      depends::usearch
  )

  add_executable(xrslam-usearch-smoke
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/usearch_smoke.cpp
  )
  target_compile_features(xrslam-usearch-smoke PRIVATE cxx_std_17)
  target_link_libraries(xrslam-usearch-smoke
    PRIVATE
      xrslam-extra-usearch-place-database
  )
endif()

if(TARGET depends::opencv)
  add_library(xrslam-extra-opencv-image
    ${XRSLAM_EXTRA_INCLUDE_DIR}/xrslam/extra/image_undistorter.h
    ${XRSLAM_EXTRA_INCLUDE_DIR}/xrslam/extra/opencv_image.h
    ${XRSLAM_EXTRA_INCLUDE_DIR}/xrslam/extra/orb_local_descriptor_extractor.h
    ${XRSLAM_EXTRA_SOURCE_DIR}/xrslam/extra/opencv_image.cpp
    ${XRSLAM_EXTRA_SOURCE_DIR}/xrslam/extra/orb_local_descriptor_extractor.cpp
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

if(XRSLAM_ENABLE_EIGENPLACES AND NOT XRSLAM_IOS AND TARGET xrslam-extra-opencv-image)
  add_library(xrslam-extra-eigenplaces
    ${XRSLAM_EXTRA_INCLUDE_DIR}/xrslam/extra/eigenplaces_descriptor_extractor.h
    ${XRSLAM_EXTRA_SOURCE_DIR}/xrslam/extra/eigenplaces_descriptor_extractor.cpp
  )

  # PyTorch wheels expose their C++ ABI requirements through TORCH_CXX_FLAGS.
  # Keep those flags local to the optional EigenPlaces target.
  separate_arguments(XRSLAM_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")

  target_compile_features(xrslam-extra-eigenplaces PRIVATE cxx_std_20)
  target_compile_options(xrslam-extra-eigenplaces PRIVATE ${XRSLAM_TORCH_CXX_FLAGS})
  target_link_libraries(xrslam-extra-eigenplaces
    PUBLIC
      xrslam-extra-opencv-image
    PRIVATE
      ${TORCH_LIBRARIES}
  )

  add_executable(xrslam-eigenplaces-smoke
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/eigenplaces_smoke.cpp
  )
  target_compile_features(xrslam-eigenplaces-smoke PRIVATE cxx_std_20)
  target_link_libraries(xrslam-eigenplaces-smoke
    PRIVATE
      xrslam-extra-eigenplaces
      depends::opencv
  )
endif()

if(XRSLAM_ENABLE_PIDNET)
  if(NOT TARGET xrslam-extra-opencv-image)
    message(FATAL_ERROR "XRSLAM_ENABLE_PIDNET requires the OpenCV image backend")
  endif()
  add_library(xrslam-extra-pidnet
    ${XRSLAM_EXTRA_INCLUDE_DIR}/xrslam/extra/pidnet_segmenter.h
    ${XRSLAM_EXTRA_SOURCE_DIR}/xrslam/extra/pidnet_segmenter.cpp
    ${XRSLAM_EXTRA_INCLUDE_DIR}/xrslam/extra/pidnet_semantic_worker.h
    ${XRSLAM_EXTRA_SOURCE_DIR}/xrslam/extra/pidnet_semantic_worker.cpp
  )
  separate_arguments(XRSLAM_PIDNET_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
  target_compile_features(xrslam-extra-pidnet PRIVATE cxx_std_20)
  target_compile_options(xrslam-extra-pidnet PRIVATE ${XRSLAM_PIDNET_TORCH_CXX_FLAGS})
  target_link_libraries(xrslam-extra-pidnet
    PUBLIC
      xrslam-extra-opencv-image
    PRIVATE
      ${TORCH_LIBRARIES}
  )
  add_executable(xrslam-pidnet-smoke
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/pidnet_smoke.cpp
  )
  target_compile_features(xrslam-pidnet-smoke PRIVATE cxx_std_20)
  target_link_libraries(xrslam-pidnet-smoke
    PRIVATE xrslam-extra-pidnet depends::opencv)
endif()

set(XRSLAM_EXTRA_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE INTERNAL "")
