# Provide the GoogleTest dependency expected by xrslam-test/test.
# Prefer a locally installed GoogleTest to avoid unnecessary downloads.
if(NOT TARGET depends::googletest)
  find_package(GTest QUIET)
  if(TARGET GTest::gtest)
    set(_xrslam_gtest_target GTest::gtest)
  elseif(TARGET GTest::GTest)
    set(_xrslam_gtest_target GTest::GTest)
  else()
    include(FetchContent)
    FetchContent_Declare(
      depends-googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG v1.14.0
    )
    set(INSTALL_GTEST OFF CACHE BOOL "Do not install bundled GoogleTest" FORCE)
    FetchContent_MakeAvailable(depends-googletest)
    set(_xrslam_gtest_target gtest)
  endif()

  add_library(depends::googletest INTERFACE IMPORTED GLOBAL)
  target_link_libraries(depends::googletest INTERFACE ${_xrslam_gtest_target})
endif()
