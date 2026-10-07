if(NOT TARGET depends::usearch)
  FetchContent_Declare(
    depends-usearch
    GIT_REPOSITORY https://github.com/unum-cloud/USearch.git
    GIT_TAG        1e8a19253138510f35ee2c68dc641e5b5c83c7b5
  )
  FetchContent_GetProperties(depends-usearch)
  if(NOT depends-usearch_POPULATED)
    message(STATUS "Fetching USearch sources")
    FetchContent_Populate(depends-usearch)
    message(STATUS "Fetching USearch sources - done")
  endif()

  add_library(depends::usearch INTERFACE IMPORTED GLOBAL)
  target_include_directories(depends::usearch
    INTERFACE
      ${depends-usearch_SOURCE_DIR}/include
  )

  set(depends-usearch-source-dir ${depends-usearch_SOURCE_DIR}
      CACHE INTERNAL "" FORCE)
  mark_as_advanced(depends-usearch-source-dir)
endif()
