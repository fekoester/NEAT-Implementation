cmake_minimum_required(VERSION 3.16)
project(neat LANGUAGES CXX VERSION 2.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(NEAT_BUILD_TESTS "Build unit tests" ON)
option(NEAT_BUILD_VIEWER "Build the SFML viewer if SFML is available" ON)

find_package(Threads REQUIRED)

add_library(neat src/config.cpp src/genome.cpp src/network.cpp src/population.cpp src/io.cpp)
target_include_directories(neat PUBLIC include)
target_link_libraries(neat PUBLIC Threads::Threads)
target_compile_options(neat PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>)

add_library(neat_envs envs/envs.cpp)
target_link_libraries(neat_envs PUBLIC neat)

add_executable(neat_train apps/neat_train.cpp)
target_link_libraries(neat_train PRIVATE neat_envs)
add_executable(neat_play apps/neat_play.cpp)
target_link_libraries(neat_play PRIVATE neat_envs)

if(NEAT_BUILD_VIEWER)
  find_package(SFML 2.5 COMPONENTS graphics window system QUIET)
  if(SFML_FOUND)
    add_executable(neat_viewer apps/neat_viewer.cpp)
    target_link_libraries(neat_viewer PRIVATE neat_envs sfml-graphics sfml-window sfml-system)
    message(STATUS "SFML found: building neat_viewer")
  else()
    message(STATUS "SFML not found: skipping neat_viewer")
  endif()
endif()

if(NEAT_BUILD_TESTS)
  # Use an installed doctest (e.g. apt install doctest-dev) or download it.
  find_package(doctest QUIET)
  if(NOT doctest_FOUND)
    include(FetchContent)
    FetchContent_Declare(doctest
      URL https://github.com/doctest/doctest/archive/refs/tags/v2.4.11.tar.gz
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_GetProperties(doctest)
    if(NOT doctest_POPULATED)
      FetchContent_Populate(doctest)
    endif()
    add_library(doctest_header INTERFACE)
    target_include_directories(doctest_header INTERFACE ${doctest_SOURCE_DIR})
    add_library(doctest::doctest ALIAS doctest_header)
  endif()
  enable_testing()
  add_executable(neat_tests tests/test_main.cpp tests/test_genome.cpp tests/test_network.cpp
                            tests/test_population.cpp tests/test_envs.cpp)
  target_link_libraries(neat_tests PRIVATE neat_envs doctest::doctest)
  add_test(NAME neat_tests COMMAND neat_tests)
endif()
