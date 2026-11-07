cmake_minimum_required(VERSION 3.20)
project(FraudNet VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
# Some third-party projects still declare an old cmake_minimum_required.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

include(FetchContent)
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")
set(INSTALL_GTEST OFF CACHE INTERNAL "")
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_Declare(nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3 GIT_SHALLOW TRUE)
# cpp-httplib is header-only: fetch the sources but skip its CMakeLists.
FetchContent_Declare(httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG v0.18.7 GIT_SHALLOW TRUE
  SOURCE_SUBDIR no_cmake_here)
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2 GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(nlohmann_json httplib googletest)

configure_file(src/version.hpp.in ${CMAKE_BINARY_DIR}/generated/fraudnet/version.hpp @ONLY)

if(MSVC)
  set(FRAUDNET_WARNINGS /W4 /permissive- /utf-8 /external:W0)
else()
  set(FRAUDNET_WARNINGS -Wall -Wextra -Wpedantic)
endif()

add_library(fraudnet_core STATIC
  src/generator.cpp src/csv.cpp src/features.cpp src/scaler.cpp src/mlp.cpp
  src/trainer.cpp src/metrics.cpp src/model.cpp src/pipeline.cpp src/api.cpp
  src/server.cpp src/util.cpp)
target_include_directories(fraudnet_core PUBLIC src ${CMAKE_BINARY_DIR}/generated)
target_include_directories(fraudnet_core SYSTEM PUBLIC ${httplib_SOURCE_DIR})
target_link_libraries(fraudnet_core PUBLIC nlohmann_json::nlohmann_json)
target_compile_options(fraudnet_core PRIVATE ${FRAUDNET_WARNINGS})
target_compile_definitions(fraudnet_core PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS
  FRAUDNET_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
if(WIN32)
  target_compile_definitions(fraudnet_core PUBLIC _WIN32_WINNT=0x0A00)
  target_link_libraries(fraudnet_core PUBLIC ws2_32)
endif()

add_executable(fraudnet src/main.cpp)
target_link_libraries(fraudnet PRIVATE fraudnet_core)
target_compile_options(fraudnet PRIVATE ${FRAUDNET_WARNINGS})

enable_testing()
add_executable(fraudnet_tests
  tests/test_mlp.cpp tests/test_metrics.cpp tests/test_scaler.cpp
  tests/test_model.cpp tests/test_api.cpp tests/test_training.cpp tests/test_generator.cpp)
target_link_libraries(fraudnet_tests PRIVATE fraudnet_core GTest::gtest_main)
target_compile_options(fraudnet_tests PRIVATE ${FRAUDNET_WARNINGS})
include(GoogleTest)
gtest_discover_tests(fraudnet_tests DISCOVERY_TIMEOUT 120)
