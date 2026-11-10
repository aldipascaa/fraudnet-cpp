cmake_minimum_required(VERSION 3.20)
project(FraudNet VERSION 1.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
# Some third-party projects still declare an old cmake_minimum_required.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

option(FRAUDNET_BUILD_TESTS "Build the GoogleTest suite" ON)
option(FRAUDNET_WARNINGS_AS_ERRORS "Treat warnings in FraudNet code as errors" OFF)

# ---- dependencies (fetched, pinned) ---------------------------------------------------------
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
FetchContent_MakeAvailable(nlohmann_json httplib)
if(FRAUDNET_BUILD_TESTS)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2 GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(googletest)
endif()

configure_file(src/core/version.hpp.in ${CMAKE_BINARY_DIR}/generated/fraudnet/version.hpp @ONLY)

# ---- warnings: one interface target applied to all FraudNet code -----------------------------
add_library(fraudnet_warnings INTERFACE)
if(MSVC)
  target_compile_options(fraudnet_warnings INTERFACE /W4 /permissive- /utf-8 /external:W0)
  if(FRAUDNET_WARNINGS_AS_ERRORS)
    target_compile_options(fraudnet_warnings INTERFACE /WX)
  endif()
else()
  target_compile_options(fraudnet_warnings INTERFACE -Wall -Wextra -Wpedantic)
  if(FRAUDNET_WARNINGS_AS_ERRORS)
    target_compile_options(fraudnet_warnings INTERFACE -Werror)
  endif()
endif()

# ---- fraudnet_core: data, features, model, training, inference. No HTTP, no CLI. -------------
add_library(fraudnet_core STATIC
  src/core/generator.cpp src/core/csv.cpp src/core/features.cpp src/core/feature_pipeline.cpp
  src/core/scaler.cpp src/core/mlp.cpp src/core/trainer.cpp src/core/metrics.cpp src/core/model.cpp
  src/core/predictor.cpp src/core/pipeline.cpp src/core/util.cpp)
add_library(fraudnet::core ALIAS fraudnet_core)
target_include_directories(fraudnet_core PUBLIC src/core ${CMAKE_BINARY_DIR}/generated)
target_link_libraries(fraudnet_core PUBLIC nlohmann_json::nlohmann_json PRIVATE fraudnet_warnings)
if(WIN32)
  target_compile_definitions(fraudnet_core PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
endif()

# ---- fraudnet_server: REST API + embedded web UI. Depends on core, never the other way. ------
# The ui/ folder is compiled into the binary at build time (pure-CMake script, no extra tools).
file(GLOB FRAUDNET_UI_FILES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/ui/*)
set(FRAUDNET_UI_CPP ${CMAKE_BINARY_DIR}/generated/ui_assets_data.cpp)
add_custom_command(
  OUTPUT ${FRAUDNET_UI_CPP}
  COMMAND ${CMAKE_COMMAND} -DUI_DIR=${CMAKE_SOURCE_DIR}/ui -DOUT=${FRAUDNET_UI_CPP}
          -P ${CMAKE_SOURCE_DIR}/cmake/embed_ui.cmake
  DEPENDS ${FRAUDNET_UI_FILES} ${CMAKE_SOURCE_DIR}/cmake/embed_ui.cmake
  COMMENT "Embedding ui/ into the binary"
  VERBATIM)

add_library(fraudnet_server STATIC src/server/api.cpp src/server/server.cpp src/server/ui_assets.cpp
  ${FRAUDNET_UI_CPP})
add_library(fraudnet::server ALIAS fraudnet_server)
target_include_directories(fraudnet_server PUBLIC src/server)
target_include_directories(fraudnet_server SYSTEM PUBLIC ${httplib_SOURCE_DIR})
target_link_libraries(fraudnet_server PUBLIC fraudnet_core PRIVATE fraudnet_warnings)
if(WIN32)
  target_compile_definitions(fraudnet_server PUBLIC _WIN32_WINNT=0x0A00)
  target_link_libraries(fraudnet_server PUBLIC ws2_32)
endif()
# The generated data file is not hand-written code: keep warnings on for it only via the target above.

# ---- fraudnet: command-line application ------------------------------------------------------
add_executable(fraudnet src/cli/main.cpp)
target_link_libraries(fraudnet PRIVATE fraudnet_server fraudnet_warnings)

# ---- tests ------------------------------------------------------------------------------------
if(FRAUDNET_BUILD_TESTS)
  enable_testing()
  add_executable(fraudnet_tests
    tests/test_mlp.cpp tests/test_metrics.cpp tests/test_scaler.cpp tests/test_model.cpp
    tests/test_features.cpp tests/test_predictor.cpp tests/test_api.cpp tests/test_training.cpp
    tests/test_generator.cpp)
  target_link_libraries(fraudnet_tests PRIVATE fraudnet_server GTest::gtest_main fraudnet_warnings)
  include(GoogleTest)
  # Tests read fixtures (tests/data, models/, ui/) relative to the source directory.
  gtest_discover_tests(fraudnet_tests DISCOVERY_TIMEOUT 120 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

# ---- install ----------------------------------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS fraudnet RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES models/fraudnet.json DESTINATION ${CMAKE_INSTALL_DATADIR}/fraudnet/models)
install(FILES README.md CHANGELOG.md DESTINATION ${CMAKE_INSTALL_DOCDIR})
