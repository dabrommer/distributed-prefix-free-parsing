cmake_minimum_required(VERSION 3.26)
project(PFP)

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_CXX_FLAGS_DEBUG "-ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -DDEBUG -march=native -ggdb")

# Default to Release builds
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_UPPER)
message(STATUS "Build Flags: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER}}")

# Format target
find_program(CLANG_FORMAT "clang-format")
if(CLANG_FORMAT)

    file(GLOB_RECURSE SRC_FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
    )

    add_custom_target(fix-format
            COMMAND ${CLANG_FORMAT} -i -style=file ${SRC_FILES}
    )
else()
    message(WARNING "clang-format not found. The 'format' target will not be available.")
endif()

include(FetchContent)

add_subdirectory(external/scalable-distributed-string-sorting)
add_subdirectory(external/psac)

FetchContent_Declare(
        kadis
        GIT_REPOSITORY https://github.com/mschimek/KaDiS.git
        GIT_TAG 9141c628202e92aa342fd7e103c3350773aa746a
)

FetchContent_Declare(
        kamping
        GIT_REPOSITORY https://github.com/kamping-site/kamping.git
        GIT_TAG v0.2.1
)


FetchContent_Declare(
        cli11
        GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
        GIT_TAG v2.6.1
)
FetchContent_MakeAvailable(cli11 kamping kadis)

add_library(base)
target_include_directories(base PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(base PUBLIC kamping::kamping kadis CLI11::CLI11)

add_subdirectory(src)

# Set DDCX_lib_DIR to the directory containing DDCX_libConfig.cmake
find_package(DDCX_lib REQUIRED)


add_executable(PFP src/executables/pfp_runner.cpp)


target_link_libraries(PFP PRIVATE base)
target_link_libraries(PFP PRIVATE sort_caller_lib)
target_link_libraries(PFP PRIVATE DDCX_lib::DDCX_lib)
target_link_libraries(PFP PRIVATE sa_lcp_caller_lib)



