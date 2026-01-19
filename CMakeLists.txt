cmake_minimum_required(VERSION 3.31)
project(PFP)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")


include(FetchContent)

add_subdirectory(external/scalable-distributed-string-sorting)


FetchContent_Declare(
        cli11
        GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
        GIT_TAG v2.6.1
)


FetchContent_MakeAvailable(cli11)

add_executable(PFP main.cpp)

target_link_libraries(PFP PRIVATE kamping::kamping CLI11::CLI11)
target_link_libraries(PFP PRIVATE sort_caller_lib)

