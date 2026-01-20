cmake_minimum_required(VERSION 3.31)
project(PFP)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")


include(FetchContent)

add_subdirectory(external/scalable-distributed-string-sorting)

FetchContent_Declare(
        kadis
        GIT_REPOSITORY https://github.com/MichaelAxtmann/KaDiS.git
        GIT_BRANCH main
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

add_executable(PFP main.cpp)

target_link_libraries(PFP PRIVATE kamping::kamping CLI11::CLI11 kadis)
target_link_libraries(PFP PRIVATE sort_caller_lib)

