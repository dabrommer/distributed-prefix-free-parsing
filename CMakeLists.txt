cmake_minimum_required(VERSION 3.31)
project(PFP)

set(CMAKE_CXX_STANDARD 23)

add_executable(PFP main.cpp)

include(FetchContent)
FetchContent_Declare(
        kamping
        GIT_REPOSITORY https://github.com/kamping-site/kamping.git
        GIT_TAG v0.2.0
)

FetchContent_Declare(
        cli11
        GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
        GIT_TAG v2.6.1
)


FetchContent_MakeAvailable(kamping cli11 )
target_link_libraries(PFP PRIVATE kamping::kamping CLI11::CLI11)
