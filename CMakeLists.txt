cmake_minimum_required(VERSION 3.31)
project(PFP)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

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

add_library(base)
target_include_directories(base PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(base PUBLIC kamping::kamping kadis CLI11::CLI11)

add_subdirectory(src)

add_executable(PFP src/executables/pfp_runner.cpp)


target_link_libraries(PFP PRIVATE base)
target_link_libraries(PFP PRIVATE sort_caller_lib)



