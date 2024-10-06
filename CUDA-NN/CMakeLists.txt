cmake_minimum_required(VERSION 3.0)
project(CUDA-NN LANGUAGES CUDA)

option(TESTS_ENABLED    "CUDA-NN build tests"         OFF)

aux_source_directory(src SRC)

set(HEAD_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(EXEC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)


add_subdirectory(src)





