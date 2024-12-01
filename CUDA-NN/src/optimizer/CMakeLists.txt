aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRC_LIST)
add_library(opts ${SRC_LIST})

target_link_libraries(opts PRIVATE tensor common kernels)