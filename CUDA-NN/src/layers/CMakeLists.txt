aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRC_LIST)
add_library(layers ${SRC_LIST})

target_link_libraries(layers PRIVATE tensor common kernels)