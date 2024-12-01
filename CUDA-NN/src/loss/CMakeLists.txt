aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRC_LIST)
add_library(losses ${SRC_LIST})

target_link_libraries(losses PRIVATE tensor common kernels)