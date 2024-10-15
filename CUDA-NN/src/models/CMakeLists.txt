aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRC_LIST)
add_library(models ${SRC_LIST})

target_link_libraries(models PRIVATE layers tensor common)