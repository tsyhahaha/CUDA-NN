aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRC_LIST)
add_library(datasets ${SRC_LIST})

target_link_libraries(datasets PRIVATE tensor common)