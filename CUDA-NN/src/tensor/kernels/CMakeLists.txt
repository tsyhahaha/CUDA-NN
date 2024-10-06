aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRC_LIST)

if (NOT SRC_LIST)
    message(FATAL_ERROR "No source files found in ${KERNEL_DIR}")
endif()

add_library(kernels ${SRC_LIST})
