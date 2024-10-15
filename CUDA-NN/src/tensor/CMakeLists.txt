aux_source_directory(./ TENSOR_SRC)
aux_source_directory(kernels KERNELS_SRC)

add_subdirectory(kernels)

add_library(tensor ${TENSOR_SRC})

target_link_libraries(tensor PRIVATE kernels common)


