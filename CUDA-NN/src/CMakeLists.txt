include_directories(${HEAD_PATH})

add_subdirectory(common)
add_subdirectory(tensor)
add_subdirectory(layers)
add_subdirectory(models)
add_subdirectory(datasets)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")


set(EXECUTABLE_OUTPUT_PATH ${EXEC_PATH})
add_executable(main main.cu)

add_executable(test1 test.cu)
target_compile_options(test1 PRIVATE -Xcompiler -O3)

set(HDF_HOME /usr/local/HDF_Group/HDF5/1.14.5)

target_include_directories(test1 PRIVATE ${HDF_HOME}/include)
target_link_directories(test1 PRIVATE ${HDF_HOME}/lib)

target_link_libraries(main PRIVATE common tensor layers models)


target_link_libraries(test1 common tensor layers models datasets
    -lhdf5 -lhdf5_cpp)




add_custom_target(run
    COMMAND main   
    DEPENDS main  
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

set(PARAMS_DIR /home/tsyhahaha/best_model)
add_custom_target(test
    COMMAND test1   ${PARAMS_DIR}
    DEPENDS test1
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
