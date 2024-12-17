include_directories(${HEAD_PATH})

add_subdirectory(common)
add_subdirectory(tensor)
add_subdirectory(layers)
add_subdirectory(models)
add_subdirectory(datasets)
add_subdirectory(loss)
add_subdirectory(optimizer)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

set(EXECUTABLE_OUTPUT_PATH ${EXEC_PATH})
add_executable(main main.cu)
add_executable(test1 test.cu)
add_executable(train1 train.cu)
target_compile_options(test1 PRIVATE -Xcompiler -O3 -lineinfo)
target_compile_options(train1 PRIVATE -Xcompiler -O3 -lineinfo)

set(HDF_HOME /usr/local/HDF_Group/HDF5/1.14.5)

target_link_libraries(main PRIVATE tensor common layers models)
target_include_directories(test1 PRIVATE ${HDF_HOME}/include)
target_link_directories(test1 PRIVATE ${HDF_HOME}/lib)
target_include_directories(train1 PRIVATE ${HDF_HOME}/include)
target_link_directories(train1 PRIVATE ${HDF_HOME}/lib)


target_link_libraries(test1 common tensor layers models datasets
    -lhdf5 -lhdf5_cpp)
target_link_libraries(train1 common tensor layers models datasets losses opts
    -lhdf5 -lhdf5_cpp)

add_custom_target(run
    COMMAND main   
    DEPENDS main  
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

set(PARAMS_DIR /home/tsyhahaha/default)
add_custom_target(test
    COMMAND test1   ${PARAMS_DIR}
    DEPENDS test1
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

set(SAVE_DIR /home/tsyhahaha/best_model)
add_custom_target(train
    COMMAND train1 ${SAVE_DIR}
    DEPENDS train1
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
