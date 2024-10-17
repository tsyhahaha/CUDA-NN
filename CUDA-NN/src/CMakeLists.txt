include_directories(${HEAD_PATH})

add_subdirectory(common)
add_subdirectory(tensor)
add_subdirectory(layers)
add_subdirectory(models)


set(EXECUTABLE_OUTPUT_PATH ${EXEC_PATH})
add_executable(main main.cu)
add_executable(test1 test.cu)

target_link_libraries(main common tensor layers models)
target_link_libraries(test1 common tensor layers models
    -lhdf5 -lhdf5_cpp -ldl -lm)


add_custom_target(run
    COMMAND main   
    DEPENDS main  
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

set(PARAMS_DIR /home/course/taosiyuan241/best_model)
add_custom_target(test
    COMMAND test1   ${PARAMS_DIR}
    DEPENDS test1
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
