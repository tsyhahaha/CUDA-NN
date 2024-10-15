include_directories(${HEAD_PATH})

add_subdirectory(common)
add_subdirectory(tensor)
add_subdirectory(layers)
add_subdirectory(models)


set(EXECUTABLE_OUTPUT_PATH ${EXEC_PATH})
add_executable(main main.cu)

target_link_libraries(main common tensor layers models)


add_custom_target(run
    COMMAND main   
    DEPENDS main  
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
