cmake_minimum_required(VERSION 3.5) # CMake version check

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(mot-app VERSION 0.0.1 LANGUAGES C CXX)

#find_package(OpenCV REQUIRED PATHS "../oa8000-mft-a6a44c81fbcc376a8984c04439da8fd3d9f66355/rtos_mftfw_ajax_source/spu/share/opencv/include" NO_DEFAULT_PATH)
#message(STATUS "---OpenCV library status:")
#message(STATUS "    ---config: ${OpenCV_DIR}")
#message(STATUS "    ---version: ${OpenCV_VERSION}")
#message(STATUS "    ---libraries: ${OpenCV_LIBS}")
#message(STATUS "    ---include path: ${OpenCV_INCLUDE_DIRS}")
#include_directories(${OpenCV_INCLUDE_DIRS})

message ("CMAKE_FIND_ROOT_PATH=${CMAKE_FIND_ROOT_PATH}")

#function(my_check validator_result_var item)
#  if(NOT item MATCHES opencv2)
#    message("OpenCV include dir = ${item}")
#    set(${validator_result_var} FALSE PARENT_SCOPE)
#  endif()
#endfunction()

get_filename_component(ALL_BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)

message("find include dir = ${ALL_BUILD_DIR}")

file(GLOB_RECURSE OPENCV2_CUSTOM_PATH "${ALL_BUILD_DIR}/opencv.hpp")

foreach(X ${OPENCV2_CUSTOM_PATH})
    if("${X}" MATCHES "opencv4")
        get_filename_component(ONE_DIR "${X}" DIRECTORY)
        message(STATUS "opencv path=${ONE_DIR}")
        get_filename_component(SECOND_DIR "${ONE_DIR}" DIRECTORY)
        include_directories(${ONE_DIR})
        include_directories(${SECOND_DIR})
    endif()
endforeach()

#file(GLOB_RECURSE OPENCV2_CUSTOM_PATH "${ALL_BUILD_DIR}/includes.h")

#foreach(X ${OPENCV2_CUSTOM_PATH})
#    if("${X}" MATCHES "rtos_mftfw_ajax_source/spu")
#        get_filename_component(ONE_DIR "${X}" DIRECTORY)
#        message(STATUS "opencv path=${ONE_DIR}")
#        get_filename_component(SECOND_DIR "${ONE_DIR}" DIRECTORY)
#        include_directories(${ONE_DIR})
#        include_directories(${SECOND_DIR})
#    endif()
#endforeach()

#file(GLOB_RECURSE OPENCV2_CUSTOM_PATH "${ALL_BUILD_DIR}/board_reg.h")

#foreach(X ${OPENCV2_CUSTOM_PATH})
#    if("${X}" MATCHES "rtos_mftfw_ajax_source")
#        get_filename_component(ONE_DIR "${X}" DIRECTORY)
#        message(STATUS "opencv path=${ONE_DIR}")
#        get_filename_component(SECOND_DIR "${ONE_DIR}" DIRECTORY)
#        include_directories(${ONE_DIR})
#        include_directories(${SECOND_DIR})
#    endif()
#endforeach()

#file(GLOB_RECURSE OPENCV2_CUSTOM_PATH "${ALL_BUILD_DIR}/debugprintf.h")

#foreach(X ${OPENCV2_CUSTOM_PATH})
#    if("${X}" MATCHES "rtos_mftfw_ajax_source")
#        get_filename_component(ONE_DIR "${X}" DIRECTORY)
#        message(STATUS "opencv path=${ONE_DIR}")
#        get_filename_component(SECOND_DIR "${ONE_DIR}" DIRECTORY)
#        include_directories(${ONE_DIR})
#        include_directories(${SECOND_DIR})
#    endif()
#endforeach()

#find_path (result NAMES opencv2/opencv.hpp PATHS ${OPENCV2_CUSTOM_PATH})# VALIDATOR my_check)
#message("OpenCV include dir = ${OPENCV2_CUSTOM_PATH}")

#include_directories(${CMAKE_SOURCE_DIR}/lib/eigen/)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})
add_definitions(${EIGEN3_DEFINITIONS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_definitions(${OpenCV_DEFINITIONS})


file(GLOB Source_Files "${PROJECT_SOURCE_DIR}/src/*.cpp" "${PROJECT_SOURCE_DIR}/src/tracker/*.cpp" "${PROJECT_SOURCE_DIR}/src/image/*.c")

set(ALL_FILES ${Source_Files})

include_directories(${PROJECT_SOURCE_DIR}/inc)

add_executable(${PROJECT_NAME} ${ALL_FILES})

set(TARGET_BINARY mot-app)

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
   #add_compile_definition(${PROJECT_NAME} PUBLIC "$<$<CONFIG:Debug>:__DEBUG__>")  #since 3.12 version
   add_compile_options(
                       "-Wall" "-Wpedantic" "-Wextra" "-fexceptions"
                       "$<$<CONFIG:DEBUG>:-O0;-g3;-ggdb>"
    )
    message("CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
    #add_custom_target(build_and_debug ALL DEPENDS ${TARGET_BINARY})
    #add_custom_target("deploy")
endif()

add_custom_target(deploy ALL DEPENDS ${TARGET_BINARY})
set(BOARD_IP  10.42.0.214)
set(PORT_NO 5050)

add_custom_command(
    TARGET deploy POST_BUILD
    #COMMAND scp ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_BINARY} root@${BOARD_IP}:/usr/bin
    COMMENT "Copy files..."
    COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_BINARY} "/nfs/rootfs/config/custom/mot-app"
    COMMAND sshpass -p "password" ssh -vv -y -f -L${PORT_NO}:localhost:${PORT_NO} root@${BOARD_IP} 'gdbserver :${PORT_NO} /config/custom/mot-app/${TARGET_BINARY} ./config.ini'
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    VERBATIM
    USES_TERMINAL
    )


