cmake_minimum_required(VERSION 3.5) # CMake version check

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(mot-app VERSION 0.0.1 LANGUAGES C CXX)

file(GLOB Source_Files "${PROJECT_SOURCE_DIR}/src/*.cpp" "${PROJECT_SOURCE_DIR}/src/tracker/*.cpp" "${PROJECT_SOURCE_DIR}/src/image/*.c")

set(ALL_FILES ${Source_Files})

include_directories(${PROJECT_SOURCE_DIR}/inc)

add_executable(${PROJECT_NAME} ${ALL_FILES})

set(TARGET_BINARY mot-app)


message ("CMAKE_FIND_ROOT_PATH=${CMAKE_FIND_ROOT_PATH}")

get_filename_component(ALL_BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)

message("find include dir = ${ALL_BUILD_DIR}")

file(GLOB_RECURSE MI_SYS_CUSTOM_PATH "${ALL_BUILD_DIR}/mi_sys.h")

foreach(X ${MI_SYS_CUSTOM_PATH})
    if("${X}" MATCHES "libssc338q")
        get_filename_component(ONE_DIR "${X}" DIRECTORY)
        message(STATUS "libsssc338q lib path=${ONE_DIR}")
        get_filename_component(SECOND_DIR "${ONE_DIR}" DIRECTORY)
        include_directories(${ONE_DIR})
        include_directories(${SECOND_DIR})
    endif()
endforeach()

file(GLOB_RECURSE MI_LIB_CUSTOM_PATH "${ALL_BUILD_DIR}/libmi_sys.so")
    if("${MI_LIB_CUSTOM_PATH}" MATCHES "libssc338q")
        get_filename_component(ONE_DIR "${MI_LIB_CUSTOM_PATH}" DIRECTORY)
        message(STATUS "mi lib path=${ONE_DIR}")
        file(GLOB files ${ONE_DIR}/*.so)
        foreach(file ${files})
            target_link_libraries(${PROJECT_NAME} ${file})
            message(STATUS "target_link_libraries = ${file}")
        endforeach()
    endif()

file(GLOB_RECURSE OPENCV2_CUSTOM_PATH "${ALL_BUILD_DIR}/opencv.hpp")

foreach(X ${OPENCV2_CUSTOM_PATH})
    if("${X}" MATCHES "opencv4")
        get_filename_component(ONE_DIR "${X}" DIRECTORY)
        message(STATUS "opencv path=${ONE_DIR}")
        get_filename_component(SECOND_DIR "${ONE_DIR}" DIRECTORY)
        include_directories(${ONE_DIR})
        message(STATUS "opencv include_directories path=${ONE_DIR}")
        include_directories(${SECOND_DIR})
        message(STATUS "opencv include_directories path=${SECOND_DIR}")
    endif()
endforeach()


file(GLOB_RECURSE OPENCV2_CUSTOM_PATH "${ALL_BUILD_DIR}/dnn.hpp")

foreach(X ${OPENCV2_CUSTOM_PATH})
    if("${X}" MATCHES "opencv2")
        get_filename_component(ONE_DIR "${X}" DIRECTORY)
        message(STATUS "opencv path=${ONE_DIR}")
        get_filename_component(SECOND_DIR "${ONE_DIR}" DIRECTORY)
        include_directories(${ONE_DIR})
        message(STATUS "opencv include_directories path=${ONE_DIR}")
        include_directories(${SECOND_DIR})
        message(STATUS "opencv include_directories path=${SECOND_DIR}")
    endif()
endforeach()


file(GLOB_RECURSE highgui_CUSTOM_PATH "${ALL_BUILD_DIR}/highgui.hpp")

foreach(X ${highgui_CUSTOM_PATH})
    if("${X}" MATCHES "opencv2")
        get_filename_component(ONE_DIR "${X}" DIRECTORY)
        get_filename_component(SECOND_DIR "${ONE_DIR}" DIRECTORY)
        include_directories(${ONE_DIR})
        message(STATUS "opencv include_directories path=${ONE_DIR}")
        include_directories(${SECOND_DIR})
        message(STATUS "opencv include_directories path=${SECOND_DIR}")
    endif()
endforeach()

file(GLOB_RECURSE imgproc_CUSTOM_PATH "${ALL_BUILD_DIR}/imgproc.hpp")

foreach(X ${imgproc_CUSTOM_PATH})
    if("${X}" MATCHES "opencv2")
        get_filename_component(ONE_DIR "${X}" DIRECTORY)
        get_filename_component(SECOND_DIR "${ONE_DIR}" DIRECTORY)
        include_directories(${ONE_DIR})
        message(STATUS "opencv include_directories path=${ONE_DIR}")
        include_directories(${SECOND_DIR})
        message(STATUS "opencv include_directories path=${SECOND_DIR}")
    endif()
endforeach()

file(GLOB_RECURSE dnn_CUSTOM_PATH "${ALL_BUILD_DIR}/dnn.hpp")

foreach(X ${dnn_CUSTOM_PATH})
    if("${X}" MATCHES "opencv2")
        get_filename_component(ONE_DIR "${X}" DIRECTORY)
        get_filename_component(SECOND_DIR "${ONE_DIR}" DIRECTORY)
        include_directories(${ONE_DIR})
        message(STATUS "opencv include_directories path=${ONE_DIR}")
        include_directories(${SECOND_DIR})
        message(STATUS "opencv include_directories path=${SECOND_DIR}")
    endif()
endforeach()

file(GLOB_RECURSE imgcodecs_CUSTOM_PATH "${ALL_BUILD_DIR}/imgcodecs.hpp")

foreach(X ${imgcodecs_CUSTOM_PATH})
    if("${X}" MATCHES "opencv2")
        get_filename_component(ONE_DIR "${X}" DIRECTORY)
        get_filename_component(SECOND_DIR "${ONE_DIR}" DIRECTORY)
        include_directories(${ONE_DIR})
        message(STATUS "opencv include_directories path=${ONE_DIR}")
        include_directories(${SECOND_DIR})
        message(STATUS "opencv include_directories path=${SECOND_DIR}")
    endif()
endforeach()


find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})
add_definitions(${EIGEN3_DEFINITIONS})

target_link_libraries( ${PROJECT_NAME} ${Eigen3_LIBS} )

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_definitions(${OpenCV_DEFINITIONS})
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )


file(GLOB_RECURSE WRAPPER_CUSTOM_PATH "${ALL_BUILD_DIR}/cam_os_wrapper.h")
get_filename_component(ONE_DIR "${WRAPPER_CUSTOM_PATH}" DIRECTORY)
get_filename_component(SECOND_DIR "${ONE_DIR}" DIRECTORY)
include_directories(${ONE_DIR})
include_directories(${SECOND_DIR})

#find_path(WRAPPER_INCLUDE_DIR NAMES cam_os_wrapper.h)
#find_library(MOT_APP_LIBRARY0 NAMES opencv_imgcodecs PATHS "${ALL_BUILD_DIR}")
#message(STATUS "MOT_APP_LIBRARY0 path = ${MOT_APP_LIBRARY0}")

#find_library(MOT_APP_LIBRARY1 NAMES opencv_imgproc PATHS "${ALL_BUILD_DIR}")
#message(STATUS "MOT_APP_LIBRARY1 path = ${MOT_APP_LIBRARY1}")

#find_library(MOT_APP_LIBRARY2 NAMES opencv_core PATHS "${ALL_BUILD_DIR}")
#message(STATUS "MOT_APP_LIBRARY2 path = ${MOT_APP_LIBRARY2}")

#find_library(MOT_APP_LIBRARY3 NAMES ittnotify PATHS "${ALL_BUILD_DIR}")
#message(STATUS "MOT_APP_LIBRARY3 path = ${MOT_APP_LIBRARY3}")

#find_library(MOT_APP_LIBRARY4 NAMES jasper PATHS "${ALL_BUILD_DIR}")
#message(STATUS "MOT_APP_LIBRARY4 path = ${MOT_APP_LIBRARY4}")

#find_library(MOT_APP_LIBRARY5 NAMES   jpeg-turbo PATHS "${ALL_BUILD_DIR}")
#message(STATUS "MOT_APP_LIBRARY5 path = ${MOT_APP_LIBRARY5}")

#find_library(MOT_APP_LIBRARY6 NAMES png PATHS "${ALL_BUILD_DIR}")
#message(STATUS "MOT_APP_LIBRARY6 path = ${MOT_APP_LIBRARY6}")

#find_library(MOT_APP_LIBRARY7 NAMES tiff PATHS "${ALL_BUILD_DIR}")
#message(STATUS "MOT_APP_LIBRARY7 path = ${MOT_APP_LIBRARY7}")

#find_library(MOT_APP_LIBRARY8 NAMES webp PATHS "${ALL_BUILD_DIR}")
#message(STATUS "MOT_APP_LIBRARY8 path = ${MOT_APP_LIBRARY8}")

#find_library(MOT_APP_LIBRARY9 NAMES tegra_hal PATHS "${ALL_BUILD_DIR}")
#message(STATUS "MOT_APP_LIBRARY9 path = ${MOT_APP_LIBRARY9}")

#find_library(MOT_APP_LIBRARY11 NAMES zlib PATHS "${ALL_BUILD_DIR}")
#message(STATUS "MOT_APP_LIBRARY11 path = ${MOT_APP_LIBRARY11}")

#find_library(MOT_APP_LIBRARY12 NAMES opencv_dnn PATHS "${ALL_BUILD_DIR}")
#message(STATUS "MOT_APP_LIBRARY12 path = ${MOT_APP_LIBRARY12}")

add_custom_target(deploy ALL DEPENDS ${TARGET_BINARY})
add_custom_target(Profile ALL DEPENDS ${TARGET_BINARY})
set(BOARD_IP  10.42.0.214)
set(PORT_NO 5050)

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
   #add_compile_definition(${PROJECT_NAME} PUBLIC "$<$<CONFIG:Debug>:__DEBUG__>")  #since 3.12 version
   #add_compile_options(
   #                    "-Wall" "-Wpedantic" "-Wextra" "-fexceptions"
   #                    "$<$<CONFIG:DEBUG>:-O0;-g;-ggdb;>"
   # )

    SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -g")
    message("CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
    #add_custom_target(build_and_debug ALL DEPENDS ${TARGET_BINARY})
    #add_custom_target("deploy")
add_custom_command(
    TARGET deploy POST_BUILD
    #COMMAND scp ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_BINARY} root@${BOARD_IP}:/usr/bin
    COMMENT "Copy files..."
    COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_BINARY} "/nfs/rootfs_533/config/custom/mot-app"
    COMMAND sshpass -p "welcome" ssh -vv -y -f -L${PORT_NO}:localhost:${PORT_NO} root@${BOARD_IP} gdbserver :${PORT_NO} /config/custom/mot-app/${TARGET_BINARY} /config/custom/mot-app/config.ini
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    VERBATIM
    USES_TERMINAL
    )
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "Profile")
   #add_compile_options(
    #                   "-Wall" "-Wpedantic" "-Wextra" "-fexceptions"
     #                  "$<$<CONFIG:DEBUG>:-O0;-g;-ggdb;-pg;>"
    #)
    message("CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
     SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -pg")
add_custom_command(
    TARGET Profile POST_BUILD
    COMMENT "Copy files..."
    COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_BINARY} "/nfs/rootfs_533/config/custom/mot-app"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    VERBATIM
    USES_TERMINAL
    )
endif()