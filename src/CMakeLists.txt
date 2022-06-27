cmake_minimum_required(VERSION 3.15)
project(vtkOSPRayExample)

find_package(ospray REQUIRED)
find_package(VTK REQUIRED)

add_executable(
    main
    main.cpp
)

target_link_libraries(
    main
    PUBLIC
        ospray::ospray
        ${VTK_LIBRARIES}
)

install(
    TARGETS main
    DESTINATION bin
)


add_executable(
    vtkPDistributedDataFilterExample
    vtkPDistributedDataFilterExample.cpp
)

target_link_libraries(
    vtkPDistributedDataFilterExample
    PUBLIC
        ospray::ospray
        ${VTK_LIBRARIES}
)

install(
    TARGETS vtkPDistributedDataFilterExample
    DESTINATION bin
)
