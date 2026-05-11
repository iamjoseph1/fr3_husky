cmake_minimum_required(VERSION 3.5)
project(fr3_husky_controller)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ros2_control_cmake REQUIRED)
set_compiler_options()
export_windows_symbols()


set(THIS_PACKAGE_INCLUDE_DEPENDS
    ament_index_cpp
    controller_interface
    generate_parameter_library
    hardware_interface
    pluginlib
    rclcpp
    rclcpp_action
    rclcpp_lifecycle
    franka_semantic_components
    Franka
    Eigen3
    geometry_msgs
    sensor_msgs
    std_msgs
    urdf
    urdfdom
    urdfdom_headers
    dyros_robot_controller
    fr3_husky_msgs
    franka_msgs
    nav_msgs
    realtime_tools
    tf2
    tf2_msgs
    control_msgs
    controller_manager_msgs
    action_msgs
    moveit_ros_planning_interface
    mujoco_ros_hardware
    srdfdom
)


find_package(ament_cmake REQUIRED)
find_package(OpenGL REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
foreach(Dependency IN ITEMS ${THIS_PACKAGE_INCLUDE_DEPENDS})
    find_package(${Dependency} REQUIRED)
endforeach()

set(MUJOCO_DIR "/opt/mujoco" CACHE PATH "MuJoCo SDK installation path")
set(MUJOCO_SEARCH_PREFIXES
    "${MUJOCO_DIR}"
    "$ENV{MUJOCO_DIR}"
    "$ENV{MUJOCO_HOME}"
    "/usr/local"
    "/usr"
)
find_path(MUJOCO_INCLUDE_DIR
    NAMES mujoco/mujoco.h
    PATHS ${MUJOCO_SEARCH_PREFIXES}
    PATH_SUFFIXES include
    REQUIRED
)
find_library(MUJOCO_LIBRARY
    NAMES mujoco
    PATHS ${MUJOCO_SEARCH_PREFIXES}
    PATH_SUFFIXES lib lib64
    REQUIRED
)

if(NOT TARGET mujoco::mujoco)
    add_library(mujoco::mujoco UNKNOWN IMPORTED)
    set_target_properties(mujoco::mujoco PROPERTIES
        IMPORTED_LOCATION "${MUJOCO_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_INCLUDE_DIR}"
    )
endif()



generate_parameter_library(fr3_controller_parameters
    src/fr3_controller_parameters.yaml
)

generate_parameter_library(fr3_husky_controller_parameters
    src/fr3_husky_controller_parameters.yaml
)

generate_parameter_library(fr3_action_controller_parameters
    src/fr3_action_controller_parameters.yaml
)

generate_parameter_library(fr3_husky_action_controller_parameters
    src/fr3_husky_action_controller_parameters.yaml
)

add_library(${PROJECT_NAME} SHARED
    src/utils/dyros_math.cpp
    src/utils/motion_primitives.cpp

    src/model/model_updater_base.cpp
    src/model/fr3_model_updater.cpp
    src/model/fr3_husky_model_updater.cpp

    src/fr3_action_controller.cpp
    src/fr3_husky_action_controller.cpp

    src/servers/action_server_base.cpp  
    src/servers/idle_control.cpp

    src/servers/fr3/gravity_compensation_action_server.cpp
    src/servers/fr3_husky/gravity_compensation_action_server.cpp
    src/servers/fr3/omega_haptic_action_server.cpp
    src/servers/fr3_husky/omega_haptic_action_server.cpp
    
    src/test_fr3_controller.cpp
    src/test_fr3_husky_controller.cpp
    src/servers/fr3/joint_trajectory_controller_action_server.cpp
    src/servers/fr3/move_to_joint_action_server.cpp
    src/servers/fr3_husky/joint_trajectory_controller_action_server.cpp
    src/servers/fr3_husky/move_to_joint_action_server.cpp
    src/servers/fr3_husky/gripper_command_action_server.cpp
    src/servers/fr3/vive_tracker_action_server.cpp
    src/servers/fr3_husky/vive_tracker_action_server.cpp
    src/servers/fr3_husky/husky_move_action_server.cpp
    src/servers/fr3_husky/husky_pedal_action_server.cpp
    src/servers/fr3_husky/apple_vision_pro_action_server.cpp
    src/servers/fr3_husky/sa_apple_vision_pro_action_server.cpp
    src/servers/fr3_husky/task_move_action_server.cpp
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(${PROJECT_NAME} PRIVATE -Wno-error=missing-braces)
endif()

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

target_include_directories(${PROJECT_NAME} PUBLIC
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include>
)

target_link_libraries(${PROJECT_NAME} PUBLIC
                      fr3_controller_parameters
                      fr3_husky_controller_parameters
                      mujoco::mujoco
                      ${OpenCV_LIBS}
                      pluginlib::pluginlib
                      rclcpp::rclcpp
                      rclcpp_action::rclcpp_action
                      rclcpp_lifecycle::rclcpp_lifecycle
                      urdf::urdf
                      dyros_robot_controller::dyros_robot_controller
                      Franka::Franka
)

pluginlib_export_plugin_description_file(controller_interface fr3_husky_plugin.xml)

ament_target_dependencies(${PROJECT_NAME} PUBLIC
    ament_index_cpp
    controller_interface
    generate_parameter_library
    hardware_interface
    pluginlib
    rclcpp
    rclcpp_action
    rclcpp_lifecycle
    Franka
    franka_semantic_components
    geometry_msgs
    sensor_msgs
    std_msgs
    urdf
    dyros_robot_controller
    fr3_husky_msgs
    franka_msgs
    nav_msgs
    realtime_tools
    tf2
    tf2_msgs
    controller_manager_msgs
    action_msgs
    moveit_ros_planning_interface
    mujoco_ros_hardware
    srdfdom
)

install(
    DIRECTORY include/
    DESTINATION include/fr3_husky_controller
)


install(TARGETS ${PROJECT_NAME} 
            fr3_controller_parameters 
            fr3_husky_controller_parameters
            fr3_action_controller_parameters
            fr3_husky_action_controller_parameters
        EXPORT export_${PROJECT_NAME}
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
)
    
install(DIRECTORY config launch rviz scripts
  DESTINATION share/${PROJECT_NAME}
)


ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(${THIS_PACKAGE_INCLUDE_DEPENDS})
ament_package()
