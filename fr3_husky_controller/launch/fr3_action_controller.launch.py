import os
import yaml
import tempfile
import xacro

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, Shutdown
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _load_yaml(package_name, rel_path):
    path = os.path.join(get_package_share_directory(package_name), rel_path)
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except EnvironmentError:
        return {}


def _parse_robot_side(raw_value):
    if raw_value is None:
        return []
    raw_value = raw_value.strip()
    if not raw_value:
        return []
    try:
        parsed = yaml.safe_load(raw_value)
    except yaml.YAMLError:
        parsed = raw_value
    if isinstance(parsed, list):
        values = parsed
    elif isinstance(parsed, str):
        if ',' in parsed:
            values = [item.strip() for item in parsed.split(',') if item.strip()]
        else:
            values = [parsed.strip()] if parsed.strip() else []
    else:
        values = [str(parsed).strip()]
    return [value for value in values if value]


def _normalize_robot_sides(robot_sides):
    if not robot_sides:
        return []
    normalized = [side.strip().lower() for side in robot_sides if side.strip()]
    if len(normalized) == 1 and normalized[0] == 'dual':
        return ['left', 'right']
    if 'dual' in normalized:
        raise RuntimeError("robot_side cannot mix 'dual' with other values.")
    return normalized


def _launch_setup(context, *args, **kwargs):
    robot_sides = _normalize_robot_sides(
        _parse_robot_side(LaunchConfiguration('robot_side').perform(context))
    )
    use_mujoco           = LaunchConfiguration('use_mujoco').perform(context)
    load_gripper         = LaunchConfiguration('load_gripper').perform(context)
    load_mobile          = LaunchConfiguration('load_mobile').perform(context)
    use_fake_hardware    = LaunchConfiguration('use_fake_hardware').perform(context)
    fake_sensor_commands = LaunchConfiguration('fake_sensor_commands').perform(context)
    namespace            = LaunchConfiguration('namespace').perform(context)
    launch_rviz          = LaunchConfiguration('launch_rviz').perform(context)
    launch_move_group    = LaunchConfiguration('launch_move_group').perform(context)

    if not robot_sides:
        raise RuntimeError("robot_side must be 'left', 'right', or 'dual'.")
    allowed = {'left', 'right'}
    if any(side not in allowed for side in robot_sides):
        raise RuntimeError("robot_side must be 'left', 'right', or 'dual'.")
    if len(set(robot_sides)) != len(robot_sides):
        raise RuntimeError("robot_side entries must be unique.")

    is_dual = len(robot_sides) == 2

    pkg_desc = get_package_share_directory('fr3_husky_description')
    pkg_ctrl = get_package_share_directory('fr3_husky_controller')

    #  URDF + MJCF paths
    if is_dual:
        urdf_path = os.path.join(pkg_desc, 'robots', 'dual_fr3.urdf.xacro')
        mjcf_path = os.path.join(pkg_desc, 'mjcf', 'dual_fr3.xml.xacro')
        xacro_mappings = {
            'ros2_control': 'true', 'with_sc': 'false', 'fix_finger': 'false',
            'hand': load_gripper, 'mobile': load_mobile,
            'use_mujoco': use_mujoco, 'use_fake_hardware': use_fake_hardware,
            'fake_sensor_commands': fake_sensor_commands,
        }
    else:
        urdf_path = os.path.join(pkg_desc, 'robots', 'single_fr3.urdf.xacro')
        mjcf_path = os.path.join(pkg_desc, 'mjcf', 'single_fr3.xml.xacro')
        xacro_mappings = {
            'ros2_control': 'true', 'with_sc': 'false', 'fix_finger': 'false',
            'side': robot_sides[0], 'hand': load_gripper, 'mobile': load_mobile,
            'use_mujoco': use_mujoco, 'use_fake_hardware': use_fake_hardware,
            'fake_sensor_commands': fake_sensor_commands,
        }

    robot_description = xacro.process_file(urdf_path, mappings=xacro_mappings).toprettyxml(indent='  ')

    # Controllers YAML
    controllers_yaml = os.path.join(pkg_ctrl, 'config', 'fr3_ros_controllers.yaml')

    #  Topic names
    joint_states_topic = 'dual_fr3/joint_states' if is_dual else f'{robot_sides[0]}_fr3/joint_states'
    if is_dual:
        jsp_sources = [joint_states_topic,
                       'left_franka_gripper/joint_states',
                       'right_franka_gripper/joint_states']
    else:
        jsp_sources = [joint_states_topic, f'{robot_sides[0]}_franka_gripper/joint_states']

    #  controller_manager parameters 
    cm_params = [controllers_yaml, {'robot_description': robot_description}]
    if use_mujoco.lower() == 'true':
        xacro_args = f' hand:={load_gripper} mobile:={load_mobile}'
        if not is_dual:
            xacro_args += f' side:={robot_sides[0]}'
        cm_params.extend([
            {'mujoco_scene_xacro_path': mjcf_path},
            {'mujoco_scene_xacro_args': xacro_args},
        ])
    controller_prefix = 'taskset -c 2,3' if use_mujoco.lower() == 'false' else None

    #  robot_state_publisher: direct subscription when using MuJoCo 
    rsp_remappings = [('joint_states', joint_states_topic)] if use_mujoco.lower() == 'true' else []

    #  Controller / Broadcaster names
    main_controller = 'dual_fr3_action_controller' if is_dual else f'{robot_sides[0]}_fr3_action_controller'
    franka_broadcaster_names = (
        ['left_franka_robot_state_broadcaster', 'right_franka_robot_state_broadcaster']
        if is_dual
        else [f'{robot_sides[0]}_franka_robot_state_broadcaster']
    )

    #  Gripper IPs
    side_ips = {'left': '172.16.5.5', 'right': '172.16.6.6'}

    #  Node list
    nodes = []

    if launch_rviz.lower() == 'true':
        nodes.append(Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='log',
            arguments=['-d', os.path.join(pkg_ctrl, 'rviz', 'fr3.rviz')],
            parameters=[{'robot_description': robot_description}],
        ))

    nodes += [
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            namespace=namespace,
            parameters=[{'robot_description': robot_description}],
            remappings=rsp_remappings,
            output='screen',
        ),
        Node(
            package='controller_manager',
            executable='ros2_control_node',
            namespace=namespace,
            parameters=cm_params,
            remappings=[('joint_states', joint_states_topic)],
            output='screen',
            on_exit=Shutdown(),
            # prefix=controller_prefix,
            prefix='taskset -c 2,3',
        ),
        # joint_state_publisher: only for real hardware (merges arm + gripper joint states).
        # For MuJoCo, robot_state_publisher subscribes directly via rsp_remappings above.
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            namespace=namespace,
            parameters=[{'source_list': jsp_sources, 'rate': 30}],
            output='screen',
            condition=UnlessCondition(PythonExpression(["'", LaunchConfiguration('use_mujoco'), "' == 'true'"])),
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            namespace=namespace,
            arguments=['joint_state_broadcaster', '--controller-manager-timeout', '60'],
            output='screen',
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            namespace=namespace,
            arguments=[main_controller, '--controller-manager-timeout', '60'],
            output='screen',
        ),
    ]

    # franka_robot_state_broadcaster: real hardware only (skip for fake or mujoco)
    for broadcaster_name in franka_broadcaster_names:
        nodes.append(
            Node(
                package='controller_manager',
                executable='spawner',
                namespace=namespace,
                arguments=[broadcaster_name],
                output='screen',
                condition=IfCondition(PythonExpression([
                    "'", LaunchConfiguration('use_fake_hardware'), "' == 'false' and '",
                    LaunchConfiguration('use_mujoco'), "' != 'true'",
                ])),
            )
        )

    # gripper: real hardware only (skip for mujoco), one per side
    gripper_sides = ['left', 'right'] if is_dual else [robot_sides[0]]
    for side in gripper_sides:
        nodes.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([FindPackageShare('franka_gripper'), 'launch', 'gripper.launch.py'])
                ),
                launch_arguments={
                    'robot_ip': side_ips[side],
                    'use_fake_hardware': use_fake_hardware,
                    'namespace': namespace,
                    'hand_prefix': side,
                }.items(),
                condition=IfCondition(PythonExpression([
                    "'", LaunchConfiguration('load_gripper'), "' == 'true' and '",
                    LaunchConfiguration('use_mujoco'), "' != 'true'",
                ])),
            )
        )

    # ---- Optional move_group (required by fr3_move_to_joint action server) ----
    if launch_move_group.lower() == 'true':
        pkg_desc    = get_package_share_directory('fr3_husky_description')
        pkg_moveit  = get_package_share_directory('fr3_husky_moveit_config')

        if is_dual:
            urdf_mg_path = os.path.join(pkg_desc, 'robots', 'dual_fr3.urdf.xacro')
            srdf_path    = os.path.join(pkg_desc, 'robots', 'dual_fr3.srdf.xacro')
            urdf_mg_map  = {'ros2_control': 'false', 'with_sc': 'false', 'fix_finger': 'false',
                            'hand': load_gripper, 'mobile': load_mobile}
            srdf_map     = {'hand': load_gripper, 'mobile': load_mobile}
            cfg_sub      = 'dual'
            ctrl_yaml    = os.path.join('config', 'dual', 'dual_fr3_controllers.yaml')
            jsp_src      = ['dual_fr3/joint_states']
        else:
            robot_side   = robot_sides[0]
            urdf_mg_path = os.path.join(pkg_desc, 'robots', 'single_fr3.urdf.xacro')
            srdf_path    = os.path.join(pkg_desc, 'robots', 'single_fr3.srdf.xacro')
            urdf_mg_map  = {'ros2_control': 'false', 'with_sc': 'false', 'fix_finger': 'false',
                            'side': robot_side, 'hand': load_gripper, 'mobile': load_mobile}
            srdf_map     = {'side': robot_side, 'hand': load_gripper, 'mobile': load_mobile}
            cfg_sub      = robot_side
            ctrl_yaml    = os.path.join('config', robot_side, 'single_fr3_controllers.yaml')
            jsp_src      = [f'{robot_side}_fr3/joint_states']

        mg_robot_desc = xacro.process_file(urdf_mg_path, mappings=urdf_mg_map).toprettyxml(indent='  ')
        mg_srdf       = xacro.process_file(srdf_path, mappings=srdf_map).toprettyxml(indent='  ')
        kinematics    = _load_yaml('fr3_husky_moveit_config', os.path.join('config', cfg_sub, 'kinematics.yaml'))
        ompl_yaml     = _load_yaml('fr3_husky_moveit_config', os.path.join('config', 'ompl_planning.yaml'))
        ctrl_mgr_yaml = _load_yaml('fr3_husky_moveit_config', ctrl_yaml)

        ompl_cfg = {
            'move_group': {
                'planning_plugin': 'ompl_interface/OMPLPlanner',
                'request_adapters':
                    'default_planner_request_adapters/AddTimeOptimalParameterization '
                    'default_planner_request_adapters/ResolveConstraintFrames '
                    'default_planner_request_adapters/FixWorkspaceBounds '
                    'default_planner_request_adapters/FixStartStateBounds '
                    'default_planner_request_adapters/FixStartStateCollision '
                    'default_planner_request_adapters/FixStartStatePathConstraints',
                'start_state_max_bounds_error': 0.1,
            }
        }
        ompl_cfg['move_group'].update(ompl_yaml)

        nodes.append(Node(
            package='moveit_ros_move_group',
            executable='move_group',
            namespace=namespace,
            output='screen',
            parameters=[
                {'robot_description': mg_robot_desc},
                {'robot_description_semantic': mg_srdf},
                kinematics,
                ompl_cfg,
                {'moveit_manage_controllers': False,
                 'trajectory_execution.allowed_execution_duration_scaling': 1.2,
                 'trajectory_execution.allowed_goal_duration_margin': 0.5,
                 'trajectory_execution.allowed_start_tolerance': 0.01},
                {'moveit_simple_controller_manager': ctrl_mgr_yaml,
                 'moveit_controller_manager':
                     'moveit_simple_controller_manager/MoveItSimpleControllerManager'},
                {'publish_planning_scene': True, 'publish_geometry_updates': True,
                 'publish_state_updates': True, 'publish_transforms_updates': True},
            ],
        ))

        # MuJoCo: bridge {side}_fr3/joint_states → joint_states for move_group's scene monitor
        if use_mujoco.lower() == 'true':
            nodes.append(Node(
                package='joint_state_publisher',
                executable='joint_state_publisher',
                name='joint_state_publisher_moveit',
                namespace=namespace,
                parameters=[{'source_list': jsp_src, 'rate': 30}],
                output='screen',
            ))

    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('robot_side',        default_value='left',  description="Robot side: left, right, or dual"),
        DeclareLaunchArgument('namespace',         default_value='',      description='Namespace for the robot'),
        DeclareLaunchArgument('load_gripper',      default_value='true',  description='Load gripper (true/false)'),
        DeclareLaunchArgument('load_mobile',       default_value='false', description='Load mobile base (true/false)'),
        DeclareLaunchArgument('use_mujoco',        default_value='false', description='Use MuJoCo hardware interface'),
        DeclareLaunchArgument('use_fake_hardware', default_value='false', description='Use fake hardware'),
        DeclareLaunchArgument('fake_sensor_commands', default_value='false', description='Fake sensor commands'),
        DeclareLaunchArgument('launch_rviz',         default_value='true',  description='Launch RViz'),
        DeclareLaunchArgument('launch_move_group',   default_value='true', description='Launch move_group (needed for fr3_move_to_joint)'),
        OpaqueFunction(function=_launch_setup),
    ])
