#!/usr/bin/env python3

import argparse

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.duration import Duration

from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener

from fr3_husky_msgs.action import TaskMove

PREDEFINED_TASK_RIGHT_POSES = {
    'threading_x': [0.798, -0.267, 0.761],
    'threading_z': [0.801, -0.294, 0.756],
    'square_x': [0.87, -0.210, 0.834],
    'square_z': [0.857, -0.222, 0.838],
    'threepiece_x': [0.816, -0.256, 0.871],
    'threepiece_z': [0.806, -0.250, 0.871],
}


def make_pose_stamped(pose, frame_id='world'):
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.header.stamp = Time(sec=0, nanosec=0)
    msg.pose.position.x = float(pose[0])
    msg.pose.position.y = float(pose[1])
    msg.pose.position.z = float(pose[2])

    if len(pose) == 3:
        qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
    else:
        qx, qy, qz, qw = map(float, pose[3:7])
        norm = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
        if norm < 1e-12:
            qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
        else:
            qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    msg.pose.orientation.x = qx
    msg.pose.orientation.y = qy
    msg.pose.orientation.z = qz
    msg.pose.orientation.w = qw
    return msg


class TaskMoveClient(Node):
    def __init__(self, arm='right', right_pose=None, left_pose=None, execution_time=3.0, frame_id='world', abs_target=True):
        super().__init__('task_move_client')
        self._action_name = '/fr3_husky_task_move'
        self._client = ActionClient(self, TaskMove, self._action_name)
        self._goal_handle = None
        self._result_future = None

        self._arm = arm
        self._right_pose = right_pose
        self._left_pose = left_pose
        self._execution_time = execution_time
        self._frame_id = frame_id
        self._abs_target = abs_target
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=False)

        self.get_logger().info(f'Waiting for action server: {self._action_name}')
        self._client.wait_for_server()
        self.get_logger().info(f'Connected to action server: {self._action_name}')

    def send_goal_and_wait(self):
        goal = TaskMove.Goal()
        goal.arm_names = self._arm
        goal.execution_time = float(self._execution_time)
        goal.abs = bool(self._abs_target)

        if self._arm == 'right':
            self._validate_pose(self._right_pose, 'right_pose')
            goal.target_poses = [make_pose_stamped(self._right_pose, self._frame_id)]
        elif self._arm == 'left':
            self._validate_pose(self._left_pose, 'left_pose')
            goal.target_poses = [make_pose_stamped(self._left_pose, self._frame_id)]
        elif self._arm in ('both', 'dual'):
            self._validate_pose(self._left_pose, 'left_pose')
            self._validate_pose(self._right_pose, 'right_pose')
            goal.target_poses = [
                make_pose_stamped(self._left_pose, self._frame_id),
                make_pose_stamped(self._right_pose, self._frame_id),
            ]
        else:
            self.get_logger().error("arm must be one of: left, right, both, dual")
            return

        send_goal_future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_goal_future)
        self._goal_handle = send_goal_future.result()

        if self._goal_handle is None:
            self.get_logger().error('Goal response is None')
            return
        if not self._goal_handle.accepted:
            self.get_logger().warn('TaskMove goal rejected')
            return

        self.get_logger().info('TaskMove goal accepted')
        self._result_future = self._goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, self._result_future)

        wrapped_result = self._result_future.result()
        if wrapped_result is None:
            self.get_logger().error('Result is None')
            return

        result = wrapped_result.result
        self.get_logger().info(f'TaskMove success: {result.success}')
        self.get_logger().info(result.message)
        if result.success:
            self._log_current_eef_positions()

    def cancel_goal(self):
        if self._goal_handle is None:
            return None
        return self._goal_handle.cancel_goal_async()

    def _validate_pose(self, pose, name):
        if pose is None or len(pose) not in (3, 7):
            self.get_logger().error(f'{name} must have either 3 elements [dx dy dz] or 7 elements [x y z qx qy qz qw]')
            rclpy.shutdown()

    def _lookup_eef_position(self, ee_frame):
        transform = self._tf_buffer.lookup_transform(
            self._frame_id,
            ee_frame,
            rclpy.time.Time(),
            timeout=Duration(seconds=1.0),
        )
        t = transform.transform.translation
        return (t.x, t.y, t.z)

    def _log_current_eef_positions(self):
        frames = []
        if self._arm == 'right':
            frames = ['right_fr3_hand_tcp']
        elif self._arm == 'left':
            frames = ['left_fr3_hand_tcp']
        elif self._arm in ('both', 'dual'):
            frames = ['left_fr3_hand_tcp', 'right_fr3_hand_tcp']

        for frame in frames:
            try:
                x, y, z = self._lookup_eef_position(frame)
                self.get_logger().info(
                    f'Current {frame} position in {self._frame_id}: '
                    f'[{x:.6f}, {y:.6f}, {z:.6f}]'
                )
            except Exception as exc:
                self.get_logger().warn(
                    f'Failed to get current position of {frame} in {self._frame_id}: {exc}'
                )


def run_task_move(arm='right', right_pose=None, left_pose=None, execution_time=3.0, frame_id='world', abs_target=False):
    rclpy.init()
    node = TaskMoveClient(
        arm=arm,
        right_pose=right_pose,
        left_pose=left_pose,
        execution_time=execution_time,
        frame_id=frame_id,
        abs_target=abs_target,
    )
    try:
        node.send_goal_and_wait()
    except KeyboardInterrupt:
        cancel_future = node.cancel_goal()
        if cancel_future is not None:
            rclpy.spin_until_future_complete(node, cancel_future, timeout_sec=2.0)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main(args=None):
    del args
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['left', 'right', 'both', 'dual'], default='right')
    parser.add_argument('--right-pose', type=float, nargs='+', default=[0.10, 0.0, 0.0])
    parser.add_argument('--left-pose', type=float, nargs='+', default=[0.10, 0.0, 0.0])
    parser.add_argument('--execution-time', type=float, default=3.0)
    parser.add_argument('--frame-id', default='odom')
    parser.add_argument('--abs', action='store_true', dest='abs_target', help='Treat target poses as absolute global poses. Default is delta pose from current global EEF pose.')
    parser.add_argument(
        '--task',
        choices=list(PREDEFINED_TASK_RIGHT_POSES.keys()),
        default=None,
        help='Use a predefined right-arm target pose and ignore --right-pose.',
    )
    cli_args = parser.parse_args()

    right_pose = cli_args.right_pose
    abs_target = cli_args.abs_target
    if cli_args.task is not None:
        right_pose = PREDEFINED_TASK_RIGHT_POSES[cli_args.task]
        abs_target = True

    run_task_move(
        arm=cli_args.arm,
        right_pose=right_pose,
        left_pose=cli_args.left_pose,
        execution_time=cli_args.execution_time,
        frame_id=cli_args.frame_id,
        abs_target=abs_target,
    )


if __name__ == '__main__':
    main()
