#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from fr3_husky_msgs.action import AppleVisionPro


class SAAppleVisionProClient(Node):
    def __init__(self):
        super().__init__('sa_apple_vision_pro_client')

        self._action_name = '/fr3_sa_AVP_tracker'
        self._client = ActionClient(self, AppleVisionPro, self._action_name)

        self._goal_handle = None

        self.get_logger().info(f'Waiting for action server: {self._action_name}')
        self._client.wait_for_server()
        self.get_logger().info(f'Connected to action server: {self._action_name}')

    def send_goal(self):
        goal = AppleVisionPro.Goal()
        goal.mode = 0
        goal.left_controller_ee_name = 'left_fr3_hand_tcp'
        goal.right_controller_ee_name = 'right_fr3_hand_tcp'
        goal.move_orientation = True
        goal.controller_pos_multiplier = 1.0
        goal.controller_ori_multiplier = 1.0

        self.get_logger().info('Sending SAAppleVisionPro goal')

        future = self._client.send_goal_async(goal)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal rejected')
            rclpy.shutdown()
            return

        self._goal_handle = goal_handle
        self.get_logger().info('Goal accepted')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result - is_completed: {result.is_completed}')

    def cancel_goal(self):
        if self._goal_handle is None:
            self.get_logger().warn('No active goal handle to cancel')
            return None

        self.get_logger().info('Canceling goal...')
        return self._goal_handle.cancel_goal_async()


def main(args=None):
    rclpy.init(args=args)
    node = SAAppleVisionProClient()
    node.send_goal()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        cancel_future = node.cancel_goal()
        if cancel_future is not None:
            rclpy.spin_until_future_complete(node, cancel_future, timeout_sec=2.0)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
