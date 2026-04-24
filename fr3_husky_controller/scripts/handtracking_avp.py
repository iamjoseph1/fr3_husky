#!/usr/bin/env python3
import math
import socket
from dataclasses import dataclass, field

import rclpy
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int32MultiArray


def parse_packet(text: str):
    """
    Supported UDP packet formats:

    wrist_world,timestamp,chirality,x,y,z,qx,qy,qz,qw
    head_world,timestamp,x,y,z,qx,qy,qz,qw
    gesture,timestamp,chirality,pinch,snap_left,snap_right,snap_up,snap_down,double_tap
    """
    parts = text.strip().split(",")

    if len(parts) == 10 and parts[0] == "wrist_world":
        try:
            _, timestamp, chirality, x, y, z, qx, qy, qz, qw = parts
            return {
                "kind": "wrist_world",
                "timestamp": float(timestamp),
                "chirality": chirality.lower(),
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "qx": float(qx),
                "qy": float(qy),
                "qz": float(qz),
                "qw": float(qw),
            }
        except ValueError:
            return None

    if len(parts) == 9 and parts[0] == "head_world":
        try:
            _, timestamp, x, y, z, qx, qy, qz, qw = parts
            return {
                "kind": "head_world",
                "timestamp": float(timestamp),
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "qx": float(qx),
                "qy": float(qy),
                "qz": float(qz),
                "qw": float(qw),
            }
        except ValueError:
            return None

    if len(parts) == 9 and parts[0] == "gesture":
        try:
            _, timestamp, chirality, pinch, snap_left, snap_right, snap_up, snap_down, double_tap = parts
            return {
                "kind": "gesture",
                "timestamp": float(timestamp),
                "chirality": chirality.lower(),
                "data": [
                    int(pinch),
                    int(snap_left),
                    int(snap_right),
                    int(snap_up),
                    int(snap_down),
                    int(double_tap),
                ],
            }
        except ValueError:
            return None

    return None


def normalized_quaternion(x: float, y: float, z: float, w: float):
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        return 0.0, 0.0, 0.0, 1.0
    return x / norm, y / norm, z / norm, w / norm


@dataclass
class PoseState:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0
    valid: bool = False


@dataclass
class GestureState:
    data: list[int] = field(default_factory=lambda: [0, 0, 0, 0, 0, 0])
    snap_up_publish_count: int = 0


class VisionProTrackerPublisher(Node):
    def __init__(self):
        super().__init__("visionpro_tracker_publisher")

        self.declare_parameter("udp_ip", "0.0.0.0")
        self.declare_parameter("udp_port", 5005)
        self.declare_parameter("frame_id", "avp_world")

        udp_ip = self.get_parameter("udp_ip").value
        udp_port = self.get_parameter("udp_port").value
        self.frame_id = self.get_parameter("frame_id").value

        tracker_pose_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.tracker_pose_pub = self.create_publisher(PoseArray, "tracker_pose", tracker_pose_qos)
        self.lhand_gesture_pub = self.create_publisher(Int32MultiArray, "lhand_gesture", 10)
        self.rhand_gesture_pub = self.create_publisher(Int32MultiArray, "rhand_gesture", 10)

        self.left_pose = PoseState()
        self.right_pose = PoseState()
        self.head_pose = PoseState()

        self.left_gesture = GestureState()
        self.right_gesture = GestureState()

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((udp_ip, udp_port))
        self.sock.setblocking(False)

        self.recv_timer = self.create_timer(0.005, self.poll_socket)
        self.tracker_pose_timer = self.create_timer(1.0 / 60.0, self.publish_tracker_pose)

        self.get_logger().info(f"Node started: {self.get_name()}")
        self.get_logger().info(f"Listening on UDP {udp_ip}:{udp_port}")
        self.get_logger().info("Publishing:")
        self.get_logger().info("  tracker_pose [geometry_msgs/msg/PoseArray] at 60 Hz, best_effort, depth=1")
        self.get_logger().info("  lhand_gesture [std_msgs/msg/Int32MultiArray]")
        self.get_logger().info("  rhand_gesture [std_msgs/msg/Int32MultiArray]")

    def poll_socket(self):
        got_any_packet = False

        while True:
            try:
                data, addr = self.sock.recvfrom(4096)
            except BlockingIOError:
                break
            except Exception as e:
                self.get_logger().error(f"Socket error: {e}")
                break

            got_any_packet = True

            try:
                text = data.decode("utf-8").strip()
            except UnicodeDecodeError:
                self.get_logger().warn(
                    "Received non-UTF8 packet; skipped",
                    throttle_duration_sec=1.0,
                )
                continue

            parsed = parse_packet(text)
            if parsed is None:
                self.get_logger().warn(
                    f"Malformed packet from {addr}: {text}",
                    throttle_duration_sec=1.0,
                )
                continue

            kind = parsed["kind"]

            if kind == "wrist_world":
                state = self.left_pose if parsed["chirality"] == "left" else self.right_pose
                qx, qy, qz, qw = normalized_quaternion(
                    parsed["qx"], parsed["qy"], parsed["qz"], parsed["qw"]
                )
                state.x = parsed["x"]
                state.y = parsed["y"]
                state.z = parsed["z"]
                state.qx = qx
                state.qy = qy
                state.qz = qz
                state.qw = qw
                state.valid = True

                # self.get_logger().info(
                #     f"[RECV wrist_world] {parsed['chirality']} "
                #     f"pos=({state.x:.3f}, {state.y:.3f}, {state.z:.3f}) "
                #     f"quat=({state.qx:.3f}, {state.qy:.3f}, {state.qz:.3f}, {state.qw:.3f})",
                #     throttle_duration_sec=0.2,
                # )

            elif kind == "head_world":
                qx, qy, qz, qw = normalized_quaternion(
                    parsed["qx"], parsed["qy"], parsed["qz"], parsed["qw"]
                )
                self.head_pose.x = parsed["x"]
                self.head_pose.y = parsed["y"]
                self.head_pose.z = parsed["z"]
                self.head_pose.qx = qx
                self.head_pose.qy = qy
                self.head_pose.qz = qz
                self.head_pose.qw = qw
                self.head_pose.valid = True

                # self.get_logger().info(
                #     f"[RECV head_world] "
                #     f"pos=({self.head_pose.x:.3f}, {self.head_pose.y:.3f}, {self.head_pose.z:.3f}) "
                #     f"quat=({self.head_pose.qx:.3f}, {self.head_pose.qy:.3f}, "
                #     f"{self.head_pose.qz:.3f}, {self.head_pose.qw:.3f})",
                #     throttle_duration_sec=0.2,
                # )

            elif kind == "gesture":
                if parsed["chirality"] == "left":
                    self.left_gesture.data = parsed["data"]
                    self.publish_left_gesture()
                else:
                    self.right_gesture.data = parsed["data"]
                    self.publish_right_gesture()

                # self.get_logger().info(
                #     f"[RECV gesture] {parsed['chirality']} data={parsed['data']}",
                #     throttle_duration_sec=0.2,
                # )

        # if got_any_packet:
        #     self.get_logger().info(
        #         "[poll_socket] processed UDP packets",
        #         throttle_duration_sec=0.5,
        #     )

    def publish_left_gesture(self):
        msg = Int32MultiArray()
        msg.data = self.limited_snap_up_gesture(self.left_gesture)
        self.lhand_gesture_pub.publish(msg)
        # self.get_logger().info(
        #     f"[PUB lhand_gesture] data={msg.data}",
        #     throttle_duration_sec=0.2,
        # )

    def publish_right_gesture(self):
        msg = Int32MultiArray()
        msg.data = self.limited_snap_up_gesture(self.right_gesture)
        self.rhand_gesture_pub.publish(msg)
        # self.get_logger().info(
        #     f"[PUB rhand_gesture] data={msg.data}",
        #     throttle_duration_sec=0.2,
        # )

    @staticmethod
    def limited_snap_up_gesture(state: GestureState) -> list[int]:
        data = list(state.data)
        if len(data) <= 3:
            return data

        if data[3] == 0:
            state.snap_up_publish_count = 0
            return data

        if state.snap_up_publish_count >= 10:
            data[3] = 0
            return data

        state.snap_up_publish_count += 1
        return data

    def publish_tracker_pose(self):
        if not (self.left_pose.valid and self.right_pose.valid):
            # self.get_logger().info(
            #     "[PUB tracker_pose] skipped: waiting for left/right hand poses",
            #     throttle_duration_sec=1.0,
            # )
            return

        head_pose = self.head_pose if self.head_pose.valid else PoseState(valid=True)

        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.poses = [
            self.to_pose_msg(self.left_pose),
            self.to_pose_msg(self.right_pose),
            self.to_pose_msg(head_pose),
        ]

        self.tracker_pose_pub.publish(msg)
        # self.get_logger().info(
        #     "[PUB tracker_pose] "
        #     f"frame_id={msg.header.frame_id} | "
        #     f"L=({self.left_pose.x:.3f}, {self.left_pose.y:.3f}, {self.left_pose.z:.3f}) "
        #     f"R=({self.right_pose.x:.3f}, {self.right_pose.y:.3f}, {self.right_pose.z:.3f}) "
        #     f"H=({head_pose.x:.3f}, {head_pose.y:.3f}, {head_pose.z:.3f})",
        #     throttle_duration_sec=0.2,
        # )

    @staticmethod
    def to_pose_msg(state: PoseState) -> Pose:
        pose = Pose()
        pose.position.x = state.x
        pose.position.y = state.y
        pose.position.z = state.z
        pose.orientation.x = state.qx
        pose.orientation.y = state.qy
        pose.orientation.z = state.qz
        pose.orientation.w = state.qw
        return pose

    def destroy_node(self):
        try:
            self.sock.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VisionProTrackerPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
