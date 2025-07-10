import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from geometry_msgs.msg import Pose, Vector3
from ros_gz_interfaces.srv import SetEntityPose
import numpy as np
import time
from transforms3d.euler import quat2euler
from transforms3d.euler import euler2quat
import sys
import signal

class BlueROVROSInterface:
    def __init__(self):
        
        # Initialize ROS 2 node
        rclpy.init()
        self.node = rclpy.create_node('bluerov_rl_agent')
        
        # Publishers
        self.thruster_topics = [
            '/bluerov2/cmd_thruster1',
            '/bluerov2/cmd_thruster2',
            '/bluerov2/cmd_thruster3',
            '/bluerov2/cmd_thruster4',
            '/bluerov2/cmd_thruster5',
            '/bluerov2/cmd_thruster6'
        ]
        self.thruster_publishers = {topic: self.node.create_publisher(Float64, topic, 10) for topic in self.thruster_topics}
        self.ocean_current_publisher = self.node.create_publisher(Vector3, '/current', 10)
        
        # Subscribers
        self.subscription = self.node.create_subscription(Pose, '/bluerov2/pose_gt', self.pose_callback, 10)
        
        # Service client
        self.client = self.node.create_client(SetEntityPose, '/world/ocean/set_pose')
        
        # Initialize robot position and pose update flag
        self.robot_position = np.zeros(6)
        self.pose_updated = False

        # Signal handler for graceful shutdown
        def shutdown_signal_handler():
            self.node.get_logger().info("Shutting down BlueROV ROS interface...")
            self.stop_all_thrusters()
            rclpy.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, shutdown_signal_handler)

    def pose_callback(self, msg):
        
        phi, theta, psi = quat2euler([msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z])
        self.robot_position = np.array([
            msg.position.x, msg.position.y, msg.position.z,
            phi, theta, psi
        ])
        
        ## Add random noise to the robot's position
        #sensor_noise = self.np_random.uniform(0.05, 0.1, size=6)
        #self.robot_position += sensor_noise

        self.pose_updated = True

    def publish_thrusters(self, actions):
        
        ## Add random noise to the action
        # action_noise = self.np_random.uniform(0.01, 0.05, size=6)
        # action += action_noise
        
        scaled_action = np.clip(actions, -1.0, 1.0) * 20.0
        for i, topic in enumerate(self.thruster_topics):
            msg = Float64()
            msg.data = float(scaled_action[i])
            self.thruster_publishers[topic].publish(msg)

    def set_pose(self, pose):
        x, y, z = pose[:3]
        yaw = pose[5]
        req = SetEntityPose.Request()
        req.entity.name = 'bluerov2'
        req.entity.type = 2  # Type MODEL
        req.pose.position.x = x
        req.pose.position.y = y
        req.pose.position.z = z
        quat = euler2quat(0.0, 0.0, yaw)
        req.pose.orientation.x = quat[1]
        req.pose.orientation.y = quat[2]
        req.pose.orientation.z = quat[3]
        req.pose.orientation.w = quat[0]
        future = self.client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        if future.result() is not None and future.result().success:
            self.node.get_logger().info("Position successfully teleported.")
        else:
            self.node.get_logger().warn("⚠️ Failed to reset position.")

        # Wait for the robot's position to be updated
        while np.linalg.norm(self.robot_position[:3] - pose[:3]) > 0.05:
            rclpy.spin_once(self.node, timeout_sec=0.1)


    def wait_for_pose_update(self, timeout=5.0):
        while not self.pose_updated:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        self.pose_updated = False

    def stop_all_thrusters(self):
        for topic in self.thruster_topics:
            msg = Float64()
            msg.data = 0.0
            self.thruster_publishers[topic].publish(msg)

    def close(self):
        self.stop_all_thrusters()
        rclpy.shutdown()
        self.node.destroy_node()
        self.node.get_logger().info("ROS 2 node destroyed.")