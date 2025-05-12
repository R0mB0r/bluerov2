import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import Header

class PosePublisher(Node):
    def __init__(self):
        super().__init__('pose_publisher')
        self.publisher = self.create_publisher(Pose, '/bluerov2/cmd_pose', 10)
        self.timer = self.create_timer(1.0, self.publish_pose)

    def publish_pose(self):
        msg = Pose()
        msg.position.x = 1.0  # Position X
        msg.position.y = 2.0  # Position Y
        msg.position.z = 0.5  # Position Z
        msg.orientation.x = 0.0  # Orientation X
        msg.orientation.y = 0.0  # Orientation Y
        msg.orientation.z = 0.0  # Orientation Z
        msg.orientation.w = 1.0  # Orientation W (quaternion)
        
        # Publier le message
        self.publisher.publish(msg)
        self.get_logger().info('Pose published')

def main(args=None):
    rclpy.init(args=args)
    node = PosePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()