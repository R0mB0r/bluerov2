import rclpy
from rclpy.node import Node
from ros_gz_interfaces.srv import SetEntityPose
import time

class BlueROVTeleporter(Node):
    def __init__(self):
        super().__init__('bluerov_teleporter')

        self.client = self.create_client(SetEntityPose, '/world/ocean/set_pose')
        while not self.client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('⏳ Attente du service /world/ocean/set_pose...')

        self.positions = [
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 2.0, 0.0),
            (0.0, 2.0, 0.0)
        ]

        self.teleport_sequence()

    def teleport_sequence(self):
        for i, (x, y, z) in enumerate(self.positions):
            req = SetEntityPose.Request()
            req.entity.name = 'bluerov2'
            req.entity.type = 2  # MODEL
            req.pose.position.x = x
            req.pose.position.y = y
            req.pose.position.z = z
            req.pose.orientation.x = 0.0
            req.pose.orientation.y = 0.0
            req.pose.orientation.z = 0.0
            req.pose.orientation.w = 1.0

            self.get_logger().info(f'🚀 Téléportation {i+1} à ({x}, {y}, {z})')
            future = self.client.call_async(req)
            rclpy.spin_until_future_complete(self, future)

            if future.result() and future.result().success:
                self.get_logger().info(f'✅ Téléportation {i+1} réussie !')
            else:
                self.get_logger().error(f'❌ Échec de la téléportation {i+1}.')

            time.sleep(2.0)

        self.get_logger().info("🏁 Téléportations terminées.")
        rclpy.shutdown()


def main():
    rclpy.init()
    BlueROVTeleporter()

if __name__ == '__main__':
    main()
