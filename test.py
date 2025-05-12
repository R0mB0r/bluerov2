import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

class BlueROVThrusterCommander(Node):
    def __init__(self):
        super().__init__('bluerov_thruster_commander')
        
        # Liste des thrusters du BlueROV2
        self.thruster_topics = [
            '/bluerov2/cmd_thruster1',
            '/bluerov2/cmd_thruster2',
            '/bluerov2/cmd_thruster3',
            '/bluerov2/cmd_thruster4',
            '/bluerov2/cmd_thruster5',
            '/bluerov2/cmd_thruster6'
        ]
        
        # Création des publishers pour chaque thruster
        self.thruster_publishers = {topic: self.create_publisher(Float64, topic, 10) for topic in self.thruster_topics}
        
        # Timer pour envoyer périodiquement les commandes moteurs
        self.timer = self.create_timer(1.0, self.send_thruster_commands)

        self.get_logger().info("BlueROV Thruster Commander Node Started")

    def send_thruster_commands(self):
        """
        Envoie des commandes de poussée aux thrusters.
        Ces valeurs peuvent être ajustées pour des tests ou basées sur une logique de contrôle.
        """
        thrust_values = [2., 2., 2., 2., 0., 0.]  # Valeurs de test pour les thrusters

        for i, topic in enumerate(self.thruster_topics):
            msg = Float64()
            msg.data = thrust_values[i]  # Attribution de la valeur de poussée
            self.thruster_publishers[topic].publish(msg)
            self.get_logger().info(f"Sent thrust {msg.data} to {topic}")

def main(args=None):
    rclpy.init(args=args)
    node = BlueROVThrusterCommander()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

