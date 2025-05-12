import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
import numpy as np
from geometry_msgs.msg import Pose
from transforms3d.euler import quat2euler
import numpy as np
import matplotlib.pyplot as plt  # 📌 Import pour affichage des erreurs

def sawtooth(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def rotation_matrix(phi, theta, psi):
    Rx = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    Rz = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def angular_transformation_matrix(phi, theta):
    if np.abs(np.cos(theta)) < 1e-6:
        raise ValueError("Singularité : cos(theta) ≈ 0, inversion impossible.")
    return np.array([
        [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]
    ])

def full_transformation_matrix(phi, theta, psi):
    R = rotation_matrix(phi, theta, psi)
    T = angular_transformation_matrix(phi, theta)
    return np.block([[R, np.zeros((3, 3))], [np.zeros((3, 3)), T]])

class VectorPIDController:
    def __init__(self, kp, ki, kd):
        self.kp = np.array(kp)
        self.ki = np.array(ki)
        self.kd = np.array(kd)
        self.prev_error = np.zeros(6)
        self.integral = np.zeros(6)

    def compute(self, setpoint, measured, dt):
        error = np.array(setpoint) - np.array(measured)
        error[3:] = np.vectorize(sawtooth)(error[3:])
        error_b = np.linalg.inv(full_transformation_matrix(measured[3], measured[4], measured[5])) @ error
        self.integral += error_b * dt
        derivative = (error_b - self.prev_error) / dt if dt > 0 else np.zeros(6)
        derivative[3:] = np.vectorize(sawtooth)(derivative[3:])
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

class BlueROVPIDController(Node):
    def __init__(self, target=None,
                 kp=[30,30, 30, 30, 30, 30],
                 ki=[0., 0., 0., 0., 0., 0.],
                 kd=[0., 0., 0., 0., 0., 0.]): 
        super().__init__('bluerov_pid_controller')

        self.target = np.array(target) if target is not None else np.zeros(6)
        #self.get_logger().info(f"Target position: {self.target}")
        #self.get_logger().info(f"Kp: {kp}, Ki: {ki}, Kd: {kd}")

        self.pid = VectorPIDController(kp=kp, ki=ki, kd=kd)

        self.H_pinv = np.linalg.pinv(np.array([
            [-0.707, -0.707, -0.707, -0.707, 0, 0],
            [-0.707, 0.707, 0.707, -0.707, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0.115, -0.115],
            [0, 0, 0, 0, 0, 0],
            [-0.177, 0.177, -0.177, 0.177, 0, 0]
        ]))

        self.subscription_pose_gt = self.create_subscription(Pose, '/bluerov2/pose_gt', self.pose_gt_callback, 10)
        self.thruster_publishers = {
            f'/bluerov2/cmd_thruster{i}': self.create_publisher(Float64, f'/bluerov2/cmd_thruster{i}', 10)
            for i in range(1, 7)
        }

        self.prev_time = self.get_clock().now()
        self.timer = self.create_timer(0.1, self.control_loop)

        self.errors_pos = []
        self.errors_ang = []
        self.time_data = []
        self.start_time = None

        #self.get_logger().info("BlueROV PID Controller Node Started")

    
    def pose_gt_callback(self, msg):
        phi, theta, psi = quat2euler([msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z])
        self.current = np.array([msg.position.x, msg.position.y, msg.position.z, phi, theta, psi])

    def control_loop(self):
        current_time = self.get_clock().now()
        dt = (current_time - self.prev_time).nanoseconds * 1e-9
        self.prev_time = current_time

        if hasattr(self, 'current'):
            forces = self.pid.compute(self.target, self.current, dt)
            thruster_cmds = -self.H_pinv @ forces
            thruster_cmds = np.clip(thruster_cmds, -30, 30)  # Limitation des commandes entre -1 et 1

            for i, topic in enumerate(self.thruster_publishers.keys()):
                msg = Float64()
                msg.data = thruster_cmds[i]
                self.thruster_publishers[topic].publish(msg)

            #self.get_logger().info(f"Thrust Commands: {', '.join(f'{cmd:.2f}' for cmd in thruster_cmds)}")

            # 📌 Stockage des erreurs
            if self.start_time is None:
                self.start_time = current_time
            time_elapsed = (current_time - self.start_time).nanoseconds * 1e-9

            #print(f"Current Position: {self.current}")

            error = self.target - self.current
            error[3:] = np.vectorize(sawtooth)(error[3:])
            self.errors_pos.append(error[:3])
            self.errors_ang.append(error[3:])
            self.time_data.append(time_elapsed)

    def plot_errors(self):
        """ 📌 Affiche les erreurs après exécution """
        self.errors_pos = np.array(self.errors_pos)
        self.errors_ang = np.array(self.errors_ang)

        fig, axs = plt.subplots(2, 1, figsize=(8, 6))

        axs[0].plot(self.time_data, self.errors_pos[:, 0], label="X")
        axs[0].plot(self.time_data, self.errors_pos[:, 1], label="Y")
        axs[0].plot(self.time_data, self.errors_pos[:, 2], label="Z")
        axs[0].set_title("Erreurs de position (m)")
        axs[0].legend()
        axs[0].grid()

        axs[1].plot(self.time_data, self.errors_ang[:, 0], label="Roll")
        axs[1].plot(self.time_data, self.errors_ang[:, 1], label="Pitch")
        axs[1].plot(self.time_data, self.errors_ang[:, 2], label="Yaw")
        axs[1].set_title("Erreurs d'orientation (rad)")
        axs[1].legend()
        axs[1].grid()

        plt.tight_layout()
        plt.show()

def main(args=None):
    rclpy.init(args=args)

    custom_target = [10, -1.0, 0.0, 0.0, 0.0, 0.0]
    kp = [20, 20, 20, 20, 20, 20]
    ki = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    kd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    node = BlueROVPIDController(target=custom_target)
    #node = BlueROVPIDController(kp=kp)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.plot_errors()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()






"""
def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        #Utilise un PID pour ramener le robot à la position d'origine.
        self.node.get_logger().info("🔄 Reset PID en cours...")

        target = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        kp = np.array([20, 20, 20, 10, 10, 10])
        node = BlueROVPIDController(target=target, kp=kp)

        threshold_pos = 0.05
        threshold_ang = 0.05
        dt = 0.01

        while True:
            rclpy.spin_once(node)

            if hasattr(node, 'current'):
                pos_error = np.linalg.norm(node.current[:3] - target[:3])
                ang_error = np.linalg.norm(node.current[3:] - target[3:])

                if pos_error < threshold_pos and ang_error < threshold_ang:
                    break

            time.sleep(dt)

        self.current_step = 0
        robot_position = np.array([node.current[0], node.current[1], node.current[2], node.current[3], node.current[4], node.current[5]])

        self.node.get_logger().info("🔄 Reset PID terminé.")
        self.node.get_logger().info(f"position actuelle : {robot_position}")
        self.dist_initial = np.linalg.norm(robot_position[:3] - self.goal_position[:3])
        node.destroy_node()
        return np.concatenate([robot_position, self.goal_position]), {}
"""
