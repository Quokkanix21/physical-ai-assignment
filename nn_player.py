import sys
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


def read_bag(path):
    reader = SequentialReader()
    storage_options = StorageOptions(uri=path, storage_id='sqlite3')
    converter_options = ConverterOptions(
        input_serialization_format='cdr', output_serialization_format='cdr')
    reader.open(storage_options, converter_options)

    cmds = []
    odoms = []
    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic == '/demo/cmd_demo':
            msg = deserialize_message(data, Twist)
            cmds.append((t, msg.linear.x, msg.angular.z))
        elif topic == '/demo/odom_demo':
            msg = deserialize_message(data, Odometry)
            odoms.append((t, msg.pose.pose.position.x, msg.pose.pose.position.y))

    t0 = cmds[0][0]
    cmd_seq = [((t - t0) / 1e9, lin, ang) for (t, lin, ang) in cmds]

    x0, y0 = odoms[0][1], odoms[0][2]
    x1, y1 = odoms[-1][1], odoms[-1][2]
    distance = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5

    return distance, cmd_seq


class NearestNeighborPlayer(Node):

    def __init__(self, demo_paths, target_distance):
        super().__init__('nearest_neighbor_player')
        self.pub = self.create_publisher(Twist, '/demo/cmd_demo', 10)

        demos = {name: read_bag(path) for name, path in demo_paths.items()}
        for name, (dist, _) in demos.items():
            self.get_logger().info(f'{name}: distance={dist:.3f}m')

        best_name = min(demos, key=lambda n: abs(demos[n][0] - target_distance))
        best_distance, best_seq = demos[best_name]
        self.get_logger().info(
            f'target={target_distance:.3f}m -> nearest neighbor = '
            f'{best_name} (distance={best_distance:.3f}m)'
        )

        self.replay(best_seq)

    def replay(self, seq):
        start = time.time()
        for dt, lin, ang in seq:
            wait = dt - (time.time() - start)
            if wait > 0:
                time.sleep(wait)
            msg = Twist()
            msg.linear.x = lin
            msg.angular.z = ang
            self.pub.publish(msg)
        self.pub.publish(Twist())


def main():
    rclpy.init()
    home = os.path.expanduser('~')
    demo_paths = {
        'demo1': os.path.join(home, 'demo1'),
        'demo2': os.path.join(home, 'demo2'),
        'demo3': os.path.join(home, 'demo3'),
    }
    target_distance = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    node = NearestNeighborPlayer(demo_paths, target_distance)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
