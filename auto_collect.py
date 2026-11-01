import os
import math
import random
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

from reactive_nn import (
    scan_to_features, goal_relative_state, yaw_from_quat,
    read_goal, HOME, GOAL_DEMO_NAME, MAX_RANGE,
)

N_EPISODES = 40
EPISODE_SECONDS = 30.0
OUT_PATH = os.path.join(HOME, 'auto_demo_data.npz')

# LIDARは360点でangle_min=0(正面)から反時計回りに並ぶ。
# 「正面・後方・左・右」の各窓で最短距離を取るための角度領域(ラジアン)。
FRONT_HALF = 0.349    # ±20度(±45度にすると左壁の端が正面窓に紛れ込むため狭くする)
FRONT_LO = 2.0 * math.pi - FRONT_HALF
REAR_LO, REAR_HI = 2.356, 3.927      # 135〜225度
LEFT_LO, LEFT_HI = 0.785, 2.356      # 45〜135度
RIGHT_LO, RIGHT_HI = 3.927, 5.498    # 225〜315度


def beam_min(ranges, angles, lo, hi):
    m = (angles >= lo) & (angles < hi)
    return float(ranges[m].min()) if m.any() else MAX_RANGE


class AutoCollector(Node):

    def __init__(self, goal_x, goal_y):
        super().__init__('auto_collector')
        self.goal_x, self.goal_y = goal_x, goal_y
        self.pose = None
        self.roll = 0.0
        self.pitch = 0.0
        self.latest_scan = None
        self.states = []
        self.actions = []
        self.avoid_dir = None
        self.avoid_until = 0.0
        self.excursion_offset = None
        self.excursion_until = 0.0
        self.excursion_trigger_dist = 0.0
        self.excursion_side_lock = None  # そのエピソードで開いている側(壁の反対側)
        self.recent = []   # 直近の状態・行動(転倒時の原因分析用)
        self.counts = {'avoid': 0, 'excursion': 0, 'base': 0, 'excursion_pending': 0}

        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.reset_cli = self.create_client(Empty, '/reset_simulation')

    def on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q)
        self.roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
        self.pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
        self.pose = (p.x, p.y, yaw)

    def on_scan(self, msg):
        self.latest_scan = msg

    def is_tipped_over(self):
        return abs(self.roll) > 0.4 or abs(self.pitch) > 0.4

    def reset(self):
        if self.reset_cli.wait_for_service(timeout_sec=2.0):
            future = self.reset_cli.call_async(Empty.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        self.pose = None
        self.latest_scan = None
        self.avoid_dir = None
        self.avoid_until = 0.0
        self.excursion_until = 0.0
        self.excursion_side_lock = None
        time.sleep(0.5)

    def plan_episode(self):
        """エピソードごとに1回だけ側方・後方シフトの場所と方向を先に仕込む。
        ランダムスピン(旧実装)は符号が毎回入れ替わって正味ゼロになり、
        前進しない+でたらめ行動を記録するだけなのでやめた。"""
        self.excursion_trigger_dist = 99.0 if random.random() < 0.3 else random.uniform(0.8, 2.4)
        if random.random() < 0.35:
            self.excursion_offset = math.pi + random.uniform(-0.25, 0.25)
        else:
            side = 1.0 if random.random() < 0.5 else -1.0
            self.excursion_offset = side * random.uniform(1.0, 2.6)

    def explore_policy(self, distance, bearing, ranges, angles, now):
        front_min = min(
            beam_min(ranges, angles, 0.0, FRONT_HALF),
            beam_min(ranges, angles, FRONT_LO, 2.0 * math.pi + 0.1))
        rear_min = beam_min(ranges, angles, REAR_LO, REAR_HI)
        left_min = beam_min(ranges, angles, LEFT_LO, LEFT_HI)
        right_min = beam_min(ranges, angles, RIGHT_LO, RIGHT_HI)

        # --- 回避モード: 正面(±20度)に障害物が近いときだけ入る ---
        # (側方の壁0.23mやその壁端0.35mでは入らない。旧実装は全方向最短かつ
        #  広い正面窓で判定していたため、走行帯の側壁だけで常時回避・ゆっくり
        #  速度になっていた)
        if front_min < 0.3 or rear_min < 0.25:
            self.excursion_offset = None
            self.counts['avoid'] += 1
            # 方向の更新は1.5秒に1回(ヒステリシス)。まひふらつきを防止
            if self.avoid_dir is None or now > self.avoid_until:
                self.avoid_dir = 1.0 if left_min > right_min else -1.0
                self.avoid_until = now + 1.5
            # 障害物が近い方向と反対方向へ移動する(後ろから押し込めない)
            if front_min < rear_min:
                lin = -0.3 if front_min < 0.22 else 0.15   # 前方有障害: 後退or低速
            else:
                lin = 0.3                                   # 後方に障害: 前進で離れる
            ang = 0.8 * self.avoid_dir
            mode = 'avoid'
        # --- 計画的シフト起動判定(1エピソード1回)と実行 ---
        else:
            if (self.excursion_offset is not None
                    and distance < self.excursion_trigger_dist):
                self.excursion_trigger_dist = 0.0
                self.excursion_until = now + random.uniform(1.0, 2.5)
                self.counts['excursion_pending'] += 1
                # 側方シフトは「開いている側」へ向かせる(壁側に進めない)
                if abs(self.excursion_offset) < math.pi - 0.3:
                    if self.excursion_side_lock is None:
                        self.excursion_side_lock = 1.0 if left_min > right_min else -1.0
                    self.excursion_offset = (self.excursion_side_lock
                                             * abs(self.excursion_offset))
            if (self.excursion_offset is not None and now < self.excursion_until):
                # 「目的地の方向+オフセット」の方向へ向かって前進する。
                # 向くに比例制御し、向きが目標方向に向かえば速度も上げる
                err = math.atan2(
                    math.sin(bearing + self.excursion_offset),
                    math.cos(bearing + self.excursion_offset))
                ang = max(-1.2, min(1.2, 2.0 * err))
                lin = 0.3 * max(0.0, math.cos(err))
                self.counts['excursion'] += 1
                mode = 'excursion'
            else:
                self.excursion_offset = None  # シフト完了(または時間超過)
                # 揺らぎは±0.12と小さく(scanごとの乱数スピンを防ぐ)
                ang = max(-1.2, min(1.2, 1.5 * bearing)) + random.uniform(-0.12, 0.12)
                lin = 0.4 * max(0.0, math.cos(bearing))
                self.counts['base'] += 1
                mode = 'base'
        return lin, ang, mode, front_min, rear_min

    def run_episode(self):
        self.plan_episode()
        self.reset()
        start = time.time()
        self.recent = []
        self.counts = {'avoid': 0, 'excursion': 0, 'base': 0, 'excursion_pending': 0}
        while rclpy.ok() and (time.time() - start) < EPISODE_SECONDS:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.pose is None or self.latest_scan is None:
                continue

            if self.is_tipped_over():
                self.get_logger().info('  転倒を検知 -> このエピソードを打ち切り')
                self.get_logger().info('  直近の推移: ' + ' | '.join(
                    f"x={r['x']:.2f},y={r['y']:.2f},roll={r['roll']:.2f},pitch={r['pitch']:.2f}"
                    f",front={r['front']:.2f},rear={r['rear']:.2f}"
                    f",lin={r['lin']:.2f},ang={r['ang']:.2f},{r['mode']}"
                    for r in self.recent[-10:]))
                break

            x, y, yaw = self.pose
            distance, bearing = goal_relative_state(x, y, yaw, self.goal_x, self.goal_y)
            if distance < 0.3:
                break

            scan = self.latest_scan
            ranges = np.array(scan.ranges, dtype=float)
            ranges[~np.isfinite(ranges)] = MAX_RANGE
            angles = np.arange(len(ranges)) * float(scan.angle_increment) + float(scan.angle_min)
            scan_feat = scan_to_features(scan.ranges)

            lin, ang, mode, front_min, rear_min = self.explore_policy(
                distance, bearing, ranges, angles, time.time())

            state = np.concatenate([[distance, bearing], scan_feat])
            self.states.append(state)
            self.actions.append((lin, ang))

            self.recent.append({
                'x': x, 'y': y, 'roll': self.roll, 'pitch': self.pitch,
                'front': front_min, 'rear': rear_min,
                'lin': lin, 'ang': ang, 'mode': mode,
            })
            self.recent = self.recent[-12:]

            out = Twist()
            out.linear.x = lin
            out.angular.z = ang
            self.pub.publish(out)

        self.pub.publish(Twist())

    def save(self):
        X = np.array(self.states)
        y = np.array(self.actions)
        if self.prev_X is not None:
            X = np.concatenate([self.prev_X, X], axis=0)
            y = np.concatenate([self.prev_y, y], axis=0)
        np.savez(OUT_PATH, X=X, y=y)
        self.get_logger().info(f'  保存済み(合計{len(X)}件) / 今エピソード: {self.counts}')

    def collect(self, n_episodes):
        self.prev_X, self.prev_y = None, None
        if os.path.exists(OUT_PATH):
            prev = np.load(OUT_PATH)
            self.prev_X, self.prev_y = prev['X'], prev['y']

        for i in range(n_episodes):
            self.get_logger().info(f'エピソード {i + 1}/{n_episodes} 開始')
            self.run_episode()
            self.save()


def main():
    rclpy.init()
    goal_x, goal_y = read_goal(os.path.join(HOME, GOAL_DEMO_NAME))
    node = AutoCollector(goal_x, goal_y)
    node.collect(N_EPISODES)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
