import os
import math
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

N_SECTORS = 16
MAX_RANGE = 3.5
HOME = os.path.expanduser('~')
# 同じ玄関ドアを、異なる角度・位置から通った3本のデモをまとめて学習に使う
DEMO_NAMES = ['house_demo4', 'house_demo5', 'house_demo6']
# 実際に目指す目的地は、最初のデモ（house_demo4）の到達点とする
GOAL_DEMO_NAME = 'house_demo4'
# auto_collect.pyで自動収集したデータ（/reset_simulationで何度もリセットしながら
# 探索走行した大量の状態・行動ペア。ポスト回避の多様な場面を含む）
AUTO_DATA_PATH = os.path.join(HOME, 'auto_demo_data.npz')


def scan_to_features(ranges):
    arr = np.array(ranges, dtype=float)
    arr[~np.isfinite(arr)] = MAX_RANGE
    arr = np.clip(arr, 0, MAX_RANGE)
    sectors = np.array_split(arr, N_SECTORS)
    return np.array([s.min() if len(s) else MAX_RANGE for s in sectors])


def yaw_from_quat(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def goal_relative_state(x, y, yaw, goal_x, goal_y):
    dx, dy = goal_x - x, goal_y - y
    distance = math.hypot(dx, dy)
    bearing = math.atan2(dy, dx) - yaw
    bearing = math.atan2(math.sin(bearing), math.cos(bearing))  # [-pi, pi]に正規化
    return distance, bearing


def read_bag_raw(path):
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=path, storage_id='sqlite3'),
        ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr'),
    )

    scans, cmds, odoms = [], [], []
    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic == '/scan':
            msg = deserialize_message(data, LaserScan)
            scans.append((t, scan_to_features(msg.ranges)))
        elif topic == '/cmd_vel':
            msg = deserialize_message(data, Twist)
            cmds.append((t, msg.linear.x, msg.angular.z))
        elif topic == '/odom':
            msg = deserialize_message(data, Odometry)
            p = msg.pose.pose.position
            yaw = yaw_from_quat(msg.pose.pose.orientation)
            odoms.append((t, p.x, p.y, yaw))

    scans.sort(key=lambda s: s[0])
    odoms.sort(key=lambda o: o[0])
    return scans, cmds, odoms


def read_bag_pairs(path, goal_x, goal_y):
    """指定した固定の目的地（goal_x, goal_y）を基準に、(状態, 速度, 旋回)のペアを作る"""
    scans, cmds, odoms = read_bag_raw(path)
    scan_times = [s[0] for s in scans]
    odom_times = [o[0] for o in odoms]

    def nearest(times, t):
        return min(range(len(times)), key=lambda i: abs(times[i] - t))

    pairs = []
    for t, lin, ang in cmds:
        if abs(lin) < 0.05 and abs(ang) < 0.05:
            continue  # 「止まる」指令は学習データに使わない
        scan_feat = scans[nearest(scan_times, t)][1]
        _, x, y, yaw = odoms[nearest(odom_times, t)]
        distance, bearing = goal_relative_state(x, y, yaw, goal_x, goal_y)
        state = np.concatenate([[distance, bearing], scan_feat])
        pairs.append((state, lin, ang))

    return pairs


def read_goal(path):
    _, _, odoms = read_bag_raw(path)
    return odoms[-1][1], odoms[-1][2]


def load_dataset():
    """人間のデモ＋自動収集データをまとめた、1-NN用の生の記憶（平均化しない）を作る"""
    goal_x, goal_y = read_goal(os.path.join(HOME, GOAL_DEMO_NAME))

    all_pairs = []
    for name in DEMO_NAMES:
        all_pairs.extend(read_bag_pairs(os.path.join(HOME, name), goal_x, goal_y))

    X = [p[0] for p in all_pairs]
    y = [[p[1], p[2]] for p in all_pairs]
    n_demo = len(X)

    n_auto = 0
    if os.path.exists(AUTO_DATA_PATH):
        data = np.load(AUTO_DATA_PATH)
        X.extend(list(data['X']))
        y.extend(list(data['y']))
        n_auto = len(data['X'])

    return np.array(X), np.array(y), (goal_x, goal_y), n_demo, n_auto


class ReactiveNNController(Node):

    def __init__(self):
        super().__init__('reactive_nn_controller')

        self.X, self.y, (self.goal_x, self.goal_y), n_demo, n_auto = load_dataset()

        # LIDARは16次元、目的地情報（距離・方位）は2次元しかないため、
        # そのままの距離計算だとLIDARの見た目の近さに支配され、肝心の
        # 方位（どちらに避けるべきか）が埋もれてしまう。各特徴量を
        # 標準化（平均0・分散1）してスケールを揃えてから最近傍を探す。
        self.feat_mean = self.X.mean(axis=0)
        self.feat_std = self.X.std(axis=0)
        self.feat_std[self.feat_std < 1e-6] = 1.0
        self.X_norm = (self.X - self.feat_mean) / self.feat_std

        self.get_logger().info(
            f'学習データ数: {len(self.X)}件（人のデモ{n_demo}件＋自動収集{n_auto}件） / '
            f'目的地=({self.goal_x:.2f}, {self.goal_y:.2f})')

        self.pose = None  # (x, y, yaw)
        self.prev_ang = 0.0  # 旋回のふらつきを抑える平滑化用
        # 本体姿勢（転倒監視用）
        self.roll = 0.0
        self.pitch = 0.0
        self.tip_logged = False
        # 二段階安全装置（真っ直ぐ後退 -> その場旋回）の状態管理
        self.device_phase = None   # None / 'retreat' / 'turn'
        self.device_deadline = 0.0
        self.device_dir = 0.0      # 開いている側へ旋回する符号
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)

    def on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q)
        self.pose = (p.x, p.y, yaw)
        # 衝突で大きく傾いた（転倒した）かどうかの監視用。
        # 転倒状態で判断を続けると危険な動きをするため、停止させる
        self.roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
        self.pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))

    def on_scan(self, msg):
        if self.pose is None:
            return

        # 転倒検知：大きく傾いた状態での判断をやめて安全に停止する
        if abs(self.roll) > 0.4 or abs(self.pitch) > 0.4:
            if not self.tip_logged:
                self.tip_logged = True
                self.get_logger().info('転倒を検知 -> 安全のため停止します（Gazeboのリセットが必要です）')
            self.pub.publish(Twist())
            return
        # 姿勢が正立に戻ったら再び転倒を検知できるようにしておく
        # （1回目の記録でフラグが立ったままになると、2回目の転倒が
        #  ログに出ず原因究明を妨げるため）
        if self.tip_logged and abs(self.roll) < 0.2 and abs(self.pitch) < 0.2:
            self.tip_logged = False

        x, y, yaw = self.pose
        distance, bearing = goal_relative_state(x, y, yaw, self.goal_x, self.goal_y)

        if distance < 0.6:
            self.pub.publish(Twist())
            self.get_logger().info('目的地に到着 -> 停止')
            return

        scan_feat = scan_to_features(msg.ranges)
        state = np.concatenate([[distance, bearing], scan_feat])

        # 1-NN：平均化せず、今の状況に一番近い記憶をそのまま真似る
        # （複数の経路を回帰で平均すると、どちらの経路でもない危険な
        # 中間の行動になる「平均化問題」を避けるため）
        state_norm = (state - self.feat_mean) / self.feat_std
        dists = np.linalg.norm(self.X_norm - state_norm, axis=1)
        idx = int(np.argmin(dists))
        lin, ang = self.y[idx]
        nn_dist = dists[idx]

        # 最近傍距離が遠い（＝似た記憶が無い、自信が無い）場合は、
        # 学習結果を使わず目的地方向への比例制御にフォールバックする。
        # 目的地付近の変わった角度は学習データがほとんど無く、
        # その場で無意味に回転し続けてしまっていたため。
        if nn_dist > 3.0:
            ang = max(-1.2, min(1.2, 1.5 * bearing))
            lin = 0.4 * max(0.0, math.cos(bearing))
            fallback_note = '（学習データ不足のためフォールバック）'
        else:
            fallback_note = ''

        # 隣接する記憶同士が入れ替わって旋回が急に反転するのを抑えるため、
        # 前回の旋回指令との平滑化（低域通過フィルタ）をかける
        ang = 0.5 * self.prev_ang + 0.5 * ang
        self.prev_ang = ang

        ranges = np.array(msg.ranges, dtype=float)
        ranges[~np.isfinite(ranges)] = MAX_RANGE
        # 正面だけでなく全方向の最短距離を見る（斜めからの衝突にも対応するため）
        overall_min = float(ranges.min()) if len(ranges) else MAX_RANGE

        # --- 近接速度制限（転倒防止の物理対策） ---
        # 0.3m/s以上の速さで0.12mのポストに接触すると転倒した。制御周期は
        # 0.2秒しかないので、近距離では減速が間に合わない。長い距離手前の
        # 0.45mから既に遅くしておけば、接触時の運動量が小さく転倒しない。
        # 窓を±30度にするのは、走行帯の左壁(0.23m)が±30度以上の斜め方向で
        # 0.23/sin(30度)≈0.46mと見え、0.45未満にはならないため（誤発動防止）
        n = len(ranges)
        angles = np.arange(n) * float(msg.angle_increment) + msg.angle_min
        front_mask = (angles[:n] < math.pi / 6) | (angles[:n] > 2 * math.pi - math.pi / 6)
        front_min = float(ranges[front_mask].min()) if front_mask.any() else MAX_RANGE
        governor_note = ''
        if front_min < 0.45 and lin > 0.15:
            lin = 0.15
            governor_note = '（前方{0:.2f}mの接近のため速度制限）'.format(front_min)

        self.get_logger().info(
            f'残り距離={distance:.2f}m 方位={math.degrees(bearing):.0f}度 '
            f'-> linear={lin:.3f} angular={ang:.3f}（最近傍距離={nn_dist:.2f}）'
            f'{fallback_note}{governor_note}')

        # --- 二段階安全装置：真っ直ぐ後退してから、その場で開いている側へ旋回 ---
        # （後退と旋回を同時にやると弧を描いて障害物に斜めに当たり
        #  転倒する試み方だったので、物理に一度に混ぜない形にした）
        now = time.time()
        if self.device_phase == 'retreat':
            if now < self.device_deadline:
                lin, ang = -0.3, 0.0
            else:
                self.device_phase = 'turn'
                self.device_deadline = now + 0.7
                lin, ang = 0.0, 0.8 * self.device_dir
                self.get_logger().info('後退完了 -> その場で旋回')
            self.prev_ang = ang
        elif self.device_phase == 'turn':
            if now < self.device_deadline:
                lin, ang = 0.0, 0.8 * self.device_dir
            else:
                self.device_phase = None
                lin, ang = 0.0, 0.0
                self.get_logger().info('旋回完了 -> 通常運転に復帰')
            self.prev_ang = ang
        elif overall_min < 0.15:
            # ハード安全装置：学習結果に関わらず、接触寸前なら後退→向き替え
            # （学習とは独立した、最終手段としての保険）
            left_min = float(ranges[: n // 2].min()) if n else MAX_RANGE
            right_min = float(ranges[n // 2 :].min()) if n else MAX_RANGE
            self.device_dir = 1.0 if left_min > right_min else -1.0
            self.device_phase = 'retreat'
            self.device_deadline = now + 0.5
            lin, ang = -0.3, 0.0
            self.prev_ang = ang
            self.get_logger().info(f'安全装置作動（最短={overall_min:.2f}m）-> 真っ直ぐ後退')

        out = Twist()
        out.linear.x = float(lin)
        out.angular.z = float(ang)
        self.pub.publish(out)


def main():
    rclpy.init()
    node = ReactiveNNController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Ctrl+Cで終了しても最後の速度指令が残って動き続けないよう、
        # 必ず停止指令を送ってから終了する
        node.pub.publish(Twist())
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
