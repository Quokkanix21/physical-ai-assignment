# TurtleBot3 家の中への配送デモ（ROS2 + Gazebo）

## 概要

TurtleBot3（burger・実在するOSSロボット）が、`turtlebot3_house` ワールドで
**家の外の庭から玄関ドアを通って家の中に入り、壁・家具を避けながら目的地の部屋まで移動する**デモです。

ロボットの自律性は **模倣学習（Imitation Learning）** で実現しています。

1. 人間がGazebo上のロボットをキーボードで操作した軌跡を、LIDAR `/scan`・速度指令 `/cmd_vel`・位置 `/odom` とともにrosbag2で記録する（＝「お手本」）
2. さらに `auto_collect.py` が `/reset_simulation` サービスでシミュレーションをリセットしながら、**複数の開始角度・経路を持った走行データを自動収集する**（講座第2回・第5回で学んだ「データの多様性で汎化させる」考え方の実装）
3. `reactive_nn.py` が、その瞬間の「**LIDAR16方位の最短距離 ＋ 目的地までの相対距離・方位**」を状態として、記憶の中から**最も近い瞬間の操作をそのまま真似る**（1-最近傍法。複数経路を平均化すると安全でない中間動作になる「平均化問題」を避ける設計）
4. 起動時に学習データを読みこんで、閉ループ（センサ→判断→指令→再観測…）を回しながら走行する

安全のため、学習結果とは独立に **ハードな安全装置**（接触に近づいたら強制後退＋旋回）と、
**転倒検知**（本体が大きく傾いたら自動停止）も入れています。

## アーキテクチャ

```
[キーボードteleop] --お手本(rosbag2)--> house_demo1〜6
                                        |
                                        v
[reactive_nn.py] 起動時に house_demo4/5/6 + auto_demo_data.npz --> 1-NN記憶
        |
   毎スキャン: (LIDAR16方位 + 目的地相対距離/方位) -> 一番近い記憶の速度指令を /cmd_vel に発行
   障害物接近時: 最短距離 < 閾値 -> 強制的に後退＋旋回(学習と独立した安全装置)
   転倒検知時:   本体が大きく傾いたら自動停止
```

## ファイル一覧

| ファイル | 役割 |
|---|---|
| `reactive_nn.py` | 自律運転ノード。起動時にデモ＋自動収集データを読み込み、1-NN推論で閉ループ制御 |
| `auto_collect.py` | 自動データ収集ノード。計画的に「側方・後方シフト」を仕込みつつ、転倒検知つきでエピソードを繰り返し、データをnpzに追記保存 |
| `nn_player.py` | rosbag2のデモから、目標距離に最も近い1本を選んで再生する（デモ選択型) 最近傍スクリプト（本題の前段の実証用） |

## 実行手順

### 0. 環境（講座で用いたDockerイメージ）

第4回講義の公式イメージ `airobotbook/ros2-desktop-ai-robot-book-humble:latest` を使用。

```bash
docker pull airobotbook/ros2-desktop-ai-robot-book-humble:latest
docker run -p 6080:80 --shm-size=512m airobotbook/ros2-desktop-ai-robot-book-humble:latest
# ブラウザで http://localhost:6080 を開き、「接続」をクリック → Ubuntuデスクトップ
```

### 1. Gazebo（家のワールド）起動

```bash
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo turtlebot3_house.launch.py
```

残存プロセスがある場合は先に掃除:

```bash
pkill -9 gzserver; pkill -9 gzclient
```

### 2. 人間のお手本（デモ）収集

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard     # i前進 j左l右 k停止
# 別ターミナル:
cd ~ && ros2 bag record -o house_demo4 /scan /cmd_vel /odom   # house_demo4,5,6の3本
```

同じ玄関ドアを、異なる角度・位置から通る3本を収集するのが良い
（1本だけでは「曲がる選択肢」が記憶に足りず、直進したまま失敗することが検証でわかった）。

### 3. 自動データ収集（学習データ増強）

```bash
python3 /home/ubuntu/auto_collect.py
```

シミュレーションを `/reset_simulation` で自動リセットしながら、40エピソードの
走行データを収集して `~/auto_demo_data.npz` に蓄積する。

### 4. 自律走行

```bash
python3 /home/ubuntu/reactive_nn.py
```

起動時に記憶（デモ＋自動収集データ）を読み込んで1-NN推論し、ロボットが自律走行する様子が見られる。

## 工夫している点（詳細はレポート参照）

- 講座で学んだ **模倣学習のワークフロー** をそのままの形で実装（第5回：データ収集→記録→学習→シミュレータ評価→デプロイ）
- 記録データのない穴を埋める **自動データ収集（エピソードリセット方式）** の導入
- 乱数による探索ではなく、**1エピソードに1度だけ予定された側方・後方シフト**という設計にすることで、玄関付近の多様なアプローチを効率よく収集
- 最近傍法に強い「**平均化問題**」（複数経路の平均で危険な中間軌道になる）を1-NN設計で避け、データが少ない場面は **比例制御へのフォールバック** で補うハイブリッド構成
- LIDARの方位を**前方±20度と左右45度窓に分けて**見分け、並走する壁を誤認しない論理設計
- 安全装置・転倒検知・ヒステリシス・平滑化など、学習とは独立した防御層を実装

## 参考にした外部教材（出典）

- Qiita N622「ROS2とGazeboの連携によるロボット操作」— ロボット（TurtleBot3に変更）、ワールド（家ワールドに変更）、駆動プラグイン（`gazebo_ros_diff_drive`ではなくTurtleBot3標準構成）を変えて再実装
- 書籍「ROS 2とPythonで作って学ぶAIロボット入門 改訂第2版」のDocker環境（講座第4回と同じ）
