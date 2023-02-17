"""
---get_camrobot_coordinate---
操作者の手のセンサ情報から，カメラロボット座標系を定義

---備考---
リアルタイム更新と初期時刻で固定か選択可能

---更新日---
20221214
"""

import cv2
import time
import math
import pickle
import struct
import socket
import tkinter.messagebox as mb
import keyboard
import threading
import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import robot_sim.robots.xarm_shuidi.xarm_shuidi as xsm
import robot_con.xarm_shuidi.xarm_shuidi_x as xsc
import drivers.devices.realsense_rpi.realsense_client as realsense_client

from direct.task.TaskManagerGlobal import taskMgr

# -------------------------各種標準設定-------------------------
# 描画系統
base = wd.World(cam_pos=[5, 5, 2], lookat_pos=[0, 0, 1]) # モデリング空間を生成
rbt_s = xsm.XArmShuidi(enable_cc=True) # ロボットモデルの読込

# 通信系統
is_data_receive = True
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((socket.gethostname(), 7001)) # 7001番のポートに接続
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
data_list = [None] # データ格納用リスト
pre_data_list = [None]
update_frequency = (1/60) # データの更新頻度(s)(ノード18個未満：120Hz 18個以上：60Hz)
main_data_length = 60*16*4
contactRL_length = 4

# モーションキャプチャ番号
rgt_hand_num = 11*16
hip_num = 21*16

# 操作者座標系 & カメラロボット座標系調整用
operator_coordinate = np.eye(3)
camrobot_coordinate = np.eye(3)
adjusment_rotmat_hip = np.array([[0, 0, -1],
                                 [0, -1, 0],
                                 [-1, 0, 0]])
# -------------------------------------------------------------
# ----------------------
data_array = []
onscreen = []
count = 0
# ----------------------


# モデル除去用関数
def model_remove(onscreen):
    for item in onscreen:
        item.detach()
    onscreen.clear()

# データ受信用関数
def data_collect(data_list):
    global is_data_receive, pre_data_list
    while True:
        if is_data_receive == True:
            # データ受信
            data_list[0] = s.recv(main_data_length)
            s.recv(2 * contactRL_length)  # contactLRの分を受信だけする（ズレ調整）
            pre_data_list = data_list
        else:
            data_list = pre_data_list


# データ調整用関数
def data_adjustment(data_list):
    global data_array
    gripper_euler = np.zeros(3)

    adjustment_rotmat1 = rm.rotmat_from_euler(0, 135, 0)  # 全変数調整用
    adjustment_rotmat2 = rm.rotmat_from_euler(0, 0, 90)  # グリッパ姿勢調整用

    # データ型の調整
    b_msg = data_list[0]
    if b_msg is None:
        return "No Data"
    tmp_data_array = np.frombuffer(b_msg, dtype=np.float32)

    # 座標調整
    data_array = np.zeros(int(tmp_data_array.size))
    for i in range(0, tmp_data_array.size):
        if (i % 16) == 2: # センサ座標調整
            adjustmented_pos = np.dot(adjustment_rotmat1, np.array(tmp_data_array[i-2: i+1]).T)
            data_array[i-2: i+1] = adjustmented_pos
        elif (i % 16) == 9: # グリッパ姿勢調整
            adjustmented_rot1 = np.dot(adjustment_rotmat1, rm.rotmat_from_quaternion(tmp_data_array[i-3: i+1])[:3, :3])
            adjustmented_rot2 = np.dot(adjustment_rotmat2, adjustmented_rot1)
            # adjustment_rotmat2を用いたグリッパ姿勢指定後の回転軸補正
            tmp_euler = rm.rotmat_to_euler(adjustmented_rot2)
            gripper_euler[0] = -tmp_euler[1]
            gripper_euler[1] = -tmp_euler[0]
            gripper_euler[2] = tmp_euler[2]
            data_array[i-3: i+1] = rm.quaternion_from_euler(ai=gripper_euler[0], aj=gripper_euler[1], ak=gripper_euler[2])
        else:
            data_array[i] = tmp_data_array[i]

    return data_array


def get_camrobot_coordinate(task):
    # データ受信が出来ているかの判定
    if data_adjustment(data_list) == "No Data":
        return task.cont
    else:
        data_array = data_adjustment(data_list)

    # リアルタイム描画のため、各時刻のモデルを除去
    model_remove(onscreen)

    # 操作者座標系の定義
    if count == 0:
        b_msg = data_list[0]
        if b_msg is None:
            return "No Data"
        tmp_data_array = np.frombuffer(b_msg, dtype=np.float32)
        tmp_euler = rm.rotmat_to_euler(
            np.dot(adjusment_rotmat_hip, rm.rotmat_from_quaternion(tmp_data_array[hip_num + 6: hip_num + 10])[:3, :3]))
        operator_coordinate_euler = np.zeros(3)
        operator_coordinate_euler[0] = tmp_euler[2]
        operator_coordinate_euler[1] = -tmp_euler[1]
        operator_coordinate_euler[2] = tmp_euler[0]
        operator_coordinate[:2, :2] = rm.rotmat_from_euler(ai=operator_coordinate_euler[0],
                                                           aj=operator_coordinate_euler[1],
                                                           ak=operator_coordinate_euler[2])[:2, :2]
        operator_coordinate_frame_color = np.array([[0, 1, 1],
                                                    [1, 0, 1],
                                                    [1, 1, 0]])
        gm.gen_frame(pos=np.zeros(3), rotmat=operator_coordinate, length=2, thickness=0.03,
                     rgbmatrix=operator_coordinate_frame_color).attach_to(base)



    return task.cont



