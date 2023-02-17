"""
---robot_test_v3概要説明---
Axis NEURONから送信されてきたPERCEPTION NEURONの情報解析用プログラム

---備考---
"""

import socket
import math
from direct.task.TaskManagerGlobal import taskMgr
from ctypes import *
import os
import threading
import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import robot_sim.robots.nextage.nextage as nxt
import motion.probabilistic.rrt_connect as rrtc
# import robot_con.xarm.xarm_client as xac
import robot_sim.robots.xarm7_shuidi_mobile.xarm7_shuidi_mobile as xarm

# ロボットモデルを読み込み
rbt_s = xarm.XArm7YunjiMobile()
# rbt_x = xac.XArm7(host="10.2.0.203:18300")
# init_jnt_angles = rbt_x.get_jnt_vlaues()
# rbt_s.fk("arm", init_jnt_angles)

# NeuronDataReaderを読み込む
NDR_path = os.path.abspath("../NeuronDataReader.dll")
NDR = cdll.LoadLibrary(NDR_path)

# モデリング空間を生成
base = wd.World(cam_pos=[5, -5, 5], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base) # 座標系の軸を表示

# サーバー側に接続
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((socket.gethostname(), 7001)) # 7001番のポートに接続
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# データの更新頻度(s)(ノード18個未満：120Hz 18個以上：60Hz)
update_frequency = (1/60)

# データ長の設定
main_data_length = 60*16*4
contactRL_length = 4

# 球のgmモデルについての情報を格納するリスト
attached_list = []

# 関数動作回数カウンタ
operation_count = 0

# update内で右手座標の更新を行う頻度(何回に1回右手座標の更新を行うか)
# update_cal_frequency = 1

# ロボット描画情報用配列
onscreen = []

# 前の時刻の右手の甲の位置に関する配列
pre_agv_pos_rgt = np.zeros(3)

# armの位置更新の際の重み
arm_update_weight = 0.01

# データ情報
data_list = [None]

# IK解なし用フラッグ
fail_IK_flag = 0


# データ受信用関数
def data_collect(data_list):
    while True:
        # データ受信
        data_list[0] = s.recv(main_data_length)
        s.recv(2 * contactRL_length)  # contactLRの分を受信だけする（ズレ調整）


# データ抽出および描画
def update(s, attached_list, rbt_s, onscreen, pre_pos, task):
    global operation_count
    global fail_IK_flag
    # データ受信
    b_msg = s.recv(main_data_length)
    s.recv(2 * contactRL_length) # contactLRの分を受信だけする（ズレ調整）

    onscreen.append(rbt_s.gen_meshmodel())
    onscreen[-1].attach_to(base)

    # 配列を空にする
    if len(onscreen) > 0:
        for objcm in onscreen:
            objcm.remove()

    # モデルを削除
    for item in onscreen:
        item.detach()

    # データ型を変更
    b_msg = data_list[0]
    if b_msg is None:
        return task.cont
    tmp_data_array = np.frombuffer(b_msg, dtype=np.float32)

    # 座標調整(位置にしか影響しない)
    data_array = np.zeros(int(tmp_data_array.size))
    for i in range(0, tmp_data_array.size):
        if (i % 16) == 0 or (i % 16) == 2:
            data_array[i] = tmp_data_array[i] * (-1)
        else:
            data_array[i] = tmp_data_array[i]
    # data_array = data_array*(-1) # 座標調整
    # print(data_array.size)

    print(rm.quaternion_to_euler(data_array[(11*16+6):(11*16+10)]))
    # 右手の姿勢をテキストファイルに書き込み
    # f = open('right_hand_euler_down.txt', 'a')
    # f.write(f'{rm.quaternion_to_euler(data_array[(11*16+6):(11*16+10)])}\n')
    # f.close()

    # 右手の姿勢の回転行列をテキストファイルに書き込み
    # f = open('right_hand_rotate_matrix.txt', 'a')
    # f.write(f'{rm.rotmat_from_quaternion(data_array[(11 * 16 + 6):(11 * 16 + 10)])}\n\n')
    # f.close()

    operation_count = operation_count + 1
    return task.cont


threading.Thread(target=data_collect, args=(data_list,)).start()

# update_frequency(s)間隔で情報を更新
taskMgr.doMethodLater(update_frequency, update, "update",
                      extraArgs=[s, attached_list, rbt_s, onscreen, pre_agv_pos_rgt],
                      appendTask=True)
base.run()
