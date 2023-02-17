"""
---robot_actual_machine概要説明---
シミュレータ動作用のrobot_test_v3ベースの実機動作用プログラム
ヌンチャクで台車部分の移動操作を行い、右手の座標をもとに手先位置を操作する

---備考---
"""

import socket
import math
import time
import keyboard
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
import robot_sim.robots.xarm7_shuidi_mobile.xarm7_shuidi_mobile as xarm
import robot_con.xarm_shuidi.xarm_shuidi_client as rbx

# キーボード信号登録用
pressed_keys = {'w': keyboard.is_pressed('w'),
                    'a': keyboard.is_pressed('a'),
                    's': keyboard.is_pressed('s'),
                    'd': keyboard.is_pressed('d'),
                    'r': keyboard.is_pressed('r'),  # x+ global
                    't': keyboard.is_pressed('t'),  # x- global
                    'f': keyboard.is_pressed('f'),  # y+ global
                    'g': keyboard.is_pressed('g'),  # y- global
                    'v': keyboard.is_pressed('v'),  # z+ global
                    'b': keyboard.is_pressed('b'),  # z- gglobal
                    'y': keyboard.is_pressed('y'),  # r+ global
                    'u': keyboard.is_pressed('u'),  # r- global
                    'h': keyboard.is_pressed('h'),  # p+ global
                    'j': keyboard.is_pressed('j'),  # p- global
                    'n': keyboard.is_pressed('n'),  # yaw+ global
                    'm': keyboard.is_pressed('m'),  # yaw- global
                    'o': keyboard.is_pressed('o'),  # gripper open
                    'p': keyboard.is_pressed('p')}  # gripper close

# ロボットモデルを読み込み＆実機の状態をもとに初期化
rbt_s = xarm.XArm7YunjiMobile()
rbt_x = rbx.XArmShuidiClient(host="10.2.0.203:18300")
jnt_values = rbt_x.get_jnt_values()
jaw_width = rbt_x.arm_get_jawwidth()
rbt_s.fk(jnt_values=jnt_values)
rbt_s.jaw_to(jawwidth=jaw_width)
last_jnt_values = jnt_values

# NeuronDataReaderの読込
# NDR_path = os.path.abspath("NeuronDataReader.dll")
# NDR = cdll.LoadLibrary(NDR_path)

# シミュレータ描画系統
base = wd.World(cam_pos=[5, -5, 5], lookat_pos=[0, 0, 0]) # モデリング空間を生成
gm.gen_frame().attach_to(base) # 座標系の軸を表示
# rbt_s.gen_meshmodel().attach_to(base)
attached_list = [] # 関節位置表示用の球(gmモデル)についての情報を格納するリスト
onscreen = [] # ロボット描画情報用配列
pre_agv_pos_rgt = np.zeros(3) # 前の時刻の右手の甲の位置に関する配列
arm_update_weight = 0.01 # armの位置更新の際の重み

# Axis NEURONに接続
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((socket.gethostname(), 7001)) # 7001番のポートに接続
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# データ系統各種
update_frequency = (1/60) # データの更新頻度(s)(ノード18個未満：120Hz 18個以上：60Hz)
main_data_length = 60*16*4 # データ長の設定
contactRL_length = 4

# rbt_x.agv_move(agv_linear_speed=-.1, agv_angular_speed=.1, time_interval=5)
agv_linear_speed = .2
agv_angular_speed = .5
arm_linear_speed = .03
arm_angular_speed = .1

# その他
operation_count = 0 # 関数動作回数カウンタ
data_list = [None] # データ情報
fail_IK_flag = 0 # IK解なし用フラッグ

# update内で右手座標の更新を行う頻度(何回に1回右手座標の更新を行うか)
# update_cal_frequency = 1


# データ受信用関数
def data_collect(data_list):
    while True:
        # データ受信
        data_list[0] = s.recv(main_data_length)
        s.recv(2 * contactRL_length)  # contactLRの分を受信だけする（ズレ調整）
        # print(data_list)

# def operate_yunji():
#     global agv_linear_speed
#     global agv_angular_speed
#     global arm_linear_speed
#     global arm_angular_speed
#
#     while True:
#         pressed_keys = {'w': keyboard.is_pressed('w'),
#                         'a': keyboard.is_pressed('a'),
#                         's': keyboard.is_pressed('s'),
#                         'd': keyboard.is_pressed('d'),
#                         'r': keyboard.is_pressed('r'),  # x+ global
#                         't': keyboard.is_pressed('t'),  # x- global
#                         'f': keyboard.is_pressed('f'),  # y+ global
#                         'g': keyboard.is_pressed('g'),  # y- global
#                         'v': keyboard.is_pressed('v'),  # z+ global
#                         'b': keyboard.is_pressed('b'),  # z- gglobal
#                         'y': keyboard.is_pressed('y'),  # r+ global
#                         'u': keyboard.is_pressed('u'),  # r- global
#                         'h': keyboard.is_pressed('h'),  # p+ global
#                         'j': keyboard.is_pressed('j'),  # p- global
#                         'n': keyboard.is_pressed('n'),  # yaw+ global
#                         'm': keyboard.is_pressed('m'),  # yaw- global
#                         'o': keyboard.is_pressed('o'),  # gripper open
#                         'p': keyboard.is_pressed('p')}  # gripper close
#
#         values_list = list(pressed_keys.values())
#         if pressed_keys["w"] and pressed_keys["a"]:
#             rbt_x.agv_move(linear_speed=agv_linear_speed, angular_speed=agv_angular_speed, time_interval=.5)
#         elif pressed_keys["w"] and pressed_keys["d"]:
#             rbt_x.agv_move(linear_speed=agv_linear_speed, angular_speed=-agv_angular_speed, time_interval=.5)
#         elif pressed_keys["s"] and pressed_keys["a"]:
#             rbt_x.agv_move(linear_speed=-agv_linear_speed, angular_speed=-agv_angular_speed, time_interval=.5)
#         elif pressed_keys["s"] and pressed_keys["d"]:
#             rbt_x.agv_move(linear_speed=-agv_linear_speed, angular_speed=agv_angular_speed, time_interval=.5)
#         elif pressed_keys["w"] and sum(values_list) == 1:  # if key 'q' is pressed
#             rbt_x.agv_move(linear_speed=agv_linear_speed, angular_speed=0, time_interval=.5)
#         elif pressed_keys["s"] and sum(values_list) == 1:  # if key 'q' is pressed
#             rbt_x.agv_move(linear_speed=-agv_linear_speed, angular_speed=0, time_interval=.5)
#         elif pressed_keys["a"] and sum(values_list) == 1:  # if key 'q' is pressed
#             rbt_x.agv_move(linear_speed=0, angular_speed=agv_angular_speed, time_interval=.5)
#         elif pressed_keys["d"] and sum(values_list) == 1:  # if key 'q' is pressed
#             rbt_x.agv_move(linear_speed=0, angular_speed=-agv_angular_speed, time_interval=.5)
#         elif pressed_keys["o"] and sum(values_list) == 1:  # if key 'q' is pressed
#             rbt_x.arm_jaw_to(jawwidth=100)
#         elif pressed_keys["p"] and sum(values_list) == 1:  # if key 'q' is pressed
#             rbt_x.arm_jaw_to(jawwidth=0)

# シミュレータ描画および実機への動作指令
def update(s, attached_list, rbt_s, onscreen, pre_pos, task):
    global operation_count
    global fail_IK_flag
    global last_jnt_values

    # onscreen.append(rbt_s.gen_meshmodel())
    # onscreen[-1].attach_to(base)

    # 配列を空にする
    # if len(onscreen) > 0:
    #     for objcm in onscreen:
    #         objcm.remove()

    # モデルを削除
    for item in onscreen:
        item.detach()

    # データ型を変更
    b_msg = data_list[0]
    if b_msg is None:
        return task.cont
    tmp_data_array = np.frombuffer(b_msg, dtype=np.float32)
    # print(len(tmp_data_array))

    # 座標調整
    data_array = np.zeros(int(tmp_data_array.size))
    for i in range(0, tmp_data_array.size):
        if (i % 16) == 0 or (i % 16) == 2:
            data_array[i] = tmp_data_array[i] * (-1)
        else:
            data_array[i] = tmp_data_array[i]
    # data_array = data_array*(-1) # 座標調整
    # print(data_array.size)

    # 右手の甲の位置データ表示およびテキストファイルへの書き込み
    for i in range(0, 16 * 23, 16):
        # データの数が規定量に達したらbreak
        if i == data_array.size:
            break
        # print(data_array[i:i + 3])

        # 描画処理
        if operation_count == 0:
            # 各部の座標情報を赤色の球として表示
            attached_list.append(gm.gen_sphere(pos=data_array[i:(i + 3)]))
            attached_list[-1].attach_to(base)
            # print(attached_list)

            # 最初の時刻の右手の座標を記録
            pre_pos[:] = data_array[11*16:(11*16+3)]
        else:
             # 更新時点での右手の座標をxarmに反映させる
            if i == 11 * 16:
                tic = time.time()
                # 描画系統
                npvec3 = data_array[i:(i + 3)] # 球の位置情報を更新
                # print(len(attached_list))
                attached_list[int(i / 16)]._objpdnp.setPos(npvec3[0], npvec3[1], npvec3[2])
                attached_list[int(i / 16)].set_rgba([0, 1, 0, 1]) # 右手の座標を示す球は緑色に設定

                # 動作反映系統
                current_jnt_values = rbt_s.get_jnt_values() # 現在のアームの関節角度及び位置と回転行列を取得
                current_tcp_pos, current_tcp_rotmat = rbt_s.manipulator_dict["arm"].get_gl_tcp()
                new_tcp_pos = current_tcp_pos + arm_update_weight * (npvec3 - pre_pos) # 初期の右手の位置との差分を動きに反映させる
                pre_pos = data_array[i:(i + 3)] # この時刻の右手の甲の位置を記録
                jnt_values = rbt_s.ik("arm", new_tcp_pos, current_tcp_rotmat) # 逆運動学によって目標位置におけるジョイントの角度を計算
                # 順運動学で先ほどのIKの計算結果を適用（IKが解けないときはFKを更新しない）
                if not(jnt_values is None):
                    rbt_s.fk("arm", jnt_values)
                    if fail_IK_flag == 1:
                        print("-----------------------\nArm operating restart!!\n-----------------------")
                        fail_IK_flag = 0
                else:
                    fail_IK_flag = 1

                # ロボットモデルを生成して描画
                onscreen.append(rbt_s.gen_meshmodel())
                onscreen[-1].attach_to(base)
                # To move the actual robot
                toc = time.time()
                start_frame_id = math.ceil((toc - tic) / .01)
                rbt_x.arm_move_jspace_path([last_jnt_values, jnt_values], time_interval=.001,
                                           start_frame_id=start_frame_id)
                last_jnt_values = jnt_values
            # print(attached_list)

        # print(f"{operation_count}")

    operation_count = operation_count + 1
    return task.cont


threading.Thread(target=data_collect, args=(data_list,)).start()
# threading.Thread(target=operate_yunji, args=()).start()

# update_frequency(s)間隔で情報を更新
taskMgr.doMethodLater(update_frequency, update, "update",
                      extraArgs=[s, attached_list, rbt_s, onscreen, pre_agv_pos_rgt],
                      appendTask=True)
base.run()