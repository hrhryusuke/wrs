"""
---robot_actual_walking_v1概要説明---
人の歩行動作を検出し，xarm_shuidiの台車部分を動作させる

---更新箇所---

---備考---
対応している台車の動作は前進のみ
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
import robot_sim.robots.xarm7_shuidi_mobile.xarm7_shuidi_mobile as rbs
import robot_con.xarm_shuidi.xarm_shuidi_client as rbx

# NeuronDataReaderを読み込む
# NDR_path = os.path.abspath("NeuronDataReader.dll")
# NDR = cdll.LoadLibrary(NDR_path)

# モデリング空間を生成
base = wd.World(cam_pos=[5, -5, 5], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base) # 座標系の軸を表示
rbt_s = rbs.XArm7YunjiMobile()
rbt_x = rbx.XArmShuidiClient(host="10.2.0.203:18300")
jnt_values = rbt_x.arm_get_jnt_values()
jawwidth = rbt_x.arm_get_jawwidth()
rbt_s.fk(jnt_values=jnt_values)
rbt_s.jaw_to(jawwidth=jawwidth)
rbt_s.gen_meshmodel().attach_to(base)

# サーバー側に接続
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((socket.gethostname(), 7001)) # 7001番のポートに接続
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# データの更新頻度(ノード18個未満：120Hz 18個以上：60Hz)
update_frequency = (1/60)*(2)
# update_frequency = 0.0001

# データ長の設定
main_data_length = 60*16*4
contactRL_length = 4

# 球のgmモデルについての情報を格納するリスト
attached_list = []

# 関数動作回数カウンタ
operation_count = 0

tmp_data = [[0] * 3] * 5
norm = [0] * 6

agv_linear_speed = .2
agv_angular_speed = .5
arm_linear_speed = .03
arm_angular_speed = .1

# データ受信用関数
data_list = [None]
data_array = [None]
def data_collect(data_list):
    while True:
        # データ受信
        data_list[0] = s.recv(main_data_length)
        s.recv(2 * contactRL_length)  # contactLRの分を受信だけする（ズレ調整）

# データ調整用関数
def data_adjustment(data_list):
    global data_array

    # データ型の調整
    b_msg = data_list[0]
    if b_msg is None:
        return "No Data"
    tmp_data_array = np.frombuffer(b_msg, dtype=np.float32)

    # 座標調整
    data_array = np.zeros(int(tmp_data_array.size))
    for i in range(0, tmp_data_array.size):
        if (i % 16) == 0 or (i % 16) == 2:
            data_array[i] = tmp_data_array[i] * (-1)
        else:
            data_array[i] = tmp_data_array[i]

    return data_array

# データ抽出を行う
def update(s, attached_list, task):
    global operation_count, data_array, tmp_data, norm

    if data_adjustment(data_list) == "No Data":
        return task.cont
    else:
        data_array = data_adjustment(data_list)

    # 各関節の位置データ表示およびテキストファイルへの書き込み
    for i in range(0, 16*23, 16):
        # データの数が規定量に達したらbreak
        if i == data_array.size:
            break

        if operation_count == 0:
            if (i > 16 and i < 16 * 8):
                attached_list.append(gm.gen_sphere(pos=data_array[i:(i + 3)]))
                attached_list[-1].attach_to(base)
            # print(attached_list)
        else:
            # attached_list[int(i/16)] = gm.gen_sphere(pos=data_array[i:(i + 3)])
            # attached_list[int(i/16)].attach_to(base)
            if (i > 16 * 1 and i < 16 * 8):
                npvec3 = data_array[i:(i + 3)]
                attached_list[int(i/16)-2]._objpdnp.setPos(npvec3[0], npvec3[1], npvec3[2])
                if i == 16 * 4 or i == 16 * 7:
                    attached_list[int(i/16)-2].set_rgba([0, 1, 0, 1])

                if i is not 16 * 7:
                    tmp_data[int(i / 16) - 2] = data_array[(i + 3):(i + 6)]
                else:
                    for j in range(0, 6):
                        if j is not 5:
                            norm[j] = np.linalg.norm(tmp_data[j], ord=2)
                        else:
                            norm[5] = np.linalg.norm(data_array[(i + 3):(i + 6)], ord=2)

                    # print(f'{np.linalg.norm(norm, ord=2)}')
                    # f = open('C:/Users/Yusuke Hirao/PycharmProjects/wrs/B4_Research/neuron_data/walking_detection_moving_norm.txt', 'a')
                    # f.write(f'{norm}')
                    # f.write("\n")
                    # f.close()

            if np.linalg.norm(norm, ord=2) * 10000 > 900:
                print('Detection result is "Walking"')
                rbt_x.agv_move(linear_speed=agv_linear_speed, angular_speed=0, time_interval=.5)
                # f = open('C:/Users/Yusuke Hirao/PycharmProjects/wrs/B4_Research/neuron_data/walking_detection_stopping_error.txt', 'a')
                # f.write("Detection Error Occurred!!\n")
                # f.close()
            else:
                print('Detection result is "Stopping"')

            # print(attached_list)

        # print(f"{operation_count}\n\n\n")

    operation_count = operation_count + 1
    return task.cont

if __name__ == '__main__':
    threading.Thread(target=data_collect, args=(data_list,)).start()
    # update_frequency(s)間隔で情報を更新
    taskMgr.doMethodLater(update_frequency, update, "update",
                          extraArgs=[s, attached_list],
                          appendTask=True)
    base.run()
