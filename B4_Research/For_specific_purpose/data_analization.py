"""
---data_analization概要説明---
Axis NEURONから送信されてきたPERCEPTION NEURONの各関節の位置情報をWRSのシステム上で表示する

---更新箇所---

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

# NeuronDataReaderを読み込む
# NDR_path = os.path.abspath("NeuronDataReader.dll")
# NDR = cdll.LoadLibrary(NDR_path)

# モデリング空間を生成
base = wd.World(cam_pos=[5, -5, 5], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base) # 座標系の軸を表示

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

attached_list_rot = []

# 関数動作回数カウンタ
operation_count = 0

adjusment_rotmat = rm.rotmat_from_euler(0, 135, 0)
# adjusment_rotmat_hip = np.eye(3)
# adjusment_rotmat_hip = np.array([[0, 0, 1],
#                              [0, -1, 0],
#                              [1, 0, 0]])
adjusment_rotmat_hip = np.array([[0, 0, -1],
                             [0, -1, 0],
                             [-1, 0, 0]])
# adjusment_rotmat_hip = np.array([[1, 0, 0],
#                                 [0, 1, 0],
#                                 [0, 0, -1]])

# データ受信用関数
data_list = [None]
data_array = [None]
def data_collect(data_list):
    while True:
        # データ受信
        data_list[0] = s.recv(main_data_length)
        s.recv(2 * contactRL_length)  # contactLRの分を受信だけする（ズレ調整）

def model_remove(onscreen):
    for item in onscreen:
        item.detach()
    onscreen.clear()

# データ調整用関数
def data_adjustment(data_list):
    global data_array, adjusment_rotmat

    # データ型の調整
    b_msg = data_list[0]
    if b_msg is None:
        return "No Data"
    tmp_data_array = np.frombuffer(b_msg, dtype=np.float32)

    # 座標調整
    data_array = np.zeros(int(tmp_data_array.size))
    for i in range(0, tmp_data_array.size):
        # if (i % 16) == 0 or (i % 16) == 2:
        #     data_array[i] = tmp_data_array[i] * (-1)
        # else:
        #     data_array[i] = tmp_data_array[i]

        if (i % 16) == 2:
            adjustmented_pos = np.dot(adjusment_rotmat, np.array(tmp_data_array[i-2: i+1]).T)
            data_array[i-2: i+1] = adjustmented_pos
        elif (i % 16) == 9:
            tmp_rot1 = np.dot(adjusment_rotmat_hip, rm.rotmat_from_quaternion(tmp_data_array[i - 3: i + 1])[:3, :3])
            tmp_rot2 = np.dot(rm.rotmat_from_euler(0, 0, 180), tmp_rot1)
            # data_array[i-3: i+1] = rm.quaternion_from_matrix(np.dot(rm.rotmat_from_euler(0, 0, 0), tmp_rot1))
            tmp_euler = rm.rotmat_to_euler(tmp_rot1)
            hip_euler = np.zeros(3)
            hip_euler[0] = tmp_euler[2]
            hip_euler[1] = -tmp_euler[1]
            hip_euler[2] = tmp_euler[0]
            data_array[i - 3: i + 1] = rm.quaternion_from_euler(ai=hip_euler[0], aj=hip_euler[1], ak=hip_euler[2])
        else:
            data_array[i] = tmp_data_array[i]

        # if (i % 16) == 8 or (i % 16) == 9:
        #     data_array[i] = data_array[i] * -1

    return data_array

# データ抽出を行う
def update(s, attached_list, task):
    global operation_count, data_array
    # # 配列を空にする
    # if len(attached_list) > 0:
    #     for objcm in attached_list:
    #         objcm.remove()

    model_remove(attached_list_rot)

    if data_adjustment(data_list) == "No Data":
        return task.cont
    else:
        data_array = data_adjustment(data_list)

    if operation_count % 2 == 0:
        # 各関節の位置データ表示およびテキストファイルへの書き込み
        for i in range(0, 16*23, 16):
            # データの数が規定量に達したらbreak
            if i == data_array.size:
                break

            # print(data_array[i:i + 3])

            # f = open('C:/Users/Yusuke Hirao/PycharmProjects/wrs/B4_Research/neuron_data/neuron_data_Rindexfinger.txt', 'a')
            # f.write(str(data_array[i:i + 3]))
            # f.write("\n")
            # f.close()

            # attached_list.append(gm.gen_sphere(pos=data_array[i:(i + 3)]))
            # attached_list[-1].attach_to(base)
            # print(attached_list)

            if operation_count == 0:
                attached_list.append(gm.gen_sphere(pos=data_array[i:(i + 3)]))
                attached_list[-1].attach_to(base)
                # print(attached_list)
            else:
                # attached_list[int(i/16)] = gm.gen_sphere(pos=data_array[i:(i + 3)])
                # attached_list[int(i/16)].attach_to(base)
                npvec3 = data_array[i:(i + 3)]
                attached_list[int(i/16)]._objpdnp.setPos(npvec3[0], npvec3[1], npvec3[2])
                # 特定センサ番号
                if i == 11*16:
                    attached_list[int(i/16)].set_rgba([0, 1, 1, 1])
                    attached_list_rot.append(gm.gen_frame(pos=data_array[i:(i + 3)], rotmat=rm.rotmat_from_quaternion(data_array[(i + 6):(i + 10)])[:3, :3]))
                    attached_list_rot[-1].attach_to(base)
                    print(rm.rotmat_from_quaternion(data_array[(i + 6):(i + 10)])[:3, :3])
                # elif i == 21*16:
                #     attached_list_rot.append(gm.gen_frame(pos=data_array[i:(i + 3)], rotmat=rm.rotmat_from_quaternion(data_array[(i + 6):(i + 10)])[:3, :3]))
                #     attached_list_rot[-1].attach_to(base)
                #     print(rm.rotmat_from_quaternion(data_array[(i + 6):(i + 10)])[:3, :3])

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
