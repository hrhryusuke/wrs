"""
v1概要説明
Axis NEURONから送信されてきたPERCEPTION NEURONの各関節の位置情報をWRSのシステム上で表示する

備考
指の関節の位置情報に関しては非対応
"""

import socket
import math
from direct.task.TaskManagerGlobal import taskMgr
from ctypes import *
import os

import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import robot_sim.robots.nextage.nextage as nxt
import motion.probabilistic.rrt_connect as rrtc

# NeuronDataReaderを読み込む
NDR_path = os.path.abspath("NeuronDataReader.dll")
NDR = cdll.LoadLibrary(NDR_path)

# モデリング空間を生成
base = wd.World(cam_pos=[5, -5, 5], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base) # 座標系の軸を表示

# サーバー側に接続
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((socket.gethostname(), 7001)) # 7001番のポートに接続
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# while True:
#     msg = s.recv(16384)
#     # print(msg)
#     data = msg.decode()
#     print(data.split(' ')[:2], data.split(' ')[-1])
#     data = data.split(' ')[2:-1]
#     # for i in range(len(data)):
#     print(len(data), len(data)%3)
#     # while True:
#     #     data[]
#     print(data)
#     # bvhf = BVH_FRAME()
#     # bvhf.ParseFromString(msg)
#     # print(bvhf.MotionData)

# データの更新頻度(ノード18個未満：120Hz 18個以上：60Hz)
update_frequency = 1/60
# update_frequency = 0.0001

# データ長の設定
main_data_length = 60*16*4
contactRL_length = 4

attached_list = []

# データ抽出を行う
def update(s, attached_list, task):
    # 配列を空にする
    if len(attached_list) > 0:
        for objcm in attached_list:
            objcm.remove()
    # データ受信
    b_msg = s.recv(main_data_length)
    s.recv(contactRL_length) # contactLRの分を受信だけする（ズレ調整）
    s.recv(contactRL_length)
    # データ型を変更
    data_array = np.frombuffer(b_msg, dtype=np.float32)
    data_array = data_array*(-1)
    # print(b_msg.decode())
    # s_msg=b_msg.decode().strip()
    # data=s_msg.split(' ')[:15*16]
    # print(data_array.size)
    # 各関節の位置データ表示およびテキストファイルへの書き込み
    for i in range(0, data_array.size):
        j = i % 16
        if j == 0:
            # データの数が規定量に達したらbreak
            if i == data_array.size:
                break
            print(data_array[i:i + 3])
            # f = open('C:/Users/Yusuke Hirao/PycharmProjects/wrs/B4_Research/neuron_data/neuron_data_Rindexfinger.txt', 'a')
            # f.write(str(data_array[i:i + 3]))
            # f.write("\n")
            # f.close()

            attached_list.append(gm.gen_sphere(pos=data_array[i:(i + 3)]))
            attached_list[-1].attach_to(base)
            # print(attached_list)

    print("\n\n\n")
    # f = open('C:/Users/Yusuke Hirao/PycharmProjects/wrs/B4_Research/neuron_data/neuron_data_Rindexfinger.txt', 'a')
    # f.write("\n\n\n")
    # f.close()

    return task.cont


# update_frequency(s)間隔で情報を更新
taskMgr.doMethodLater(update_frequency, update, "update",
                      extraArgs=[s, attached_list],
                      appendTask=True)
base.run()

# f = open('C:/Users/Yusuke Hirao/PycharmProjects/wrs/B4_Research/neuron_data.txt', 'w')
# f.write(msg.decode())
# f.close()
