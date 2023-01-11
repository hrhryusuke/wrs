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
NDR_path = os.path.abspath("../../NeuronDataReader.dll")
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

# 差分計算時に無視する数値の閾値
DIFF_THRESHOLD = 1

# 描画用データ格納用リスト
attached_list = []

# 関数動作回数カウンタ
operation_count = 0
# 位置情報保存用リスト
data_memory_list = []

# データ抽出を行う
def update(s, attached_list, data_memory_list, task):
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
    data_array = data_array * (-1) # 向きの修正
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
            # print(data_array[i:i + 3])
            f = open('/B4_Research/neuron_data/neuron_data_static.txt', 'a')
            f.write(str(data_array[i:i + 3]))
            f.write("\n")
            f.close()
            # １つ前のデータとの差分を書き込む
            # if i != 0:
            #     diff = data_array[i:i + 3] - data_array[i - 16:i - 13]
            #     # 差が小さい数値は無視
            #     for k in range(0, 3):
            #         if diff[k] < DIFF_THRESHOLD:
            #             diff[k] = 0
            #     print(diff)
            #     f = open('C:/Users/Yusuke Hirao/PycharmProjects/wrs/B4_Research/neuron_data/for_detection/finger.txt', 'a')
            #     f.write(str(diff))
            #     f.write("\n")
            #     f.close()
            attached_list.append(gm.gen_sphere(pos=data_array[i:(i + 3)]))
            attached_list[-1].attach_to(base)
            # print(attached_list)
            data_memory_list.append(data_array[i:i + 3])

    # 一つ前のデータとの差分を計算および書き込み
    # if i != 0:
    #     for k in range(0, 60):
    #         diff = data_memory_list[i:i + 3] - data_memory_list[i - 60:i - 57]
    #     # 変化が小さい数値は無視
    #     for l in range(0, 3):
    #         if diff[l] < DIFF_THRESHOLD:
    #             diff[l] = 0


    # print("\n\n\n")
    f = open('/B4_Research/neuron_data/neuron_data_static.txt', 'a')
    f.write("\n\n\n")
    f.close()

    # 一つ前の時刻のデータとの差分を計算および書き込み
    global operation_count
    diff = [[0] for p in range(0,3)]
    # 差分を取るため2回目の実行から計算
    start_flag = 60 * operation_count
    if operation_count != 0:
        for k in range(0, 60):
            diff[k][0] = data_memory_list[(start_flag+k)] - data_memory_list[(start_flag+k)-60]
            diff[k][1] = data_memory_list[(start_flag+k)+1] - data_memory_list[(start_flag + k)-59]
            diff[k][2] = data_memory_list[(start_flag+k)+2] - data_memory_list[(start_flag + k)-58]
        # 変化が小さい数値は無視
            for m in range(0, 3):
                if diff[k][m] < DIFF_THRESHOLD:
                    diff[k][m] = 0

    print(diff)
    print("\n\n\n")

    # 動作回数カウンタ
    # print(operation_count)
    operation_count = operation_count + 1

    return task.cont


# 0.01秒間隔で情報を更新
taskMgr.doMethodLater(update_frequency, update, "update",
                      extraArgs=[s, attached_list, data_memory_list],
                      appendTask=True)
base.run()

# f = open('C:/Users/Yusuke Hirao/PycharmProjects/wrs/B4_Research/neuron_data.txt', 'w')
# f.write(msg.decode())
# f.close()
