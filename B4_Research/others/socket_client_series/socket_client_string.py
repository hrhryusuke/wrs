import socket
import math
import numpy as np
from direct.task.TaskManagerGlobal import taskMgr

import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import robot_sim.robots.nextage.nextage as nxt
import motion.probabilistic.rrt_connect as rrtc

#モデリング空間を生成
base = wd.World(cam_pos=[5, -5, 5], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base) #座標系の軸を表示

#サーバー側に接続
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((socket.gethostname(), 7003))
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

attached_list = []

#データ抽出を行う
def update(s, attached_list, task):
    #不必要なデータを除去
    if len(attached_list) > 0:
        for objcm in attached_list:
            objcm.remove()
    #データ受信
    b_msg = s.recv(59*16*4)
    s.recv(2)
    s.recv(2)
    #データ型を変更
    data_array = np.frombuffer(b_msg, dtype=np.float32)
    # print(b_msg.decode())
    # s_msg=b_msg.decode().strip()
    # data=s_msg.split(' ')[:15*16]
    print(data_array.size)
    for i in range(0, data_array.size):
        j = i % 16
        if j == 0:
            #データの数が規定量に達したらbreak
            if i == 59*16*4:
                break
            print(data_array[i:i + 3])
            attached_list.append(gm.gen_sphere(pos=data_array[i:i + 3]))
            attached_list[-1].attach_to(base)
    return task.cont

#0.01秒間隔で情報を更新
taskMgr.doMethodLater(0.01, update, "update",
                      extraArgs=[s, attached_list],
                      appendTask=True)
base.run()

# f = open('C:/Users/Yusuke Hirao/PycharmProjects/wrs/B4_Research/neuron_data.txt', 'w')
# f.write(msg.decode())
# f.close()
