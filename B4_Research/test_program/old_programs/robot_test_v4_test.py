"""
---robot_test_v4概要説明---
Axis NEURONから送信されてきたPERCEPTION NEURONの右手の位置情報を元にシミュレータ上のxarmに動きを同期させる

---v4更新箇所---
全体を簡潔なものへと書き換え
プログラムを右手の甲のみに対応した形に変更

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

# -------------------------各種標準設定-------------------------
# 描画系統
rbt_s = xarm.XArm7YunjiMobile() # ロボットモデルの読込
# rbt_x = xac.XArm7(host="10.2.0.203:18300")
# init_jnt_angles = rbt_x.get_jnt_vlaues()
# rbt_s.fk("arm", init_jnt_angles)
base = wd.World(cam_pos=[5, -5, 5], lookat_pos=[0, 0, 0]) # モデリング空間を生成
gm.gen_frame().attach_to(base) # 座標系の軸を表示

# NeuronDataReader
# NDR_path = os.path.abspath("NeuronDataReader.dll")
# NDR = cdll.LoadLibrary(NDR_path)

# 通信系統
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((socket.gethostname(), 7001)) # 7001番のポートに接続
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
data_list = [None] # データ格納用リスト
update_frequency = (1/60) # データの更新頻度(s)(ノード18個未満：120Hz 18個以上：60Hz)
main_data_length = 60*16*4
contactRL_length = 4
# ------------------------------------------------------------

# -----その他の変数-----
data_array = []
attached_list = [] # 球のgmモデルについての情報を格納するリスト
onscreen = [] # ロボット描画情報用配列
pre_agv_pos_rgt = np.zeros(3) # 前の時刻の右手の甲の位置に関する配列
operation_count = 0 # 関数動作回数カウンタ
arm_update_weight = 0.01 # armの位置更新の際の重み
fail_IK_flag = 0 # IK解無しフラッグ

rgt_hand_num = 11*16
# --------------------

# データ受信用関数
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
        return [-1]
    tmp_data_array = np.frombuffer(b_msg, dtype=np.float32)

    # 座標調整
    data_array = np.zeros(int(tmp_data_array.size))
    for i in range(0, tmp_data_array.size):
        if (i % 16) == 0 or (i % 16) == 2:
            data_array[i] = tmp_data_array[i] * (-1)
        else:
            data_array[i] = tmp_data_array[i]
    # data_array = data_array*(-1) # 座標調整
    # print(data_array.size)
    return data_array

# シミュレータ描画用関数
def display(attached_list, rbt_s, onscreen, pre_pos, task):
    global data_array, fail_IK_flag, operation_count

    for item in onscreen:
        item.detach()
    # data_array = data_adjustment(data_list)
    if data_adjustment(data_list) == "Data None":
        return task.cont
    else:
        data_array = data_adjustment(data_list)

    if operation_count == 0:
        attached_list.append(gm.gen_sphere(pos=data_array[rgt_hand_num:(rgt_hand_num+3)]))
        attached_list[-1].attach_to(base)

        pre_pos[:] = data_array[rgt_hand_num:(rgt_hand_num+3)]
    else:
        # 球の位置情報を更新
        npvec3 = data_array[rgt_hand_num:(rgt_hand_num + 3)]
        attached_list[0]._objpdnp.setPos(npvec3[0], npvec3[1], npvec3[2])
        # 右手の座標を示す球は緑色に設定
        attached_list[0].set_rgba([0, 1, 0, 1])
        # 現在のアームの位置と回転行列を取得
        tcp_pos, tcp_rotmat = rbt_s.manipulator_dict["arm"].get_gl_tcp()
        # 初期の右手の位置との差分を動きに反映させる
        new_tcp_pos = tcp_pos + arm_update_weight * (npvec3 - pre_pos)
        # この時刻の右手の甲の位置を記録
        pre_pos = data_array[rgt_hand_num:(rgt_hand_num + 3)]
        # 逆運動学によって目標位置におけるジョイントの角度を計算
        jnt_values = rbt_s.ik("arm", new_tcp_pos, tcp_rotmat)
        # 順運動学で先ほどのIKの計算結果を適用（IKが解けないときはFKを更新しない）
        if not (jnt_values is None):
            rbt_s.fk("arm", jnt_values)
            if fail_IK_flag == 1:
                print("-----------------------\nArm operating restart!!\n-----------------------")
                fail_IK_flag = 0
            else:
                fail_IK_flag = 1

        # ロボットモデルを生成して描画
        onscreen.append(rbt_s.gen_meshmodel())
        onscreen[-1].attach_to(base)

    operation_count = operation_count + 1
    return task.cont

threading.Thread(target=data_collect, args=(data_list,)).start()

# update_frequency(s)間隔で情報を更新
taskMgr.doMethodLater(update_frequency, display, "display",
                      extraArgs=[attached_list, rbt_s, onscreen, pre_agv_pos_rgt],
                      appendTask=True)
base.run()
