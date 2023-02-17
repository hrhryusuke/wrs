"""
---robot_test_v5概要説明---
Axis NEURONから送信されてきたPERCEPTION NEURONの右手の位置姿勢情報を元にシミュレータ上のxarmに動きを同期させる

---v5更新箇所---
右手の姿勢情報の反映を追加

---備考---
"""
import socket
import math
import time
import keyboard
import basis.trimesh.transformations as tr
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

# -------------------------各種標準設定-------------------------
# ロボット関係
rbt_s = xarm.XArm7YunjiMobile(enable_cc=True) # ロボットモデルの読込
rbt_x = rbx.XArmShuidiClient(host="10.2.0.203:18300")
init_jnt_values = rbt_x.get_jnt_values()
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

pre_agv_pos_rgt = np.zeros(3)
pre_rgt_hand_info = np.zeros(6) # 前の時刻の右手の甲の位置姿勢に関する配列
current_rgt_hand_info = np.zeros(6)
displacement = None
operation_count = 0 # 関数動作回数カウンタ
arm_update_weight = 0.1 # armの位置更新の際の重み
arm_stop_flag = 0 # 実機動作停止フラッグ

current_jnt_values_rgt = None
jnt_displacement = None
threshold_jnt_displacement = 0.1
pre_jnt_displacement = None
k = np.array([0, 0, 0, 0, 0, 0, 0])
pre_manipulability = None
current_manipulability = None
manipulability_displacement = None

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
        return "No Data"
    tmp_data_array = np.frombuffer(b_msg, dtype=np.float32)

    # 座標調整
    data_array = np.zeros(int(tmp_data_array.size))
    for i in range(0, tmp_data_array.size):
        if (i % 16) == 0 or (i % 16) == 2:
            data_array[i] = tmp_data_array[i] * (-1)
        else:
            data_array[i] = tmp_data_array[i]

        if (i % 16) == 8 or (i % 16) == 9:
            data_array[i] = data_array[i] * -1

    return data_array

# シミュレータ描画用関数
def operate(attached_list, rbt_s, onscreen, pre_pos, task):
    global k, pre_manipulability, current_manipulability, manipulability_displacement, data_array, pre_rgt_hand_info, current_rgt_hand_info, \
        current_jnt_values_rgt, displacement, pre_jnt_displacement, arm_stop_flag, operation_count

    for item in onscreen:
        item.detach()

    if data_adjustment(data_list) == "No Data":
        return task.cont
    else:
        data_array = data_adjustment(data_list)

    if operation_count == 0:
        attached_list.append(gm.gen_sphere(pos=data_array[rgt_hand_num:(rgt_hand_num+3)]))
        attached_list[-1].attach_to(base)

        pre_pos[:] = data_array[rgt_hand_num:(rgt_hand_num+3)]
        pre_rgt_hand_info = np.hstack([data_array[rgt_hand_num:(rgt_hand_num+3)], np.array(list(tr.euler_from_quaternion(data_array[(rgt_hand_num+6):(rgt_hand_num+10)])))])
        pre_manipulability = rbt_s.manipulator_dict['arm'].jlc._ikt.manipulability(tcp_jntid=7)
    else:
        tic = time.time()
        # 球の描画処理
        npvec3 = data_array[rgt_hand_num:(rgt_hand_num + 3)]
        attached_list[0]._objpdnp.setPos(npvec3[0], npvec3[1], npvec3[2])
        attached_list[0].set_rgba([0, 1, 0, 1])

        # 制御処理
        current_rgt_hand_info = np.hstack([data_array[rgt_hand_num:(rgt_hand_num+3)], np.array(list(tr.euler_from_quaternion(data_array[(rgt_hand_num+6):(rgt_hand_num+10)])))])
        displacement = current_rgt_hand_info - pre_rgt_hand_info
        pre_rgt_hand_info = current_rgt_hand_info
        current_jnt_values = rbt_s.get_jnt_values(component_name="arm")

        ef_jacobian = rbt_s.manipulator_dict['arm'].jlc._ikt.jacobian(tcp_jntid=7)
        ef_jacobian_pinv = np.linalg.pinv(ef_jacobian)

        if not operation_count == 1:
            current_manipulability = rbt_s.manipulator_dict['arm'].jlc._ikt.manipulability(tcp_jntid=7)
            manipulability_displacement = current_manipulability - pre_manipulability
            if np.all(abs(pre_jnt_displacement) > 0.00001):
                k = 0.001 * (manipulability_displacement / pre_jnt_displacement)
            else:
                print("Exception Occured!!")

        jnt_displacement = np.dot(ef_jacobian_pinv, displacement) + np.dot((np.eye(7)-np.dot(ef_jacobian_pinv, ef_jacobian)), k)
        pre_jnt_displacement = jnt_displacement
        pre_manipulability = rbt_s.manipulator_dict['arm'].jlc._ikt.manipulability(tcp_jntid=7)

        if (not (np.any(jnt_displacement > threshold_jnt_displacement)) or np.any(jnt_displacement < -1 * threshold_jnt_displacement)) and pre_manipulability >= 0.01:
            rbt_s.fk("arm", current_jnt_values + jnt_displacement)
            collided_result = rbt_s.is_collided()
            # print(collided_result)
            if collided_result == False:
                toc = time.time()
                start_frame_id = math.ceil((toc - tic) / .01)
                rbt_x.move_jnts(component_name="arm", jnt_values=current_jnt_values + jnt_displacement)
                # rbt_x.arm_move_jspace_path([current_jnt_values, current_jnt_values + jnt_displacement], start_frame_id=start_frame_id)
                if arm_stop_flag == 1:
                    print("-----------------------\nArm operating restart!!\n-----------------------")
                    arm_stop_flag = 0
        else:
            arm_stop_flag = 1

        # ロボットモデルを生成して描画
        onscreen.append(rbt_s.gen_meshmodel())
        onscreen[-1].attach_to(base)

    operation_count = operation_count + 1
    return task.cont

if __name__ == '__main__':
    # arm_homeconf = np.array([0, -math.pi / 3, 0, math.pi / 12, 0, -math.pi / 12])
    # rbt_x.arm_move_jspace_path([init_jnt_values, arm_homeconf])
    # rbt_s.fk("arm", arm_homeconf)
    # time.sleep(1)

    threading.Thread(target=data_collect, args=(data_list,)).start()
    taskMgr.doMethodLater(update_frequency, operate, "operate",
                          extraArgs=[attached_list, rbt_s, onscreen, pre_agv_pos_rgt],
                          appendTask=True)
    base.run()
