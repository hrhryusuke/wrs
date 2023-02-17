"""
---get_camrobot_coordinate---
操作者の手のセンサ情報から，カメラロボット座標系を定義

---備考---
リアルタイム更新と初期時刻で固定か選択可能

---更新日---
20230111
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
# import robot_con.xarm_shuidi.xarm_shuidi_x as xsc
# import drivers.devices.realsense_rpi.realsense_client as realsense_client

from direct.task.TaskManagerGlobal import taskMgr

# -------------------------各種標準設定-------------------------
# 描画系統
base = wd.World(cam_pos=[3, 3, 3], lookat_pos=[0, 0, 0.5]) # モデリング空間を生成
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
onscreen_tcp = []
onscreen_operator = []
operation_count = 0

# アーム・台車動作系統
init_error = np.zeros(3)
init_tcp_rot = np.eye(3)
current_jnt_values = None
pre_jnt_values = None
pre_agv_pos = np.zeros(3)

# 異常動作検知系統
abnormal_flag = False
abnormal_count = 0
abnormal_cancel_count = 0
stop_standard_pos = np.zeros(3)
stop_standard_rot = np.zeros((3, 3))
cancel_pos_displacement = np.zeros(3)
cancel_euler_displacement = np.zeros(3)
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
        return False
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


def abnormal_judgement(standard_pos, standard_rot, current_pos, current_rot, is_stable_cancel):
    global abnormal_count, abnormal_cancel_count, cancel_pos_displacement, cancel_euler_displacement
    threshold_pos = 0.05
    threshold_rot_euler = 0.5

    pre_judge_pos = np.zeros(3)
    pre_judge_rot = np.zeros((3, 3))

    # 動作停止時の手の位置姿勢を記録
    if abnormal_count == 1:
        pre_judge_pos = current_pos
        pre_judge_rot = current_rot

    pos_judge = (abs(current_pos - standard_pos) <= threshold_pos).all()
    rot_judge = np.linalg.norm(rm.rotmat_to_euler(current_rot) - rm.rotmat_to_euler(standard_rot), ord=2) <= threshold_rot_euler
    print(f"displacement[pos]{abs(current_pos - standard_pos)}, [euler_norm]{np.linalg.norm(rm.rotmat_to_euler(current_rot) - rm.rotmat_to_euler(standard_rot), ord=2)}")
    print(f"judgement[pos]{pos_judge}, [euler_norm]{rot_judge}")

    if pos_judge and rot_judge:
        # マニピュレータの手先の位置姿勢と操作者の手先の位置姿勢の比較
        # 動作復帰条件クリア
        abnormal_cancel_count = 0
        print('operating restart!!')
        return False
    elif is_stable_cancel == True:
        # 操作者の手先の位置姿勢の比較
        # 一定時間手を動かさなければ動作復帰
        if np.all(current_pos - pre_judge_pos) <= 0.5 and np.all(rm.rotmat_to_euler(current_rot)-rm.rotmat_to_euler(pre_judge_rot)) <= 0.5:
            abnormal_cancel_count += 1
            pre_judge_pos = current_pos
            pre_judge_rot = current_rot
        else:
            abnormal_cancel_count = 0

        if abnormal_cancel_count <= 30:
            if abnormal_count == 1:
                print("abnormal operation is detected")
            print(f'pos_error: {abs(current_pos - standard_pos)}, euler_error: {rm.rotmat_to_euler(current_rot) - rm.rotmat_to_euler(standard_rot)}')
            return True
        else:
            print('operating is restarted!!')
            cancel_pos_displacement += current_pos - standard_pos
            cancel_euler_displacement += rm.rotmat_to_euler(standard_rot) - rm.rotmat_to_euler(current_rot)
            print(f'cancel_pos_displacement:{cancel_pos_displacement}, cancel_euler_displacement:{cancel_euler_displacement}')
            abnormal_cancel_count = 0
            return False
    else:
        return True


def tangent_angle(v0, v1):
    """
    calculate tangent angle "from v0 to v1"
    input:
        v0: numpy array
        v1: numpy array
    return:
        tangent angle(rad)
    """
    inner = np.inner(v0, v1)
    cross = np.cross(v0, v1)
    norm = np.linalg.norm(v0) * np.linalg.norm(v1)
    angle = np.arccos(np.clip(inner / norm, -1.0, 1.0))
    if cross >= 0:
        return angle
    else:
        return -1 * angle


def operate_camrobot(task):
    global data_array, init_error, init_tcp_rot, current_jnt_values, pre_jnt_values, operation_count, operator_coordinate, camrobot_coordinate, \
        abnormal_flag, abnormal_count, stop_standard_pos, stop_standard_rot, cancel_pos_displacement, cancel_euler_displacement

    # 異常動作検知の閾値
    threshold_abnormal_jnt_values = 0.5
    threshold_abnormal_jnt_values_norm = 0.5

    # データ受信が出来ているかの判定
    if data_adjustment(data_list) is False:
        return task.cont
    else:
        data_array = data_adjustment(data_list)

    # リアルタイム描画のため、各時刻のモデルを除去
    model_remove(onscreen)
    model_remove(onscreen_tcp)
    model_remove(onscreen_operator)

    if operation_count == 0:
        b_msg = data_list[0]
        if b_msg is None:
            return task.cont
        tmp_data_array = np.frombuffer(b_msg, dtype=np.float32)
        tmp_euler = rm.rotmat_to_euler(
            np.dot(adjusment_rotmat_hip, rm.rotmat_from_quaternion(tmp_data_array[hip_num + 6: hip_num + 10])[:3, :3]))

        # 操作者座標系の定義
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

        operator_direction_vector = np.dot(operator_coordinate, np.array([1, 0, 0]))
        rotation_world_to_operator = tangent_angle(v0=np.array([1, 0, 0], v1=operator_direction_vector))

        # カメラロボット座標系の定義
        init_operator_hand_rot = np.dot(np.linalg.pinv(operator_coordinate),
                                        rm.rotmat_from_quaternion(data_array[(rgt_hand_num + 6):(rgt_hand_num + 10)])[:3, :3])
        camrobot_direction_vector = np.dot(init_operator_hand_rot, np.array([0, 0, 1]))
        rotation_operator_to_camrobot = tangent_angle(v0=operator_direction_vector, v1=camrobot_direction_vector)
        # init_operator_hand_pos = np.dot(np.linalg.pinv(operator_coordinate), data_array[rgt_hand_num:(rgt_hand_num + 3)])
        # gm.gen_frame(pos=init_operator_hand_pos, rotmat=init_operator_hand_rot).attach_to(base)
        # gm.gen_dashstick(spos=init_operator_hand_pos,
        #                  epos=np.array([init_operator_hand_pos[0], init_operator_hand_pos[1], 0])).attach_to(base)
        # gm.gen_arrow(spos=np.array([init_operator_hand_pos[0], init_operator_hand_pos[1], 0]),
        #              epos=np.array([init_operator_hand_pos[0] + camrobot_direction_vector[0], init_operator_hand_pos[1] + camrobot_direction_vector[1], 0])).attach_to(base)
        camrobot_coordinate = rm.rotmat_from_axangle(axis=[0, 0, 1],
                                                     angle=rotation_operator_to_camrobot)
        gm.gen_frame(rotmat=camrobot_coordinate, length=2, thickness=0.03).attach_to(base)
        rbt_s.fk(component_name="agv",
                 jnt_values=np.array([0, 0, rotation_world_to_operator + rotation_operator_to_camrobot]))

        # 初期エラー等の記録
        init_tcp_pos, init_tcp_rot = rbt_s.get_gl_tcp()
        init_error = np.dot(np.linalg.pinv(operator_coordinate), data_array[rgt_hand_num:(rgt_hand_num+3)])-init_tcp_pos
        pre_jnt_values = rbt_s.get_jnt_values(component_name="arm")
    else:
        # マニピュレータの手先の座標系表示
        operator_hand_pos = np.dot(np.linalg.pinv(operator_coordinate), data_array[rgt_hand_num:(rgt_hand_num + 3)])
        operator_hand_rot = np.dot(np.linalg.pinv(operator_coordinate),
                                   rm.rotmat_from_quaternion(data_array[(rgt_hand_num + 6):(rgt_hand_num + 10)])[:3, :3])

        # 操作者座標系から見た右手の位置姿勢についてLM法で逆運動学の解を導出
        camrobot_hand_pos = operator_hand_pos
        camrobot_hand_rot = rm.rotmat_from_euler(ai=-1*rm.rotmat_to_euler(operator_hand_rot)[1],
                                                 aj=rm.rotmat_to_euler(operator_hand_rot)[0],
                                                 ak=rm.rotmat_to_euler(operator_hand_rot)[2])

        current_jnt_values = rbt_s.manipulator_dict['arm'].jlc._ikt.num_ik(tgt_pos=camrobot_hand_pos - init_error - cancel_pos_displacement,
                                                                           tgt_rot=np.dot(rm.rotmat_from_euler(ai=cancel_euler_displacement[0], aj=cancel_euler_displacement[1], ak=cancel_euler_displacement[2]),
                                                                                          camrobot_hand_rot),
                                                                           seed_jnt_values=pre_jnt_values,
                                                                           tcp_jnt_id=7)

        onscreen_tcp.append(gm.gen_frame(pos=camrobot_hand_pos - init_error - cancel_pos_displacement,
                                         rotmat=np.dot(rm.rotmat_from_euler(ai=cancel_euler_displacement[0],
                                                                            aj=cancel_euler_displacement[1],
                                                                            ak=cancel_euler_displacement[2]),
                                                       camrobot_hand_rot)))
        onscreen_tcp[-1].attach_to(base)

        # 関節角度の更新と衝突判定
        if current_jnt_values is not None:
            if abnormal_flag == False:
                # 異常動作判定
                if not ((abs(current_jnt_values - pre_jnt_values) <= threshold_abnormal_jnt_values).all() and
                    abs(np.linalg.norm(current_jnt_values - pre_jnt_values, ord=2)) <= threshold_abnormal_jnt_values_norm):
                    if operation_count >= 10:
                        abnormal_flag = True
            else:
                # 操作復帰判定
                abnormal_count += 1
                if abnormal_count == 1:
                    # 動作停止時のマニピュレータの手先の位置姿勢の記録
                    stop_standard_pos, stop_standard_rot = rbt_s.get_gl_tcp()
                    print("abnormal operation is detected!")
                else:
                    camrobot_hand_pos = np.dot(np.linalg.pinv(camrobot_coordinate), data_array[(rgt_hand_num):(rgt_hand_num + 3)])
                    camrobot_hand_rot = np.dot(np.linalg.pinv(camrobot_coordinate), rm.rotmat_from_quaternion(data_array[(rgt_hand_num + 6):(rgt_hand_num + 10)])[:3, :3])

                    judge = abnormal_judgement(standard_pos=stop_standard_pos, standard_rot=stop_standard_rot,
                                               current_pos=camrobot_hand_pos - init_error - cancel_pos_displacement,
                                               current_rot=np.dot(rm.rotmat_from_euler(ai=cancel_euler_displacement[0],
                                                                                       aj=cancel_euler_displacement[1],
                                                                                       ak=cancel_euler_displacement[2]),
                                                                                       camrobot_hand_rot),
                                               is_stable_cancel=False)
                    if judge == False:
                        abnormal_count = 0
                        abnormal_flag = False

        if abnormal_flag == False and current_jnt_values is not None:
            rbt_s.fk("arm", current_jnt_values)
            collided_result = rbt_s.is_collided()
            if collided_result == True:
                print('Collided! jnt_values is not updated!')
                rbt_s.fk("arm", pre_jnt_values)
            else:
                pre_jnt_values = current_jnt_values

    onscreen.append(rbt_s.gen_meshmodel())
    onscreen[-1].attach_to(base)

    # onscreen.append(gm.gen_frame(pos=init_operator_hand_pos, rotmat=init_operator_hand_rot))
    # onscreen[-1].attach_to(base)

    operation_count += 1

    return task.cont


if __name__ == '__main__':
    threading.Thread(target=data_collect, args=(data_list,)).start()
    taskMgr.add(operate_camrobot, "operate_camrobot", extraArgs=None, appendTask=True)
    base.run()
