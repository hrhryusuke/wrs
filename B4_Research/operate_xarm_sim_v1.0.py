"""
---operate_xarm_sim_v1.0概要---
Axis NEURONから送信されてきたPERCEPTION NEURONの右手の位置姿勢情報を元にシミュレータ上のxarmに動きを同期させる(1台のみ)
WASDキーで台車部分の操作が可能

---備考---

---更新日---
20220830
"""

import socket
import math
import time
import keyboard
import threading
import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import tkinter.messagebox as mb
import robot_sim.robots.xarm_shuidi.xarm_shuidi as xarm

from direct.task.TaskManagerGlobal import taskMgr

# -------------------------各種標準設定-------------------------
# 描画系統
base = wd.World(cam_pos=[3, 3, 2], lookat_pos=[0, 0, 1]) # モデリング空間を生成
rbt_s = xarm.XArmShuidi(enable_cc=True) # ロボットモデルの読込

# 通信系統
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((socket.gethostname(), 7001)) # 7001番のポートに接続
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
data_list = [None] # データ格納用リスト
pre_data_list = [None]
update_frequency = (1/60) # データの更新頻度(s)(ノード18個未満：120Hz 18個以上：60Hz)
main_data_length = 60*16*4
contactRL_length = 4

# -----その他の変数-----
data_array = []
onscreen = [] # ロボット描画情報用配列
onscreen_tcpframe = []

operation_count = 0 # 関数動作回数カウンタ
fail_IK_flag = 0 # IK解無しフラッグ

current_jnt_values_rgt = None
pre_jnt_values_rgt = None
pre_agv_pos_rgt = np.zeros(3)
modify_rot_for_pos = np.zeros((3, 3))
modify_rot = None
pre_tcp_pos = None
pre_sensor_pos_value = None
threshold_abnormal_jnt_values = 0.3
threshold_abnormal_jnt_values_norm = 0.3

adjustment_rotmat1 = rm.rotmat_from_euler(0, 135, 0) # 全変数調整用
adjustment_rotmat2 = rm.rotmat_from_euler(0, 0, 90) # グリッパ姿勢調整用

emergency_stop_flag = 0
stop_tic = 0
stop_toc = 0
stopped_flag = 0
is_data_receive = 0

rgt_hand_num = 11*16
hip_num = 21*16

operator_coordinate = np.eye(3)
adjusment_rotmat_hip = np.array([[0, 0, -1],
                                 [0, -1, 0],
                                 [-1, 0, 0]])
# --------------------


def gen_floor(x, y):
    ground = gm.gen_box(extent=np.array([edge_of_panel, edge_of_panel, .01]), homomat=rm.homomat_from_posrot(
        pos=np.array([x + edge_of_panel / 2, y + edge_of_panel / 2, -.01])))
    ground.set_rgba([.6, .6, .6, 1])
    ground.attach_to(base)
    ground = gm.gen_box(extent=np.array([edge_of_panel, edge_of_panel, .01]), homomat=rm.homomat_from_posrot(
        pos=np.array([x + edge_of_panel / 2 * 3, y + edge_of_panel / 2 * 3, -.01])))
    ground.set_rgba([.6, .6, .6, 1])
    ground.attach_to(base)
    ground = gm.gen_box(extent=np.array([edge_of_panel, edge_of_panel, .01]), homomat=rm.homomat_from_posrot(
        pos=np.array([x + edge_of_panel / 2 * 3, y + edge_of_panel / 2, -.01])))
    ground.set_rgba([.5, .5, .5, 1])
    ground.attach_to(base)
    ground = gm.gen_box(extent=np.array([edge_of_panel, edge_of_panel, .01]), homomat=rm.homomat_from_posrot(
        pos=np.array([x + edge_of_panel / 2, y + edge_of_panel / 2 * 3, -.01])))
    ground.set_rgba([.5, .5, .5, 1])
    ground.attach_to(base)


def gen_sim_floor(range_of_floor, edge_of_panel):
    for i in range(0, range_of_floor):
        for j in range(0, range_of_floor):
            gen_floor(-range_of_floor + i * (edge_of_panel * 2), -range_of_floor + j * (edge_of_panel * 2))


# モデル除去用関数
def model_remove(onscreen):
    for item in onscreen:
        item.detach()
    onscreen.clear()


# データ受信用関数
def data_collect(data_list):
    global is_data_receive, pre_data_list
    while True:
        if data_reception_flag == 0:
            # データ受信
            data_list[0] = s.recv(main_data_length)
            s.recv(2 * contactRL_length)  # contactLRの分を受信だけする（ズレ調整）
            pre_data_list = data_list
        else:
            data_list = pre_data_list


# 緊急停止用関数（コントローラからの入力により停止）
def program_stop(task):
    global emergency_stop_flag, stop_tic, stop_toc, stopped_flag, is_data_receive

    pressed_keys_stop = {'z': keyboard.is_pressed('z'),
                         'x': keyboard.is_pressed('x'),
                         'n': keyboard.is_pressed('n'),
                         'm': keyboard.is_pressed('m')}

    if (pressed_keys_stop["z"] and pressed_keys_stop["x"]) or (pressed_keys_stop["n"] and pressed_keys_stop["m"]):
        # print('operation check')
        if stop_flag == 0:
            stop_flag = 1
            stop_tic = time.time()
        else:
            stop_toc = time.time()
    else:
        stop_flag = 0

    if stop_toc - stop_tic >= 0.5 and stopped_flag == 0:
        mb.showwarning("データ受信停止", "モーションキャプチャからのデータ受信が停止されました．\n再度操作するにはプログラムを再起動してください．")
        print("Data Reception is Stopped!! Updating of jnt_values will not be done from now on.\n")
        data_reception_flag = 1
        stopped_flag = 1

    return task.cont


# データ調整用関数
def data_adjustment(data_list):
    global data_array, adjustment_rotmat1
    gripper_euler = np.zeros(3)

    # データ型の調整
    b_msg = data_list[0]
    if b_msg is None:
        return "No Data"
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


# 台車移動用関数
def agv_move(task):
    global onscreen, pre_agv_pos_rgt, current_jnt_values_rgt
    agv_speed_weight = .01
    agv_rotation_weight = .8

    model_remove(onscreen)
    agv_pos = rbt_s.get_jnt_values("agv")
    agv_loc_rotmat = rm.rotmat_from_axangle(axis=[0, 0, 1], angle=-1*agv_pos[2])
    agv_direction = np.dot(np.array([1, 0, 0]), agv_loc_rotmat)

    pressed_keys = {'w': keyboard.is_pressed('w'),
                    'a': keyboard.is_pressed('a'),
                    's': keyboard.is_pressed('s'),
                    'd': keyboard.is_pressed('d'),
                    'o': keyboard.is_pressed('o'),  # gripper open
                    'p': keyboard.is_pressed('p')}  # gripper close
    values_list = list(pressed_keys.values())

    if pressed_keys["w"] and pressed_keys["a"]:
        current_jnt_values = np.array(pre_pos + [agv_speed_weight*agv_direction[0], agv_speed_weight*agv_direction[1], math.radians(agv_rotation_weight)])
        rbt_s.fk(component_name='agv', jnt_values=current_jnt_values)
        pre_pos = current_jnt_values
    elif pressed_keys["w"] and pressed_keys["d"]:
        current_jnt_values = np.array(pre_pos + [agv_speed_weight * agv_direction[0], agv_speed_weight * agv_direction[1], math.radians(-agv_rotation_weight)])
        rbt_s.fk(component_name='agv', jnt_values=current_jnt_values)
        pre_pos = current_jnt_values
    elif pressed_keys["s"] and pressed_keys["a"]:
        current_jnt_values = np.array(pre_pos + [-agv_speed_weight * agv_direction[0], -agv_speed_weight * agv_direction[1], math.radians(-agv_rotation_weight)])
        rbt_s.fk(component_name='agv', jnt_values=current_jnt_values)
        pre_pos = current_jnt_values
    elif pressed_keys["s"] and pressed_keys["d"]:
        current_jnt_values = np.array(pre_pos + [-agv_speed_weight * agv_direction[0], -agv_speed_weight * agv_direction[1], math.radians(agv_rotation_weight)])
        rbt_s.fk(component_name='agv', jnt_values=current_jnt_values)
        pre_pos = current_jnt_values
    elif pressed_keys["w"] and sum(values_list) == 1:
        current_jnt_values = np.array(pre_pos + [agv_speed_weight * agv_direction[0], agv_speed_weight * agv_direction[1], math.radians(0)])
        rbt_s.fk(component_name='agv', jnt_values=current_jnt_values)
        pre_pos = current_jnt_values
    elif pressed_keys["s"] and sum(values_list) == 1:
        current_jnt_values = np.array(pre_pos + [-agv_speed_weight * agv_direction[0], -agv_speed_weight * agv_direction[1], math.radians(0)])
        rbt_s.fk(component_name='agv', jnt_values=current_jnt_values)
        pre_pos = current_jnt_values
    elif pressed_keys["a"] and sum(values_list) == 1:
        current_jnt_values = np.array(pre_pos + [0, 0, math.radians(agv_rotation_weight)])
        rbt_s.fk(component_name='agv', jnt_values=current_jnt_values)
        pre_pos = current_jnt_values
    elif pressed_keys["d"] and sum(values_list) == 1:
        current_jnt_values = np.array(pre_pos + [0, 0, math.radians(-agv_rotation_weight)])
        rbt_s.fk(component_name='agv', jnt_values=current_jnt_values)
        pre_pos = current_jnt_values
    elif pressed_keys["o"] and sum(values_list) == 1:
        rbt_s.jaw_to(jawwidth=.085)
    elif pressed_keys["p"] and sum(values_list) == 1:
        rbt_s.jaw_to(jawwidth=0)

    onscreen.append(rbt_s.gen_meshmodel())
    onscreen[-1].attach_to(base)

    return task.cont


abnormal_flag = False
abnormal_count = 0
abnormal_cancel_count = 0
pre_judge_pos = np.zeros(3)
pre_judge_rot = np.zeros((3, 3))

stop_standard_pos_rgt = np.zeros(3)
stop_standard_rot_rgt = np.zeros((3, 3))

cancel_pos_displacement_rgt = np.zeros(3)
cancel_euler_displacement_rgt = np.zeros(3)
cancel_rot_displacement = np.eye(3)
def abnormal_judgement(standard_pos, standard_rot, current_pos, current_rot, is_stable_cancel):
    global abnormal_count, abnormal_cancel_count, pre_judge_pos, pre_judge_rot, cancel_pos_displacement_rgt, cancel_rot_displacement, cancel_euler_displacement_rgt
    threshold_pos = 0.05
    threshold_rot_euler = 0.5

    # 動作停止時の位置姿勢を
    if abnormal_count == 1:
        pre_judge_pos = current_pos
        pre_judge_rot = current_rot

    pos_judge = (abs(current_pos - standard_pos) <= threshold_pos).all()
    rot_judge = np.linalg.norm(rm.rotmat_to_euler(current_rot) - rm.rotmat_to_euler(standard_rot), ord=2) <= threshold_rot_euler

    if pos_judge and rot_judge:
        # マニピュレータの手先の位置姿勢と操作者の手先の位置姿勢の比較
        # 動作復帰条件クリア
        abnormal_cancel_count = 0
        print('operating is restarted!!')
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
        print(f"displacement[pos]{abs(current_pos - standard_pos)}, [euler_norm]{np.linalg.norm(rm.rotmat_to_euler(current_rot) - rm.rotmat_to_euler(standard_rot), ord=2)}")
        print(f"judgement[pos]{pos_judge}, [euler_norm]{rot_judge}")
        return True


init_error_rgt = np.zeros(3)
pre_sensor_quarternion = np.ones(4)
pre_tcp_rot = np.zeros((3, 3))
init_tcp_rot = None
init_hand_rot = np.zeros((3, 3))
# アーム動作用関数
def arm_move(rbt_s, onscreen, task):
    global data_array, current_jnt_values_rgt, pre_jnt_values_rgt, fail_IK_flag, operation_count, modify_rot, modify_rot_for_pos, pre_tcp_pos, pre_sensor_pos_value, operator_coordinate, \
        init_error_rgt, pre_sensor_quarternion, pre_tcp_rot, init_tcp_rot, init_hand_rot, abnormal_flag, stop_standard_pos_rgt, stop_standard_rot_rgt, abnormal_count, cancel_pos_displacement_rgt, cancel_rot_displacement

    # agvの情報を取得（手先の位置姿勢の反映に使用）
    agv_pos_and_angle = rbt_s.get_jnt_values("agv")
    agv_pos = np.array([agv_pos_and_angle[0], agv_pos_and_angle[1], 0])

    # データ受信が出来ているかの判定
    if data_adjustment(data_list) == "No Data":
        return task.cont
    else:
        data_array = data_adjustment(data_list)

    # リアルタイム描画のため、各時刻のモデルを除去
    model_remove(onscreen)
    model_remove(onscreen_tcpframe)

    # 操作者座標系の定義
    if operation_count == 0:
        b_msg = data_list[0]
        if b_msg is None:
            return "No Data"
        tmp_data_array = np.frombuffer(b_msg, dtype=np.float32)
        tmp_euler = rm.rotmat_to_euler(np.dot(adjusment_rotmat_hip, rm.rotmat_from_quaternion(tmp_data_array[hip_num+6: hip_num+10])[:3, :3]))
        operator_coordinate_euler = np.zeros(3)
        operator_coordinate_euler[0] = tmp_euler[2]
        operator_coordinate_euler[1] = -tmp_euler[1]
        operator_coordinate_euler[2] = tmp_euler[0]
        operator_coordinate[:2, :2] = rm.rotmat_from_euler(ai=operator_coordinate_euler[0], aj=operator_coordinate_euler[1], ak=operator_coordinate_euler[2])[:2, :2]
        operator_coordinate_frame_color = np.array([[0, 1, 1],
                                                    [1, 0, 1],
                                                    [1, 1, 0]])
        gm.gen_frame(pos=np.zeros(3), rotmat=operator_coordinate, length=2, thickness=0.03, rgbmatrix=operator_coordinate_frame_color).attach_to(base)

    if operation_count == 0:
        tcp_pos, tcp_rot = rbt_s.get_gl_tcp()

        init_error = np.dot(np.linalg.pinv(operator_coordinate), data_array[(rgt_hand_num):(rgt_hand_num + 3)]) - tcp_pos
        # init_tcp_rot = tcp_rot

        pre_jnt_values = rbt_s.get_jnt_values(component_name="arm")
        pre_tcp_pos = tcp_pos
        # pre_sensor_pos_value = data_array[rgt_hand_num:(rgt_hand_num+3)]
        pre_tcp_rot = tcp_rot

        pre_sensor_quarternion = data_array[(rgt_hand_num + 6):(rgt_hand_num + 10)]
        # init_hand_rot = rm.rotmat_from_quaternion(pre_sensor_quarternion)[:3, :3]
    else:
        tcp_pos, tcp_rot = rbt_s.manipulator_dict['arm'].jlc.get_gl_tcp()
        # マニピュレータの手先の座標系表示
        onscreen_tcpframe.append(gm.gen_frame(pos=tcp_pos, rotmat=tcp_rot, length=0.3))
        onscreen_tcpframe[-1].attach_to(base)

        # 操作者座標系から見た右手の位置姿勢についてLM法で逆運動学の解を導出
        pre_sensor_quarternion = data_array[(rgt_hand_num + 6):(rgt_hand_num + 10)]
        # rel_world_euler = rm.quaternion_to_euler(data_array[(rgt_hand_num + 6):(rgt_hand_num + 10)]) - rm.quaternion_to_euler(pre_sensor_quarternion)
        # rel_tcp_rot = np.dot(np.dot(np.linalg.pinv(operator_coordinate), init_hand_rot), rm.rotmat_from_euler(ai=rel_world_euler[0], aj=rel_world_euler[1], ak=rel_world_euler[2]))
        operator_hand_pos = np.dot(np.linalg.pinv(operator_coordinate), data_array[(rgt_hand_num):(rgt_hand_num + 3)])
        operator_hand_rot = np.dot(np.linalg.pinv(operator_coordinate), rm.rotmat_from_quaternion(data_array[(rgt_hand_num+6):(rgt_hand_num+10)])[:3, :3])

        current_jnt_values = rbt_s.manipulator_dict['arm'].jlc._ikt.num_ik(tgt_pos=operator_hand_pos - init_error + agv_pos - cancel_pos_displacement,
                                                                           tgt_rot=np.dot(rm.rotmat_from_euler(ai = cancel_euler_displacement_rgt[0], aj = cancel_euler_displacement_rgt[1], ak = cancel_euler_displacement_rgt[2]),
                                                                                          operator_hand_rot),
                                                                           seed_jnt_values=pre_jnt_values,
                                                                           tcp_jnt_id=7)

        # 関節角度の更新と衝突判定
        if abnormal_flag == False:
            # 異常動作判定
            if not (np.all(abs(current_jnt_values - pre_jnt_values) <= threshold_abnormal_jnt_values) and
                    abs(np.linalg.norm(current_jnt_values - pre_jnt_values, ord=2)) <= threshold_abnormal_jnt_values_norm):
                if operation_count >= 10:
                    abnormal_flag = True
        else:
            # 操作復帰判定
            abnormal_count += 1
            if abnormal_count == 1:
                # 動作停止時のマニピュレータの手先の位置姿勢の記録
                standard_pos, standard_rot = rbt_s.get_gl_tcp()
                print("abnormal operation is detected!")
            else:
                operator_hand_pos = np.dot(np.linalg.pinv(operator_coordinate), data_array[(rgt_hand_num):(rgt_hand_num + 3)])
                operator_hand_rot = np.dot(np.linalg.pinv(operator_coordinate), rm.rotmat_from_quaternion(data_array[(rgt_hand_num + 6):(rgt_hand_num + 10)])[:3, :3])

                judge = abnormal_judgement(standard_pos=standard_pos, standard_rot=standard_rot,
                                           current_pos=operator_hand_pos - init_error + agv_pos - cancel_pos_displacement,
                                           current_rot=np.dot(rm.rotmat_from_euler(ai=cancel_euler_displacement_rgt[0],
                                                                                   aj=cancel_euler_displacement_rgt[1],
                                                                                   ak=cancel_euler_displacement_rgt[2]),
                                                              operator_hand_rot),
                                           is_stable_cancel=False)
                if judge == False:
                    abnormal_count = 0
                    abnormal_flag = False

        if abnormal_flag == False:
            rbt_s.fk("arm", current_jnt_values)
            collided_result = rbt_s.is_collided()
            if collided_result == True:
                print('Collided! jnt_values is not updated!')
                rbt_s.fk("arm", pre_jnt_values)
            else:
                pre_jnt_values = current_jnt_values

        # pre_jnt_values = current_jnt_values

        onscreen.append(rbt_s.gen_meshmodel())
        onscreen[-1].attach_to(base)

    operation_count = operation_count + 1
    return task.cont


if __name__ == '__main__':
    is_gen_floor = False
    gm.gen_frame(length=10, thickness=0.02).attach_to(base)

    if is_gen_floor == True:
        edge_of_panel = 1
        range_of_floor = 10
        gen_sim_floor(range_of_floor, edge_of_panel)

    # シミュレータ内のロボットに指定の初期姿勢を反映
    sim_init_pos, sim_init_rot = rbt_s.get_gl_tcp()
    sim_init_pos[2] = sim_init_pos[2] - 0.2 # 高さを変えないとikの解が導出不可
    specified_init_pos = sim_init_pos
    specified_init_rot = np.array([[0, -1, 0],
                                   [-1, 0, 0],
                                   [0, 0, -1]])
    specified_init_jnt_values = rbt_s.ik(tgt_pos=specified_init_pos, tgt_rotmat=specified_init_rot)
    rbt_s.fk(jnt_values=specified_init_jnt_values)

    # 各関数を実行
    threading.Thread(target=data_collect, args=(data_list,)).start()
    taskMgr.doMethodLater(update_frequency, program_stop, "program_stop",
                          extraArgs=None,
                          appendTask=True)
    taskMgr.doMethodLater(update_frequency, agv_move, "agv_move",
                          extraArgs=None,
                          appendTask=True)
    taskMgr.doMethodLater(update_frequency, arm_move, "arm_move",
                          extraArgs=[rbt_s, onscreen],
                          appendTask=True)
    base.run()

