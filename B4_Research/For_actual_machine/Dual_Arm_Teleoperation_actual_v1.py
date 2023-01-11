"""
---Dual_Arm_Teleoperation_actual_v1概要説明---
Axis NEURONから送信されてきたPERCEPTION NEURONの右手の位置姿勢情報を元にシミュレータ上の2台のxarmに動きを同期させる
キー入力で左右のxarmを独立に台車移動およびグリッパの開閉が可能　[移動]（左：WASDキー，右：IJKLキー） [グリッパ開閉]（左：ZXキー，右：NMキー）

---備考---
事前にgenerate_actual_init_jnt_valuesから初期の関節角度を生成する必要あり

---BASE PROGRAM---
robot_actual_v9
"""
import socket
import math
import time
import keyboard
import basis.trimesh.transformations as tr
import threading
import numpy as np
import tkinter.messagebox as mb
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import robot_sim.robots.xarm7_shuidi_mobile.xarm7_shuidi_mobile as xarm
import robot_con.xarm_shuidi.xarm_shuidi_client as rbx

from direct.task.TaskManagerGlobal import taskMgr

# -------------------------各種標準設定-------------------------
# 描画系統
rbt_s_rgt = xarm.XArm7YunjiMobile(enable_cc=True) # シミュレータモデル
rbt_s_lft = xarm.XArm7YunjiMobile(enable_cc=True)

rbt_x_rgt = rbx.XArmShuidiClient(host="10.2.0.201:18300") # 実機の呼び出し
rbt_x_lft = rbx.XArmShuidiClient(host="10.2.0.203:18300")
# init_jnt_angles = rbt_x.get_jnt_vlaues()
# rbt_s.fk("arm", init_jnt_angles)
base = wd.World(cam_pos=[-5, 0, 1.5], lookat_pos=[0, 0, 1]) # モデリング空間を生成
# gm.gen_frame().attach_to(base) # 座標系の軸を表示

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

# ロボット初期位置
init_pos_rgt = np.array([0, -.35, 0])
init_pos_lft = np.array([0, .35, 0])

# センサ番号
rgt_hand_num = 11*16
lft_hand_num = 15*16

# グリッパ姿勢調整用回転行列
adjustment_rotmat1 = rm.rotmat_from_euler(0, 135, 0) # 全変数調整用
adjustment_rotmat2 = rm.rotmat_from_euler(0, 0, 90) # グリッパ姿勢調整用

# xarmのデフォルトの初期姿勢
arm_homeconf = np.zeros(7)
arm_homeconf[1] = -math.pi / 3
arm_homeconf[3] = math.pi / 12
arm_homeconf[5] = -math.pi / 12

# 初期姿勢関節角度
init_jnt_values_rgt = np.loadtxt('init_jnt_values_rgt.txt')
init_jnt_values_lft = np.loadtxt('init_jnt_values_lft.txt')
# ------------------------------------------------------------

# -----その他の変数-----
# 描画用
data_array = []
onscreen = [] # ロボット描画情報用配列
onscreen_tcpframe_rgt = []
onscreen_lft = []
onscreen_tcpframe_lft = []

# 制御用
current_jnt_values_rgt = None
pre_jnt_values_rgt = None
current_agv_pos_rgt = np.zeros(3)
pre_agv_pos_rgt = init_pos_rgt
modify_rot_for_pos_rgt = np.zeros((3, 3))
modify_rot_rgt = None
pre_tcp_pos_rgt = None
pre_sensor_pos_value_rgt = None

current_jnt_values_lft = None
pre_jnt_values_lft = None
current_agv_pos_lft = np.zeros(3)
pre_agv_pos_lft = init_pos_lft
modify_rot_for_pos_lft = np.zeros((3, 3))
modify_rot_lft = None
pre_tcp_pos_lft = None
pre_sensor_pos_value_lft = None

# その他
operation_count = 0 # 関数動作回数カウンタ
fail_IK_flag = 0 # IK解無しフラッグ
emergency_stop_flag = 0
stop_tic = 0
stop_toc = 0
stop_threshold = 0.5
is_data_receive = 0
threshold_jnt_values_displacement_norm = 0.2
# --------------------
operator_coordinate = np.eye(3)
adjusment_rotmat_hip = np.array([[0, 0, -1],
                                 [0, -1, 0],
                                 [-1, 0, 0]])
hip_num = 21*16
abnormal_flag = 0
abnormal_count = 0

# シミュレータ内モデル除去用関数
def model_remove(onscreen):
    for item in onscreen:
        item.detach()
    onscreen.clear()

# データ受信用関数
def data_collect(data_list):
    global is_data_receive
    while True:
        if data_reception_flag == 0:
            # データ受信
            data_list[0] = s.recv(main_data_length)
            s.recv(2 * contactRL_length)  # contactLRの分を受信だけする（ズレ調整）

abnormal_cancel_count = 0
pre_judge_pos = np.zeros(3)
pre_judge_rot = np.zeros((3, 3))

cancel_pos_displacement_rgt = np.zeros(3)
cancel_rot_displacement = np.eye(3)
def abnormal_judgement(standard_pos, standard_rot, current_pos, current_rot):
    global abnormal_count, abnormal_cancel_count, pre_judge_pos, pre_judge_rot, cancel_pos_displacement_rgt, cancel_rot_displacement
    threshold_pos = 5
    threshold_rot_euler = 0.05

    if abnormal_count == 0:
        # print('operation check1')
        pre_judge_pos = standard_pos
        pre_judge_rot = standard_rot

    if np.all(abs(current_pos-standard_pos) <= threshold_pos) and np.linalg.norm(rm.rotmat_to_euler(current_rot)-rm.rotmat_to_euler(standard_rot), ord=2) <= threshold_rot_euler:
        abnormal_cancel_count = 0
        return 0
    else:
        if np.linalg.norm(current_pos - pre_judge_pos) <= 1:
            abnormal_cancel_count += 1
            pre_judge_pos = current_pos
            # print('operation check2')
        else:
            abnormal_cancel_count = 0

        if abnormal_cancel_count <= 30:
            print(f'{abs(current_pos-standard_pos)},{rm.rotmat_to_euler(current_rot)-rm.rotmat_to_euler(standard_rot)}')
            return 1
        else:
            print('abnormal_cancel!!')
            cancel_pos_displacement = current_pos - standard_pos
            print(f'cancel_pos_displacement:{cancel_pos_displacement}')
            # cancel_rot_displacement = rm.rotmat_from_euler(rm.rotmat_to_euler(current_rot)-rm.rotmat_to_euler(standard_rot))
            abnormal_cancel_count = 0
            return 0

# マニピュレータ追従動作停止用関数
def program_stop(task):
    global emergency_stop_flag, stop_tic, stop_toc, stop_threshold, is_data_receive

    pressed_keys_stop = {'z': keyboard.is_pressed('z'),
                         'x': keyboard.is_pressed('x'),
                         'n': keyboard.is_pressed('n'),
                         'm': keyboard.is_pressed('m')}

    if (pressed_keys_stop["z"] and pressed_keys_stop["x"]) or (pressed_keys_stop["n"] and pressed_keys_stop["m"]):
        if stop_flag == 0:
            stop_flag = 1
            stop_tic = time.time()
        else:
            stop_toc = time.time()
    else:
        stop_flag = 0

    if stop_toc - stop_tic >= stop_threshold:
        mb.showwarning("データ受信停止", "モーションキャプチャからのデータ受信が停止されました．\n再度操作するにはプログラムを再起動してください．")
        print("Data Reception is Stopped!! Updating of jnt_values will not be done from now on.\n")
        data_reception_flag = 1

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
    global onscreen, pre_agv_pos_rgt, current_agv_pos_rgt, onscreen_lft, pre_agv_pos_lft, current_agv_pos_lft
    agv_speed_weight = .01
    agv_rotation_weight = .8
    agv_linear_speed = .2
    agv_angular_speed = .5
    arm_linear_speed = .03
    arm_angular_speed = .1

    # model_remove(onscreen_rgt)
    # model_remove(onscreen_lft)

    agv_pos_rgt = rbt_s_rgt.get_jnt_values("agv")
    agv_loc_rotmat_rgt = rm.rotmat_from_axangle(axis=[0, 0, 1], angle=-1*agv_pos_rgt[2])
    agv_direction_rgt = np.dot(np.array([1, 0, 0]), agv_loc_rotmat_rgt)
    agv_pos_lft = rbt_s_lft.get_jnt_values("agv")
    agv_loc_rotmat_lft = rm.rotmat_from_axangle(axis=[0, 0, 1], angle=-1 * agv_pos_lft[2])
    agv_direction_lft = np.dot(np.array([1, 0, 0]), agv_loc_rotmat_lft)

    pressed_keys_rgt = {'i': keyboard.is_pressed('i'),
                        'j': keyboard.is_pressed('j'),
                        'k': keyboard.is_pressed('k'),
                        'l': keyboard.is_pressed('l'),
                        'n': keyboard.is_pressed('n'),  # gripper open
                        'm': keyboard.is_pressed('m')}  # gripper close
    values_list_rgt = list(pressed_keys_rgt.values())
    pressed_keys_lft = {'w': keyboard.is_pressed('w'),
                        'a': keyboard.is_pressed('a'),
                        's': keyboard.is_pressed('s'),
                        'd': keyboard.is_pressed('d'),
                        'z': keyboard.is_pressed('z'),  # gripper open
                        'x': keyboard.is_pressed('x')}  # gripper close
    values_list_lft = list(pressed_keys_lft.values())

    # left_arm
    if pressed_keys_lft["w"] and pressed_keys_lft["a"]:
        rbt_x_rgt.agv_move(linear_speed=agv_linear_speed, angular_speed=agv_angular_speed, time_interval=.5)
        # current_agv_pos_lft = np.array(pre_agv_pos_lft + [agv_speed_weight*agv_direction_lft[0], agv_speed_weight*agv_direction_lft[1], math.radians(agv_rotation_weight)])
        # rbt_s_lft.fk(component_name='agv', jnt_values=current_agv_pos_lft)
        # pre_agv_pos_lft = current_agv_pos_lft
    elif pressed_keys_lft["w"] and pressed_keys_lft["d"]:
        rbt_x_rgt.agv_move(linear_speed=agv_linear_speed, angular_speed=-agv_angular_speed, time_interval=.5)
        # current_agv_pos_lft = np.array(pre_agv_pos_lft + [agv_speed_weight * agv_direction_lft[0], agv_speed_weight * agv_direction_lft[1], math.radians(-agv_rotation_weight)])
        # rbt_s_lft.fk(component_name='agv', jnt_values=current_agv_pos_lft)
        # pre_agv_pos_lft = current_agv_pos_lft
    elif pressed_keys_lft["s"] and pressed_keys_lft["a"]:
        rbt_x_rgt.agv_move(linear_speed=-agv_linear_speed, angular_speed=-agv_angular_speed, time_interval=.5)
        # current_agv_pos_lft = np.array(pre_agv_pos_lft + [-agv_speed_weight * agv_direction_lft[0], -agv_speed_weight * agv_direction_lft[1], math.radians(-agv_rotation_weight)])
        # rbt_s_lft.fk(component_name='agv', jnt_values=current_agv_pos_lft)
        # pre_agv_pos_lft = current_agv_pos_lft
    elif pressed_keys_lft["s"] and pressed_keys_lft["d"]:
        rbt_x_rgt.agv_move(linear_speed=-agv_linear_speed, angular_speed=agv_angular_speed, time_interval=.5)
        # current_agv_pos_lft = np.array(pre_agv_pos_lft + [-agv_speed_weight * agv_direction_lft[0], -agv_speed_weight * agv_direction_lft[1], math.radians(agv_rotation_weight)])
        # rbt_s_lft.fk(component_name='agv', jnt_values=current_agv_pos_lft)
        # pre_agv_pos_lft = current_agv_pos_lft
    elif pressed_keys_lft["w"] and sum(values_list_lft) == 1:
        rbt_x_rgt.agv_move(linear_speed=agv_linear_speed, angular_speed=0, time_interval=.5)
        # current_agv_pos_lft = np.array(pre_agv_pos_lft + [agv_speed_weight * agv_direction_lft[0], agv_speed_weight * agv_direction_lft[1], math.radians(0)])
        # rbt_s_lft.fk(component_name='agv', jnt_values=current_agv_pos_lft)
        # pre_agv_pos_lft = current_agv_pos_lft
    elif pressed_keys_lft["s"] and sum(values_list_lft) == 1:
        rbt_x_rgt.agv_move(linear_speed=-agv_linear_speed, angular_speed=0, time_interval=.5)
        # current_agv_pos_lft = np.array(pre_agv_pos_lft + [-agv_speed_weight * agv_direction_lft[0], -agv_speed_weight * agv_direction_lft[1], math.radians(0)])
        # rbt_s_lft.fk(component_name='agv', jnt_values=current_agv_pos_lft)
        # pre_agv_pos_lft = current_agv_pos_lft
    elif pressed_keys_lft["a"] and sum(values_list_lft) == 1:
        rbt_x_rgt.agv_move(linear_speed=0, angular_speed=agv_angular_speed, time_interval=.5)
        # current_agv_pos_lft = np.array(pre_agv_pos_lft + [0, 0, math.radians(agv_rotation_weight)])
        # rbt_s_lft.fk(component_name='agv', jnt_values=current_agv_pos_lft)
        # pre_agv_pos_lft = current_agv_pos_lft
    elif pressed_keys_lft["d"] and sum(values_list_lft) == 1:
        rbt_x_rgt.agv_move(linear_speed=0, angular_speed=-agv_angular_speed, time_interval=.5)
        # current_agv_pos_lft = np.array(pre_agv_pos_lft + [0, 0, math.radians(-agv_rotation_weight)])
        # rbt_s_lft.fk(component_name='agv', jnt_values=current_agv_pos_lft)
        # pre_agv_pos_lft = current_agv_pos_lft
    elif pressed_keys_lft["z"] and sum(values_list_lft) == 1:
        rbt_x_rgt.arm_jaw_to(jawwidth=100)
        # rbt_s_lft.jaw_to(jawwidth=.085)
    elif pressed_keys_lft["x"] and sum(values_list_lft) == 1:
        rbt_x_rgt.arm_jaw_to(jawwidth=0)
        # rbt_s_lft.jaw_to(jawwidth=0)

    # right_arm
    if pressed_keys_rgt["i"] and pressed_keys_rgt["j"]:
        rbt_x_rgt.agv_move(linear_speed=agv_linear_speed, angular_speed=agv_angular_speed, time_interval=.5)
        # current_agv_pos_rgt = np.array(pre_agv_pos_rgt + [agv_speed_weight*agv_direction_rgt[0], agv_speed_weight*agv_direction_rgt[1], math.radians(agv_rotation_weight)])
        # rbt_s_rgt.fk(component_name='agv', jnt_values=current_agv_pos_rgt)
        # pre_agv_pos_rgt = current_agv_pos_rgt
    elif pressed_keys_rgt["i"] and pressed_keys_rgt["l"]:
        rbt_x_rgt.agv_move(linear_speed=agv_linear_speed, angular_speed=-agv_angular_speed, time_interval=.5)
        # current_agv_pos_rgt = np.array(pre_agv_pos_rgt + [agv_speed_weight * agv_direction_rgt[0], agv_speed_weight * agv_direction_rgt[1], math.radians(-agv_rotation_weight)])
        # rbt_s_rgt.fk(component_name='agv', jnt_values=current_agv_pos_rgt)
        # pre_agv_pos_rgt = current_agv_pos_rgt
    elif pressed_keys_rgt["k"] and pressed_keys_rgt["j"]:
        rbt_x_rgt.agv_move(linear_speed=-agv_linear_speed, angular_speed=-agv_angular_speed, time_interval=.5)
        # current_agv_pos_rgt = np.array(pre_agv_pos_rgt + [-agv_speed_weight * agv_direction_rgt[0], -agv_speed_weight * agv_direction_rgt[1], math.radians(-agv_rotation_weight)])
        # rbt_s_rgt.fk(component_name='agv', jnt_values=current_agv_pos_rgt)
        # pre_agv_pos_rgt = current_agv_pos_rgt
    elif pressed_keys_rgt["k"] and pressed_keys_rgt["l"]:
        rbt_x_rgt.agv_move(linear_speed=-agv_linear_speed, angular_speed=agv_angular_speed, time_interval=.5)
        # current_agv_pos_rgt = np.array(pre_agv_pos_rgt + [-agv_speed_weight * agv_direction_rgt[0], -agv_speed_weight * agv_direction_rgt[1], math.radians(agv_rotation_weight)])
        # rbt_s_rgt.fk(component_name='agv', jnt_values=current_agv_pos_rgt)
        # pre_agv_pos_rgt = current_agv_pos_rgt
    elif pressed_keys_rgt["i"] and sum(values_list_rgt) == 1:
        rbt_x_rgt.agv_move(linear_speed=agv_linear_speed, angular_speed=0, time_interval=.5)
        # current_agv_pos_rgt = np.array(pre_agv_pos_rgt + [agv_speed_weight * agv_direction_rgt[0], agv_speed_weight * agv_direction_rgt[1], math.radians(0)])
        # rbt_s_rgt.fk(component_name='agv', jnt_values=current_agv_pos_rgt)
        # pre_agv_pos_rgt = current_agv_pos_rgt
    elif pressed_keys_rgt["k"] and sum(values_list_rgt) == 1:
        rbt_x_rgt.agv_move(linear_speed=-agv_linear_speed, angular_speed=0, time_interval=.5)
        # current_agv_pos_rgt = np.array(pre_agv_pos_rgt + [-agv_speed_weight * agv_direction_rgt[0], -agv_speed_weight * agv_direction_rgt[1], math.radians(0)])
        # rbt_s_rgt.fk(component_name='agv', jnt_values=current_agv_pos_rgt)
        # pre_agv_pos_rgt = current_agv_pos_rgt
    elif pressed_keys_rgt["j"] and sum(values_list_rgt) == 1:
        rbt_x_rgt.agv_move(linear_speed=0, angular_speed=agv_angular_speed, time_interval=.5)
        # current_agv_pos_rgt = np.array(pre_agv_pos_rgt + [0, 0, math.radians(agv_rotation_weight)])
        # rbt_s_rgt.fk(component_name='agv', jnt_values=current_agv_pos_rgt)
        # pre_agv_pos_rgt = current_agv_pos_rgt
    elif pressed_keys_rgt["l"] and sum(values_list_rgt) == 1:
        rbt_x_rgt.agv_move(linear_speed=0, angular_speed=-agv_angular_speed, time_interval=.5)
        # current_agv_pos_rgt = np.array(pre_agv_pos_rgt + [0, 0, math.radians(-agv_rotation_weight)])
        # rbt_s_rgt.fk(component_name='agv', jnt_values=current_agv_pos_rgt)
        # pre_agv_pos_rgt = current_agv_pos_rgt
    elif pressed_keys_rgt["n"] and sum(values_list_rgt) == 1:
        rbt_x_rgt.arm_jaw_to(jawwidth=100)
        # rbt_s_rgt.jaw_to(jawwidth=.085)
    elif pressed_keys_rgt["m"] and sum(values_list_rgt) == 1:
        rbt_x_rgt.arm_jaw_to(jawwidth=0)
        # rbt_s_rgt.jaw_to(jawwidth=0)

    # onscreen_rgt.append(rbt_s_rgt.gen_meshmodel())
    # onscreen_rgt[-1].attach_to(base)
    # onscreen_lft.append(rbt_s_lft.gen_meshmodel())
    # onscreen_lft[-1].attach_to(base)

    return task.cont


init_error_rgt = np.zeros(3)
init_error_lft = np.zeros(3)
pre_sensor_quarternion = np.ones(4)
pre_tcp_rot = np.zeros((3, 3))
init_tcp_rot = None
init_hand_rot = np.zeros((3, 3))

standard_pos_rgt = np.zeros(3)
standard_rot_rgt = np.zeros((3, 3))
standard_pos_lft = np.zeros(3)
standard_rot_lft = np.zeros((3, 3))
# アーム動作用関数
def arm_move(rbt_s_rgt, rbt_s_lft, onscreen_rgt, onscreen_lft, task):
    global data_array, fail_IK_flag, operation_count, init_error_rgt, init_error_lft, abnormal_flag, abnormal_count, \
        current_jnt_values_rgt, pre_jnt_values_rgt, modify_rot_rgt, pre_tcp_pos_rgt, pre_sensor_pos_value_rgt, \
        current_jnt_values_lft, pre_jnt_values_lft, modify_rot_lft, pre_tcp_pos_lft, pre_sensor_pos_value_lft, \
        standard_pos_rgt, standard_rot_rgt, standard_pos_lft, standard_rot_lft

    agv_pos_and_angle_rgt = rbt_s_rgt.get_jnt_values("agv")
    agv_pos_rgt = np.array([agv_pos_and_angle_rgt[0], agv_pos_and_angle_rgt[1], 0])
    agv_direction_rot_rgt = rm.rotmat_from_axangle(axis=[0, 0, 1], angle=-1 * agv_pos_rgt[2])
    agv_pos_and_angle_lft = rbt_s_lft.get_jnt_values("agv")
    agv_pos_lft = np.array([agv_pos_and_angle_lft[0], agv_pos_and_angle_lft[1], 0])
    agv_direction_rot_lft = rm.rotmat_from_axangle(axis=[0, 0, 1], angle=-1 * agv_pos_lft[2])

    model_remove(onscreen_rgt)
    model_remove(onscreen_tcpframe_rgt)
    model_remove(onscreen_lft)
    model_remove(onscreen_tcpframe_lft)

    # --------------------------------------------------------
    if operation_count == 0:
        b_msg = data_list[0]
        if b_msg is None:
            return "No Data"
        tmp_data_array = np.frombuffer(b_msg, dtype=np.float32)
        tmp_euler = rm.rotmat_to_euler(np.dot(adjusment_rotmat_hip, rm.rotmat_from_quaternion(tmp_data_array[hip_num + 6: hip_num + 10])[:3, :3]))
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
    # --------------------------------------------------------

    if data_adjustment(data_list) == "No Data":
        return task.cont
    else:
        data_array = data_adjustment(data_list)

    if operation_count == 0:
        # right_arm
        tcp_pos_rgt, tcp_rot_rgt = rbt_s_rgt.manipulator_dict['arm'].jlc.get_gl_tcp()
        modify_rot_rgt = np.dot(tcp_rot_rgt, np.linalg.pinv(rm.rotmat_from_quaternion(data_array[(rgt_hand_num+6):(rgt_hand_num+10)])[:3, :3]))
        modify_rot_for_pos_rgt[:2, :2] = modify_rot_rgt[:2, :2]
        modify_rot_for_pos_rgt[2, 2] = 1
        pre_jnt_values_rgt = rbt_s_rgt.get_jnt_values(component_name="arm")
        pre_tcp_pos_rgt = tcp_pos_rgt
        pre_sensor_pos_value_rgt = data_array[rgt_hand_num:(rgt_hand_num+3)]
        init_error_rgt = np.dot(np.linalg.pinv(operator_coordinate), data_array[(rgt_hand_num):(rgt_hand_num + 3)]) - tcp_pos_rgt
        # left_arm
        tcp_pos_lft, tcp_rot_lft = rbt_s_lft.manipulator_dict['arm'].jlc.get_gl_tcp()
        modify_rot_lft = np.dot(tcp_rot_lft, np.linalg.pinv(rm.rotmat_from_quaternion(data_array[(lft_hand_num+6):(lft_hand_num+10)])[:3, :3]))
        modify_rot_for_pos_lft[:2, :2] = modify_rot_lft[:2, :2]
        modify_rot_for_pos_lft[2, 2] = 1
        pre_jnt_values_lft = rbt_s_lft.get_jnt_values(component_name="arm")
        pre_tcp_pos_lft = tcp_pos_lft
        pre_sensor_pos_value_lft = data_array[lft_hand_num:(lft_hand_num+3)]
        init_error_lft = np.dot(np.linalg.pinv(operator_coordinate), data_array[(lft_hand_num):(lft_hand_num + 3)]) - tcp_pos_lft
    else:
        # -----制御処理-----
        # right_arm
        tcp_pos_rgt, tcp_rot_rgt = rbt_s_rgt.manipulator_dict['arm'].jlc.get_gl_tcp()
        onscreen_tcpframe_rgt.append(gm.gen_frame(pos=tcp_pos_rgt, rotmat=tcp_rot_rgt, length=0.3))
        onscreen_tcpframe_rgt[-1].attach_to(base)

        rel_tcp_pos_rgt = data_array[rgt_hand_num:(rgt_hand_num+3)] - pre_sensor_pos_value_rgt
        pre_sensor_pos_value_rgt = data_array[rgt_hand_num:(rgt_hand_num+3)]
        current_jnt_values_rgt = rbt_s_rgt.manipulator_dict['arm'].jlc._ikt.num_ik(tgt_pos=np.dot(np.linalg.pinv(operator_coordinate), rel_tcp_pos_rgt) + pre_tcp_pos_rgt - init_error_rgt,
                                                                                   tgt_rot=np.dot(np.linalg.pinv(operator_coordinate), rm.rotmat_from_quaternion(data_array[(rgt_hand_num+6):(rgt_hand_num+10)])[:3, :3]),
                                                                                   seed_jnt_values=pre_jnt_values_rgt,
                                                                                   tcp_jntid=7)
        pre_jnt_values_rgt = current_jnt_values_rgt
        pre_tcp_pos_rgt += np.dot(np.linalg.pinv(operator_coordinate), rel_tcp_pos_rgt)
        # left_arm
        tcp_pos_lft, tcp_rot_lft = rbt_s_lft.manipulator_dict['arm'].jlc.get_gl_tcp()
        onscreen_tcpframe_lft.append(gm.gen_frame(pos=tcp_pos_lft, rotmat=tcp_rot_lft, length=0.3))
        onscreen_tcpframe_lft[-1].attach_to(base)

        rel_tcp_pos_lft = data_array[lft_hand_num:(lft_hand_num + 3)] - pre_sensor_pos_value_lft
        pre_sensor_pos_value_lft = data_array[lft_hand_num:(lft_hand_num + 3)]
        current_jnt_values_lft = rbt_s_lft.manipulator_dict['arm'].jlc._ikt.num_ik(tgt_pos=np.dot(np.linalg.pinv(operator_coordinate), rel_tcp_pos_lft) + pre_tcp_pos_lft - init_error_lft,
                                                                                   tgt_rot=np.dot(np.linalg.pinv(operator_coordinate), rm.rotmat_from_quaternion(data_array[(rgt_hand_num+6):(rgt_hand_num+10)])[:3, :3]),
                                                                                   seed_jnt_values=pre_jnt_values_lft,
                                                                                   tcp_jntid=7)
        pre_jnt_values_lft = current_jnt_values_lft
        pre_tcp_pos_lft += np.dot(np.linalg.pinv(operator_coordinate), rel_tcp_pos_lft)

        # -----シミュレータモデル・実機の関節角度の更新等-----
        # right_arm
        collided_result_rgt = rbt_s_rgt.is_collided()
        if collided_result_rgt == True:
            print('Right Arm is Collided!!')
        if abnormal_flag == 0:
            if (current_jnt_values_rgt is not None) and (pre_jnt_values_rgt is not None) and (collided_result_rgt == False):
                if abs(np.linalg.norm(current_jnt_values_rgt - pre_jnt_values_rgt,
                                      ord=2)) <= threshold_jnt_values_displacement_norm and np.all(
                        abs(current_jnt_values_rgt - pre_jnt_values_rgt) <= 0.08):
                    rbt_s_rgt.fk("arm", current_jnt_values_rgt)
                elif operation_count >= 10:
                    abnormal_flag = 1
                pre_jnt_values = current_jnt_values_rgt
                # print(abs(current_jnt_values - pre_jnt_values))
        else:
            abnormal_count += 1
            print(abnormal_count)
            if abnormal_count == 1:
                standard_pos_rgt = np.dot(np.linalg.pinv(operator_coordinate), data_array[(rgt_hand_num):(rgt_hand_num + 3)]) - init_error_rgt - cancel_pos_displacement_rgt
                standard_rot_rgt = np.dot(np.linalg.pinv(operator_coordinate),rm.rotmat_from_quaternion(data_array[(rgt_hand_num + 6):(rgt_hand_num + 10)])[:3,:3])
                # print(f'standard_pos:{standard_pos}, standard_rot:{standard_rot}')
            else:
                judge_rgt = abnormal_judgement(standard_pos=standard_pos_rgt, standard_rot=standard_rot_rgt,
                                               current_pos=np.dot(np.linalg.pinv(operator_coordinate), data_array[(rgt_hand_num):(rgt_hand_num + 3)]) - init_error_rgt - cancel_pos_displacement_rgt,
                                               current_rot=np.dot(np.linalg.pinv(operator_coordinate),rm.rotmat_from_quaternion(data_array[(rgt_hand_num + 6):(rgt_hand_num + 10)])[:3, :3]))
                if judge_rgt == 0:
                    abnormal_count = 0
                    abnormal_flag = 0
                    # print(f'recovery_pos:{np.dot(np.linalg.pinv(operator_coordinate), data_array[(rgt_hand_num):(rgt_hand_num + 3)]) - init_error + agv_pos}, recovery_rot:{np.dot(np.linalg.pinv(operator_coordinate), rm.rotmat_from_quaternion(data_array[(rgt_hand_num+6):(rgt_hand_num+10)])[:3, :3])}')
                    # print(f'recovery_rot_norm:{np.linalg.norm(rm.rotmat_to_euler(np.dot(np.linalg.pinv(operator_coordinate), rm.rotmat_from_quaternion(data_array[(rgt_hand_num+6):(rgt_hand_num+10)])[:3, :3]))-rm.rotmat_to_euler(standard_rot), ord=2)}')
        # print(abs(np.linalg.norm(current_jnt_values-pre_jnt_values, ord=2)))
        # left_arm
        collided_result_lft = rbt_s_lft.is_collided()
        if collided_result_lft == True:
            print('Left Arm is Collided!!')
        if abnormal_flag == 0:
            if (current_jnt_values_lft is not None) and (pre_jnt_values_lft is not None) and (collided_result_lft == False):
                if abs(np.linalg.norm(current_jnt_values_lft - pre_jnt_values_lft, ord=2)) <= threshold_jnt_values_displacement_norm and np.all(abs(current_jnt_values_lft - pre_jnt_values_lft) <= 0.08):
                    rbt_s_rgt.fk("arm", current_jnt_values_lft)
                elif operation_count >= 10:
                    abnormal_flag = 1
                pre_jnt_vallft = current_jnt_values_lft
                # print(abs(current_jnt_values - pre_jnt_values))
        else:
            abnormal_count += 1
            print(abnormal_count)
            if abnormal_count == 1:
                standard_pos_lft = np.dot(np.linalg.pinv(operator_coordinate), data_array[(lft_hand_num):(lft_hand_num + 3)]) - init_error_lft - cancel_pos_displacement_rgt
                standard_rot_lft = np.dot(np.linalg.pinv(operator_coordinate),rm.rotmat_from_quaternion(data_array[(lft_hand_num + 6):(lft_hand_num + 10)])[:3, :3])
                # print(f'standard_pos:{standard_pos}, standard_rot:{standard_rot}')
            else:
                judge_lft = abnormal_judgement(standard_pos=standard_pos_lft, standard_rot=standard_rot_lft,
                                               current_pos=np.dot(np.linalg.pinv(operator_coordinate), data_array[(lft_hand_num):(lft_hand_num + 3)]) - init_error_lft - cancel_pos_displacement_rgt,
                                               current_rot=np.dot(np.linalg.pinv(operator_coordinate), rm.rotmat_from_quaternion(data_array[(lft_hand_num + 6):(lft_hand_num + 10)])[ :3, :3]))
                if judge_lft == 0:
                    abnormal_count = 0
                    abnormal_flag = 0
                    # print(f'recovery_pos:{np.dot(np.linalg.pinv(operator_coordinate), data_array[(rgt_hand_num):(rgt_hand_num + 3)]) - init_error + agv_pos}, recovery_rot:{np.dot(np.linalg.pinv(operator_coordinate), rm.rotmat_from_quaternion(data_array[(rgt_hand_num+6):(rgt_hand_num+10)])[:3, :3])}')
                    # print(f'recovery_rot_norm:{np.linalg.norm(rm.rotmat_to_euler(np.dot(np.linalg.pinv(operator_coordinate), rm.rotmat_from_quaternion(data_array[(rgt_hand_num+6):(rgt_hand_num+10)])[:3, :3]))-rm.rotmat_to_euler(standard_rot), ord=2)}')
        # print(abs(np.linalg.norm(current_jnt_values-pre_jnt_values, ord=2)))

        onscreen_rgt.append(rbt_s_rgt.gen_meshmodel())
        onscreen_rgt[-1].attach_to(base)
        onscreen_lft.append(rbt_s_lft.gen_meshmodel())
        onscreen_lft[-1].attach_to(base)

    operation_count = operation_count + 1
    return task.cont

if __name__ == '__main__':
    edge_of_panel = 1
    range_of_floor = 10
    gm.gen_frame(length=10, thickness=0.02).attach_to(base)

    # シミュレータ内の床の描画
    # def gen_floor(x, y):
    #     ground = gm.gen_box(extent=np.array([edge_of_panel, edge_of_panel, .01]), homomat=rm.homomat_from_posrot(pos=np.array([x + edge_of_panel/2, y + edge_of_panel/2, -.01])))
    #     ground.set_rgba([.6, .6, .6, 1])
    #     ground.attach_to(base)
    #     ground = gm.gen_box(extent=np.array([edge_of_panel, edge_of_panel, .01]), homomat=rm.homomat_from_posrot(pos=np.array([x + edge_of_panel/2*3, y + edge_of_panel/2*3, -.01])))
    #     ground.set_rgba([.6, .6, .6, 1])
    #     ground.attach_to(base)
    #     ground = gm.gen_box(extent=np.array([edge_of_panel, edge_of_panel, .01]), homomat=rm.homomat_from_posrot(pos=np.array([x + edge_of_panel/2*3, y + edge_of_panel / 2, -.01])))
    #     ground.set_rgba([.5, .5, .5, 1])
    #     ground.attach_to(base)
    #     ground = gm.gen_box(extent=np.array([edge_of_panel, edge_of_panel, .01]), homomat=rm.homomat_from_posrot(pos=np.array([x + edge_of_panel/2, y + edge_of_panel/2*3, -.01])))
    #     ground.set_rgba([.5, .5, .5, 1])
    #     ground.attach_to(base)
    # for i in range(0, range_of_floor):
    #     for j in range(0, range_of_floor):
    #         gen_floor(-range_of_floor + i*(edge_of_panel*2), -range_of_floor + j*(edge_of_panel*2))

    # 初期時刻の台車位置を指定位置へ
    # rbt_s_rgt.fk(component_name="agv", jnt_values=init_pos_rgt)
    # rbt_s_lft.fk(component_name="agv", jnt_values=init_pos_lft)

    # xarmのデフォルトの初期姿勢に戻す
    # right_arm
    rbt_s_rgt.fk(jnt_values=arm_homeconf)
    rbt_x_rgt.move_jnts(jnt_values=arm_homeconf)
    # left_arm
    rbt_s_lft.fk(jnt_values=arm_homeconf)
    rbt_x_lft.move_jnts(jnt_values=arm_homeconf)

    if not ((arm_homeconf == rbt_x_rgt.get_jnt_values()) and (arm_homeconf == rbt_x_lft.get_jnt_values())):
        print('Initialization is failed! Program is stopped!')
        exit()

    # TODO: xarmのグリッパを指定初期姿勢へ更新(generate_init_jnt_valuesにて初期姿勢を確認すること)
    # right_arm
    rbt_s_rgt.fk(jnt_values=init_jnt_values_rgt)
    rbt_x_rgt.move_jnts(jnt_values=init_jnt_values_rgt, component_name="arm")
    # left_arm
    rbt_s_lft.fk(jnt_values=init_jnt_values_lft)
    rbt_x_lft.move_jnts(jnt_values=init_jnt_values_lft, component_name="arm")

    # 安全確認
    safety_check = mb.askokcancel("安全確認", "ロボット周辺の安全を今一度確認してください")
    if safety_check == True:
        print('actual machine will start operation in 3 seconds.')
        time.sleep(3)
    else:
        exit()

    # 並列処理
    threading.Thread(target=data_collect, args=(data_list,)).start()
    taskMgr.doMethodLater(update_frequency, program_stop, "program_stop",
                          extraArgs=None,
                          appendTask=True)
    taskMgr.doMethodLater(update_frequency, agv_move, "agv_move",
                          extraArgs=None,
                          appendTask=True)
    taskMgr.doMethodLater(update_frequency, arm_move, "arm_move",
                          extraArgs=[rbt_s_rgt, rbt_s_lft, onscreen, onscreen_lft],
                          appendTask=True)

    base.run()
