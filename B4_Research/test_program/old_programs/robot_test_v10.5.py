"""
---robot_test_v10概要説明---
Axis NEURONから送信されてきたPERCEPTION NEURONの右手の位置姿勢情報を元にシミュレータ上のxarmに動きを同期させる
WASDキーで台車部分の操作が可能

---robot_test_v10更新箇所---
初期時刻における操作者の手の座標系をxarmの手先の座標系に同期する機能を追加
安全確認ポップアップを追加
ジョイスティックコントローラのボタン入力からマニピュレータの追従動作を停止する機能を追加（2つのボタンを一定時間同時押し）

---備考---
data_adjustment内の台車の位置に関しての補正をarm_move内に移動
"""
import socket
import math
import time
import keyboard
import basis.trimesh.transformations as tr
import threading
import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import tkinter.messagebox as mb
import robot_sim.robots.xarm7_shuidi_mobile.xarm7_shuidi_mobile as xarm

from direct.task.TaskManagerGlobal import taskMgr

# -------------------------各種標準設定-------------------------
# 描画系統
rbt_s = xarm.XArm7YunjiMobile(enable_cc=True) # ロボットモデルの読込
# rbt_x = xac.XArm7(host="10.2.0.203:18300")
# init_jnt_angles = rbt_x.get_jnt_vlaues()
# rbt_s.fk("arm", init_jnt_angles)
base = wd.World(cam_pos=[3, 3, 2], lookat_pos=[0, 0, 1]) # モデリング空間を生成
# gm.gen_frame().attach_to(base) # 座標系の軸を表示

# NeuronDataReader
# NDR_path = os.path.abspath("NeuronDataReader.dll")
# NDR = cdll.LoadLibrary(NDR_path)

# 通信系統
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((socket.gethostname(), 7001)) # 7001番のポートに接続
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
data_list = [None] # データ格納用リスト
pre_data_list = [None]
update_frequency = (1/60) # データの更新頻度(s)(ノード18個未満：120Hz 18個以上：60Hz)
main_data_length = 60*16*4
contactRL_length = 4
# ------------------------------------------------------------

# -----その他の変数-----
data_array = []
onscreen = [] # ロボット描画情報用配列
onscreen_tcpframe = []

operation_count = 0 # 関数動作回数カウンタ
fail_IK_flag = 0 # IK解無しフラッグ

current_jnt_values_rgt = None
pre_jnt_values_rgt = None
pre_agv_pos_rgt = np.zeros(3)
modify_pos = None
modify_rot = None
pre_tcp_pos = None
pre_sensor_pos_value = None
threshold_jnt_values_displacement_norm = 0.15

adjustment_rotmat1 = rm.rotmat_from_euler(0, 135, 0) # 全変数調整用
adjustment_rotmat2 = rm.rotmat_from_euler(0, 0, 90) # グリッパ姿勢調整用

emergency_stop_flag = 0
stop_tic = 0
stop_toc = 0
stopped_flag = 0
is_data_receive = 0

rgt_hand_num = 11*16
# --------------------

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

def cvt_rot_WtoM(world_rot):
    global adjustment_rotmat2
    gripper_euler = np.zeros(3)

    tmp_euler = rm.rotmat_to_euler(np.dot(adjustment_rotmat2, world_rot))
    gripper_euler[0] = -tmp_euler[1]
    gripper_euler[1] = -tmp_euler[0]
    gripper_euler[2] = tmp_euler[2]

    return rm.rotmat_from_euler(ai=gripper_euler[0], aj=gripper_euler[1], ak=gripper_euler[2])

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
            data_array[i-3: i+1] = rm.quaternion_from_matrix(adjustmented_rot1)
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
                    'x': keyboard.is_pressed('x'),  # gripper open
                    'z': keyboard.is_pressed('z')}  # gripper close
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
    elif pressed_keys["x"] and sum(values_list) == 1:
        rbt_s.jaw_to(jawwidth=.085)
    elif pressed_keys["z"] and sum(values_list) == 1:
        rbt_s.jaw_to(jawwidth=0)

    onscreen.append(rbt_s.gen_meshmodel())
    onscreen[-1].attach_to(base)

    return task.cont

pre_sensor_quarternion = np.zeros(4)
pre_rot = np.zeros((3, 3))
modify_rot_for_pos = np.zeros((3, 3))
# アーム動作用関数
def arm_move(rbt_s, onscreen, task):
    global data_array, current_jnt_values_rgt, pre_jnt_values_rgt, fail_IK_flag, operation_count, modify_pos, modify_rot, pre_tcp_pos, pre_sensor_pos_value, \
        pre_rot, pre_sensor_quarternion, modify_rot_for_pos

    agv_pos_and_angle = rbt_s.get_jnt_values("agv")
    agv_pos = np.array([agv_pos_and_angle[0], agv_pos_and_angle[1], 0])

    model_remove(onscreen)
    model_remove(onscreen_tcpframe)

    if data_adjustment(data_list) == "No Data":
        return task.cont
    else:
        data_array = data_adjustment(data_list)

    if operation_count == 0:
        tcp_pos, tcp_rot = rbt_s.manipulator_dict['arm'].jlc.get_gl_tcp()
        modify_rot = np.dot(tcp_rot, np.linalg.pinv(rm.rotmat_from_quaternion(data_array[(rgt_hand_num+6):(rgt_hand_num+10)])[:3, :3]))

        #
        modify_rot_for_pos[:2, :2] = modify_rot[:2, :2]
        modify_rot_for_pos[2, 2] = 1
        print(modify_rot)
        print(modify_rot_for_pos)

        pre_jnt_values = rbt_s.get_jnt_values(component_name="arm")
        pre_tcp_pos = tcp_pos
        pre_sensor_pos_value = data_array[rgt_hand_num:(rgt_hand_num+3)]

        #
        pre_rot = tcp_rot
        pre_sensor_qua = data_array[(rgt_hand_num + 6):(rgt_hand_num + 10)]
    else:
        tcp_pos, tcp_rot = rbt_s.manipulator_dict['arm'].jlc.get_gl_tcp()
        onscreen_tcpframe.append(gm.gen_frame(pos=tcp_pos, rotmat=tcp_rot, length=0.3))
        onscreen_tcpframe[-1].attach_to(base)

        # 制御処理
        rel_tcp_pos = data_array[rgt_hand_num:(rgt_hand_num+3)] - pre_sensor_pos_value
        pre_sensor_pos_value = data_array[rgt_hand_num:(rgt_hand_num+3)]

        # 追加
        rel_tcp_euler = rm.quaternion_to_euler(data_array[(rgt_hand_num + 6):(rgt_hand_num + 10)]) - rm.quaternion_to_euler(pre_sensor_qua)
        print(np.linalg.norm(rel_tcp_euler, ord=2))
        pre_sensor_qua = data_array[(rgt_hand_num + 6):(rgt_hand_num + 10)]
        rel_tcp_rot = rm.rotmat_from_euler(ai=rel_tcp_euler[0], aj=rel_tcp_euler[1], ak=rel_tcp_euler[2])

        current_jnt_values = rbt_s.manipulator_dict['arm'].jlc._ikt.num_ik(tgt_pos=pre_tcp_pos + np.dot(modify_rot_for_pos, rel_tcp_pos) + agv_pos,
                                                                           tgt_rot=np.dot(cvt_rot_WtoM(rel_tcp_rot), pre_rot),
                                                                           seed_jnt_values=pre_jnt_values,
                                                                           tcp_jntid=7)
        # pre_jnt_values = current_jnt_values
        pre_tcp_pos += np.dot(modify_rot_for_pos, rel_tcp_pos)
        pre_rot = np.dot(cvt_rot_WtoM(rel_tcp_rot), pre_rot)

        # 衝突判定
        collided_result = rbt_s.is_collided()
        if collided_result == True:
            print('Collided!!')

        if (current_jnt_values is not None) and (pre_jnt_values is not None) and (collided_result == False):
            if np.linalg.norm(current_jnt_values-pre_jnt_values, ord=2) <= threshold_jnt_values_displacement_norm and np.all(np.abs(current_jnt_values-pre_jnt_values) < 0.1):
                rbt_s.fk("arm", current_jnt_values)
        # print(abs(np.linalg.norm(current_jnt_values-pre_jnt_values, ord=2)))
        pre_jnt_values = current_jnt_values

        onscreen.append(rbt_s.gen_meshmodel())
        onscreen[-1].attach_to(base)

    operation_count = operation_count + 1
    return task.cont

if __name__ == '__main__':
    edge_of_panel = 1
    range_of_floor = 10
    gm.gen_frame(length=10, thickness=0.02).attach_to(base)

    def gen_floor(x, y):
        ground = gm.gen_box(extent=np.array([edge_of_panel, edge_of_panel, .01]), homomat=rm.homomat_from_posrot(pos=np.array([x + edge_of_panel/2, y + edge_of_panel/2, -.01])))
        ground.set_rgba([.6, .6, .6, 1])
        ground.attach_to(base)
        ground = gm.gen_box(extent=np.array([edge_of_panel, edge_of_panel, .01]), homomat=rm.homomat_from_posrot(pos=np.array([x + edge_of_panel/2*3, y + edge_of_panel/2*3, -.01])))
        ground.set_rgba([.6, .6, .6, 1])
        ground.attach_to(base)
        ground = gm.gen_box(extent=np.array([edge_of_panel, edge_of_panel, .01]), homomat=rm.homomat_from_posrot(pos=np.array([x + edge_of_panel/2*3, y + edge_of_panel / 2, -.01])))
        ground.set_rgba([.5, .5, .5, 1])
        ground.attach_to(base)
        ground = gm.gen_box(extent=np.array([edge_of_panel, edge_of_panel, .01]), homomat=rm.homomat_from_posrot(pos=np.array([x + edge_of_panel/2, y + edge_of_panel/2*3, -.01])))
        ground.set_rgba([.5, .5, .5, 1])
        ground.attach_to(base)
    for i in range(0, range_of_floor):
        for j in range(0, range_of_floor):
            gen_floor(-range_of_floor + i*(edge_of_panel*2), -range_of_floor + j*(edge_of_panel*2))

    sim_init_pos, sim_init_rot = rbt_s.manipulator_dict['arm'].jlc.get_gl_tcp()
    specified_init_pos = sim_init_pos
    specified_init_rot = np.array([[0, -1, 0],
                                   [-1, 0, 0],
                                   [0, 0, -1]])
    specified_init_jnt_values = rbt_s.ik(tgt_pos=specified_init_pos, tgt_rotmat=specified_init_rot)
    rbt_s.fk(jnt_values=specified_init_jnt_values)

    safety_check = mb.askokcancel("安全確認", "ロボット周辺の安全を今一度確認してください")
    if safety_check == False:
        exit()

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
