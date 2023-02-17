"""
---robot_dual_v2概要説明---
Axis NEURONから送信されてきたPERCEPTION NEURONの右手の位置姿勢情報を元にシミュレータ上の2台のxarmに動きを同期させる
キー入力で左右のxarmを独立に台車移動させることが可能（左：WASDキー，右：IJKLキー）

---robot_dual_v2更新箇所---
初期時刻における操作者の手の座標系をxarmの手先の座標系に同期する機能を追加
左右のxarmが独立に台車移動出来る機能を追加

---備考---
アームの方向は台車の方向に依存しない
"""
import socket
import math
import keyboard
import basis.trimesh.transformations as tr
import threading
import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import robot_sim.robots.xarm7_shuidi_mobile.xarm7_shuidi_mobile as xarm

from direct.task.TaskManagerGlobal import taskMgr

# -------------------------各種標準設定-------------------------
# 描画系統
rbt_s_rgt = xarm.XArm7YunjiMobile(enable_cc=True) # ロボットモデルの読込
rbt_s_lft = xarm.XArm7YunjiMobile(enable_cc=True)
# rbt_x = xac.XArm7(host="10.2.0.203:18300")
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
# modify_pos = None
modify_rot_rgt = None
pre_tcp_pos_rgt = None
pre_sensor_pos_value_rgt = None

current_jnt_values_lft = None
pre_jnt_values_lft = None
current_agv_pos_lft = np.zeros(3)
pre_agv_pos_lft = init_pos_lft
# modify_pos = None
modify_rot_lft = None
pre_tcp_pos_lft = None
pre_sensor_pos_value_lft = None

# その他
operation_count = 0 # 関数動作回数カウンタ
fail_IK_flag = 0 # IK解無しフラッグ
# --------------------

# シミュレータ内モデル除去用関数
def model_remove(onscreen):
    for item in onscreen:
        item.detach()
    onscreen.clear()

# データ受信用関数
def data_collect(data_list):
    while True:
        # データ受信
        data_list[0] = s.recv(main_data_length)
        s.recv(2 * contactRL_length)  # contactLRの分を受信だけする（ズレ調整）

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

    model_remove(onscreen_rgt)
    model_remove(onscreen_lft)

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
        current_agv_pos_lft = np.array(pre_agv_pos_lft + [agv_speed_weight*agv_direction_lft[0], agv_speed_weight*agv_direction_lft[1], math.radians(agv_rotation_weight)])
        rbt_s_lft.fk(component_name='agv', jnt_values=current_agv_pos_lft)
        pre_agv_pos_lft = current_agv_pos_lft
    elif pressed_keys_lft["w"] and pressed_keys_lft["d"]:
        current_agv_pos_lft = np.array(pre_agv_pos_lft + [agv_speed_weight * agv_direction_lft[0], agv_speed_weight * agv_direction_lft[1], math.radians(-agv_rotation_weight)])
        rbt_s_lft.fk(component_name='agv', jnt_values=current_agv_pos_lft)
        pre_agv_pos_lft = current_agv_pos_lft
    elif pressed_keys_lft["s"] and pressed_keys_lft["a"]:
        current_agv_pos_lft = np.array(pre_agv_pos_lft + [-agv_speed_weight * agv_direction_lft[0], -agv_speed_weight * agv_direction_lft[1], math.radians(-agv_rotation_weight)])
        rbt_s_lft.fk(component_name='agv', jnt_values=current_agv_pos_lft)
        pre_agv_pos_lft = current_agv_pos_lft
    elif pressed_keys_lft["s"] and pressed_keys_lft["d"]:
        current_agv_pos_lft = np.array(pre_agv_pos_lft + [-agv_speed_weight * agv_direction_lft[0], -agv_speed_weight * agv_direction_lft[1], math.radians(agv_rotation_weight)])
        rbt_s_lft.fk(component_name='agv', jnt_values=current_agv_pos_lft)
        pre_agv_pos_lft = current_agv_pos_lft
    elif pressed_keys_lft["w"] and sum(values_list_lft) == 1:
        current_agv_pos_lft = np.array(pre_agv_pos_lft + [agv_speed_weight * agv_direction_lft[0], agv_speed_weight * agv_direction_lft[1], math.radians(0)])
        rbt_s_lft.fk(component_name='agv', jnt_values=current_agv_pos_lft)
        pre_agv_pos_lft = current_agv_pos_lft
    elif pressed_keys_lft["s"] and sum(values_list_lft) == 1:
        current_agv_pos_lft = np.array(pre_agv_pos_lft + [-agv_speed_weight * agv_direction_lft[0], -agv_speed_weight * agv_direction_lft[1], math.radians(0)])
        rbt_s_lft.fk(component_name='agv', jnt_values=current_agv_pos_lft)
        pre_agv_pos_lft = current_agv_pos_lft
    elif pressed_keys_lft["a"] and sum(values_list_lft) == 1:
        current_agv_pos_lft = np.array(pre_agv_pos_lft + [0, 0, math.radians(agv_rotation_weight)])
        rbt_s_lft.fk(component_name='agv', jnt_values=current_agv_pos_lft)
        pre_agv_pos_lft = current_agv_pos_lft
    elif pressed_keys_lft["d"] and sum(values_list_lft) == 1:
        current_agv_pos_lft = np.array(pre_agv_pos_lft + [0, 0, math.radians(-agv_rotation_weight)])
        rbt_s_lft.fk(component_name='agv', jnt_values=current_agv_pos_lft)
        pre_agv_pos_lft = current_agv_pos_lft
    elif pressed_keys_lft["z"] and sum(values_list_lft) == 1:
        rbt_s_lft.jaw_to(jawwidth=.085)
    elif pressed_keys_lft["x"] and sum(values_list_lft) == 1:
        rbt_s_lft.jaw_to(jawwidth=0)

    # right_arm
    if pressed_keys_rgt["i"] and pressed_keys_rgt["j"]:
        current_agv_pos_rgt = np.array(pre_agv_pos_rgt + [agv_speed_weight*agv_direction_rgt[0], agv_speed_weight*agv_direction_rgt[1], math.radians(agv_rotation_weight)])
        rbt_s_rgt.fk(component_name='agv', jnt_values=current_agv_pos_rgt)
        pre_agv_pos_rgt = current_agv_pos_rgt
    elif pressed_keys_rgt["i"] and pressed_keys_rgt["l"]:
        current_agv_pos_rgt = np.array(pre_agv_pos_rgt + [agv_speed_weight * agv_direction_rgt[0], agv_speed_weight * agv_direction_rgt[1], math.radians(-agv_rotation_weight)])
        rbt_s_rgt.fk(component_name='agv', jnt_values=current_agv_pos_rgt)
        pre_agv_pos_rgt = current_agv_pos_rgt
    elif pressed_keys_rgt["k"] and pressed_keys_rgt["j"]:
        current_agv_pos_rgt = np.array(pre_agv_pos_rgt + [-agv_speed_weight * agv_direction_rgt[0], -agv_speed_weight * agv_direction_rgt[1], math.radians(-agv_rotation_weight)])
        rbt_s_rgt.fk(component_name='agv', jnt_values=current_agv_pos_rgt)
        pre_agv_pos_rgt = current_agv_pos_rgt
    elif pressed_keys_rgt["k"] and pressed_keys_rgt["l"]:
        current_agv_pos_rgt = np.array(pre_agv_pos_rgt + [-agv_speed_weight * agv_direction_rgt[0], -agv_speed_weight * agv_direction_rgt[1], math.radians(agv_rotation_weight)])
        rbt_s_rgt.fk(component_name='agv', jnt_values=current_agv_pos_rgt)
        pre_agv_pos_rgt = current_agv_pos_rgt
    elif pressed_keys_rgt["i"] and sum(values_list_rgt) == 1:
        current_agv_pos_rgt = np.array(pre_agv_pos_rgt + [agv_speed_weight * agv_direction_rgt[0], agv_speed_weight * agv_direction_rgt[1], math.radians(0)])
        rbt_s_rgt.fk(component_name='agv', jnt_values=current_agv_pos_rgt)
        pre_agv_pos_rgt = current_agv_pos_rgt
    elif pressed_keys_rgt["k"] and sum(values_list_rgt) == 1:
        current_agv_pos_rgt = np.array(pre_agv_pos_rgt + [-agv_speed_weight * agv_direction_rgt[0], -agv_speed_weight * agv_direction_rgt[1], math.radians(0)])
        rbt_s_rgt.fk(component_name='agv', jnt_values=current_agv_pos_rgt)
        pre_agv_pos_rgt = current_agv_pos_rgt
    elif pressed_keys_rgt["j"] and sum(values_list_rgt) == 1:
        current_agv_pos_rgt = np.array(pre_agv_pos_rgt + [0, 0, math.radians(agv_rotation_weight)])
        rbt_s_rgt.fk(component_name='agv', jnt_values=current_agv_pos_rgt)
        pre_agv_pos_rgt = current_agv_pos_rgt
    elif pressed_keys_rgt["l"] and sum(values_list_rgt) == 1:
        current_agv_pos_rgt = np.array(pre_agv_pos_rgt + [0, 0, math.radians(-agv_rotation_weight)])
        rbt_s_rgt.fk(component_name='agv', jnt_values=current_agv_pos_rgt)
        pre_agv_pos_rgt = current_agv_pos_rgt
    elif pressed_keys_rgt["n"] and sum(values_list_rgt) == 1:
        rbt_s_rgt.jaw_to(jawwidth=.085)
    elif pressed_keys_rgt["m"] and sum(values_list_rgt) == 1:
        rbt_s_rgt.jaw_to(jawwidth=0)

    onscreen_rgt.append(rbt_s_rgt.gen_meshmodel())
    onscreen_rgt[-1].attach_to(base)
    onscreen_lft.append(rbt_s_lft.gen_meshmodel())
    onscreen_lft[-1].attach_to(base)

    return task.cont

# アーム動作用関数
def arm_move(rbt_s_rgt, rbt_s_lft, onscreen_rgt, onscreen_lft, task):
    global data_array, fail_IK_flag, operation_count, \
        current_jnt_values_rgt, pre_jnt_values_rgt, modify_rot_rgt, pre_tcp_pos_rgt, pre_sensor_pos_value_rgt, \
        current_jnt_values_lft, pre_jnt_values_lft, modify_rot_lft, pre_tcp_pos_lft, pre_sensor_pos_value_lft

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

    if data_adjustment(data_list) == "No Data":
        return task.cont
    else:
        data_array = data_adjustment(data_list)

    if operation_count == 0:
        # right_arm
        tcp_pos_rgt, tcp_rot_rgt = rbt_s_rgt.manipulator_dict['arm'].jlc.get_gl_tcp()
        modify_rot_rgt = np.dot(tcp_rot_rgt, np.linalg.pinv(rm.rotmat_from_quaternion(data_array[(rgt_hand_num+6):(rgt_hand_num+10)])[:3, :3]))
        pre_jnt_values_rgt = rbt_s_rgt.get_jnt_values(component_name="arm")
        pre_tcp_pos_rgt = tcp_pos_rgt
        pre_sensor_pos_value_rgt = data_array[rgt_hand_num:(rgt_hand_num+3)]
        # left_arm
        tcp_pos_lft, tcp_rot_lft = rbt_s_lft.manipulator_dict['arm'].jlc.get_gl_tcp()
        modify_rot_lft = np.dot(tcp_rot_lft, np.linalg.pinv(rm.rotmat_from_quaternion(data_array[(lft_hand_num + 6):(lft_hand_num + 10)])[:3, :3]))
        pre_jnt_values_lft = rbt_s_lft.get_jnt_values(component_name="arm")
        pre_tcp_pos_lft = tcp_pos_lft
        pre_sensor_pos_value_lft = data_array[lft_hand_num:(lft_hand_num + 3)]
    else:
        # -----制御処理-----
        # right_arm
        tcp_pos_rgt, tcp_rot_rgt = rbt_s_rgt.manipulator_dict['arm'].jlc.get_gl_tcp()
        onscreen_tcpframe_rgt.append(gm.gen_frame(pos=tcp_pos_rgt, rotmat=tcp_rot_rgt, length=0.3))
        onscreen_tcpframe_rgt[-1].attach_to(base)

        rel_tcp_pos_rgt = data_array[rgt_hand_num:(rgt_hand_num+3)] - pre_sensor_pos_value_rgt
        pre_sensor_pos_value_rgt = data_array[rgt_hand_num:(rgt_hand_num+3)]
        current_jnt_values_rgt = rbt_s_rgt.manipulator_dict['arm'].jlc._ikt.num_ik(tgt_pos=pre_tcp_pos_rgt + np.dot(modify_rot_rgt, rel_tcp_pos_rgt) + agv_pos_rgt - init_pos_rgt,
                                                                                   tgt_rot=np.dot(modify_rot_rgt, rm.rotmat_from_quaternion(data_array[(rgt_hand_num+6):(rgt_hand_num+10)])[:3, :3]),
                                                                                   seed_jnt_values=pre_jnt_values_rgt,
                                                                                   tcp_jntid=7)
        pre_jnt_values_rgt = current_jnt_values_rgt
        pre_tcp_pos_rgt += np.dot(modify_rot_rgt, rel_tcp_pos_rgt)
        # left_arm
        tcp_pos_lft, tcp_rot_lft = rbt_s_lft.manipulator_dict['arm'].jlc.get_gl_tcp()
        onscreen_tcpframe_lft.append(gm.gen_frame(pos=tcp_pos_lft, rotmat=tcp_rot_lft, length=0.3))
        onscreen_tcpframe_lft[-1].attach_to(base)

        rel_tcp_pos_lft = data_array[lft_hand_num:(lft_hand_num + 3)] - pre_sensor_pos_value_lft
        pre_sensor_pos_value_lft = data_array[lft_hand_num:(lft_hand_num + 3)]
        current_jnt_values_lft = rbt_s_lft.manipulator_dict['arm'].jlc._ikt.num_ik(tgt_pos=pre_tcp_pos_lft + np.dot(modify_rot_lft, rel_tcp_pos_lft) + agv_pos_lft - init_pos_lft,
                                                                                   tgt_rot=np.dot(modify_rot_lft, rm.rotmat_from_quaternion(data_array[(lft_hand_num + 6):(lft_hand_num + 10)])[:3, :3]),
                                                                                   seed_jnt_values=pre_jnt_values_lft,
                                                                                   tcp_jntid=7)
        pre_jnt_values_lft = current_jnt_values_lft
        pre_tcp_pos_lft += np.dot(modify_rot_lft, rel_tcp_pos_lft)

        # -----シミュレータモデルの更新等-----
        # right_arm
        collided_result_rgt = rbt_s_rgt.is_collided()
        if collided_result_rgt == True:
            print('Collided!!')
        if current_jnt_values_rgt is not None:
            rbt_s_rgt.fk("arm", current_jnt_values_rgt)
        # left_arm
        collided_result_lft = rbt_s_lft.is_collided()
        if collided_result_lft == True:
            print('Collided!!')
        if current_jnt_values_lft is not None:
            rbt_s_lft.fk("arm", current_jnt_values_lft)

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

    rbt_s_rgt.fk(component_name="agv", jnt_values=init_pos_rgt)
    rbt_s_lft.fk(component_name="agv", jnt_values=init_pos_lft)

    # シミュレータ内のxarmのグリッパを指定初期姿勢へ更新
    # right_arm
    sim_init_pos_rgt, sim_init_rot_rgt = rbt_s_rgt.manipulator_dict['arm'].jlc.get_gl_tcp()
    specified_init_pos_rgt = sim_init_pos_rgt
    specified_init_rot_rgt = np.array([[0, -1, 0],
                                       [-1, 0, 0],
                                       [0, 0, -1]])
    specified_init_jnt_values_rgt = rbt_s_rgt.ik(tgt_pos=specified_init_pos_rgt, tgt_rotmat=specified_init_rot_rgt)
    rbt_s_rgt.fk(jnt_values=specified_init_jnt_values_rgt)
    # left_arm
    sim_init_pos_lft, sim_init_rot_lft = rbt_s_lft.manipulator_dict['arm'].jlc.get_gl_tcp()
    specified_init_pos_lft = sim_init_pos_lft
    specified_init_rot_lft = np.array([[0, -1, 0],
                                       [-1, 0, 0],
                                       [0, 0, -1]])
    specified_init_jnt_values_lft = rbt_s_lft.ik(tgt_pos=specified_init_pos_lft, tgt_rotmat=specified_init_rot_lft)
    rbt_s_lft.fk(jnt_values=specified_init_jnt_values_lft)

    # 並列処理の開始
    threading.Thread(target=data_collect, args=(data_list,)).start()
    taskMgr.doMethodLater(update_frequency, agv_move, "agv_move",
                          extraArgs=None,
                          appendTask=True)
    taskMgr.doMethodLater(update_frequency, arm_move, "arm_move",
                          extraArgs=[rbt_s_rgt, rbt_s_lft, onscreen, onscreen_lft],
                          appendTask=True)

    base.run()
