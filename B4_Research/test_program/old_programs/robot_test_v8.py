"""
---robot_test_v8概要説明---
Axis NEURONから送信されてきたPERCEPTION NEURONの右手の位置姿勢情報を元にシミュレータ上のxarmに動きを同期させる（手首の回転が微妙？）
WASDキーで台車部分の操作が可能

---v8更新箇所---
台車の移動を追加

---備考---
台車の移動および回転の速度は実機のものと対応はしていない

台車移動についてのアーム部分への指定する方法が2通りある
①右手の位置に単純に台車部分の位置を足すだけの方法（動作は理想的ではあるが，実機での実装に向いていない）
②コンバータを用いて世界座標系と台車座標系の変換を行う方法（動作は理想的ではないが，実機での実装に向いている）
***方法②正常に動作せず（2022/1/5）
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
rbt_s = xarm.XArm7YunjiMobile(enable_cc=True) # ロボットモデルの読込
# rbt_x = xac.XArm7(host="10.2.0.203:18300")
# init_jnt_angles = rbt_x.get_jnt_vlaues()
# rbt_s.fk("arm", init_jnt_angles)
base = wd.World(cam_pos=[6.5, -6.5, 2], lookat_pos=[0, 0, 0]) # モデリング空間を生成
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
fail_IK_flag = 0 # IK解無しフラッグ

current_jnt_values_rgt = None
pre_jnt_values_rgt = None
jnt_displacement = None
threshold_jnt_displacement = 5
pre_jnt_displacement = None
k = np.array([0, 0, 0, 0, 0, 0, 0])
pre_manipulability = None
current_manipulability = None
manipulability_displacement = None
init_error_rgt = None
init_ef_pos = None
init_ef_rot = None
init_operator_posture = None
init_operator_posture_pinv = None
posture_modify_matrix = None

rgt_hand_num = 11*16
# --------------------

def model_remove():
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
    global data_array
    agv_pos_and_angle = rbt_s.get_jnt_values("agv")
    agv_pos = np.array([agv_pos_and_angle[0], agv_pos_and_angle[1], 0])

    # データ型の調整
    b_msg = data_list[0]
    if b_msg is None:
        return "No Data"
    tmp_data_array = np.frombuffer(b_msg, dtype=np.float32)

    # 座標等調整
    data_array = np.zeros(int(tmp_data_array.size))
    for i in range(0, tmp_data_array.size):
        if (i % 16) == 0 or (i % 16) == 2:
            data_array[i] = tmp_data_array[i] * (-1)
        else:
            data_array[i] = tmp_data_array[i]
    # 台車座標系への調整
    for i in range(0, tmp_data_array.size, 16): # 方法①
        data_array[i:(i + 3)] = data_array[i:(i + 3)] + agv_pos
    # 方法②はarm_move内に記述

    return data_array

# 台車移動用関数
def agv_move(task):
    global onscreen, pre_agv_pos_rgt, current_jnt_values_rgt
    agv_speed_weight = .01
    agv_rotation_weight = .8

    model_remove()
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

# アーム動作用関数
def arm_move(rbt_s, onscreen, task):
    global k, pre_manipulability, current_manipulability, manipulability_displacement, data_array, pre_rgt_hand_info, current_rgt_hand_info, \
        current_jnt_values_rgt, displacement, pre_jnt_displacement, fail_IK_flag, operation_count, pre_jnt_values_rgt, init_error_rgt, init_ef_pos, init_ef_rot, \
        init_operator_posture, init_operator_posture_pinv, posture_modify_matrix

    model_remove()
    if data_adjustment(data_list) == "No Data":
        return task.cont
    else:
        data_array = data_adjustment(data_list)

    if operation_count == 0:
        init_ef_pos, init_ef_rot = rbt_s.manipulator_dict['arm'].jlc.get_gl_tcp()
        init_error = data_array[rgt_hand_num:(rgt_hand_num+3)] - init_ef_pos
        init_operator_posture = rm.rotmat_from_quaternion(data_array[(rgt_hand_num + 6):(rgt_hand_num + 10)])[:3, :3]
        init_operator_posture_pinv = np.linalg.pinv(init_operator_posture)
        posture_modify_matrix = np.dot(init_ef_rot, init_operator_posture_pinv)
        attached_list.append(gm.gen_sphere(pos=(data_array[rgt_hand_num:(rgt_hand_num+3)] - init_error)))
        attached_list[-1].attach_to(base)

        pre_jnt_values = rbt_s.get_jnt_values(component_name="arm")

        pre_rgt_hand_info = np.hstack([data_array[rgt_hand_num:(rgt_hand_num+3)], np.array(list(tr.euler_from_quaternion(data_array[(rgt_hand_num+6):(rgt_hand_num+10)])))])
        pre_manipulability = rbt_s.manipulator_dict['arm'].jlc._ikt.manipulability(tcp_jntid=7)
    else:
        # 球の描画処理
        npvec3 = data_array[rgt_hand_num:(rgt_hand_num + 3)] - init_ef_pos
        attached_list[0]._objpdnp.setPos(npvec3[0], npvec3[1], npvec3[2])
        attached_list[0].set_rgba([0, 1, 0, 1])
        # print(init_error)
        # print(npvec3)

        # 手先姿勢補正
        ef_rotation_euler = rm.quaternion_to_euler(data_array[(rgt_hand_num + 6):(rgt_hand_num + 10)])
        ef_rotation_euler[0] = ef_rotation_euler[0] * 1
        ef_rotation_euler[1] = ef_rotation_euler[1] - math.pi * 0.5
        ef_rotation_euler[2] = ef_rotation_euler[2] * -1

        # 制御処理
        # 方法①
        current_jnt_values = rbt_s.manipulator_dict['arm'].jlc._ikt.num_ik(tgt_pos=npvec3,
                                                                           tgt_rot=rm.rotmat_from_euler(ef_rotation_euler[0], ef_rotation_euler[1], ef_rotation_euler[2]),
                                                                           seed_jnt_values=pre_jnt_values,
                                                                           tcp_jntid=7)
        pre_jnt_values = current_jnt_values
        # 方法②
        # ef_rot = np.dot(posture_modify_matrix, rm.rotmat_from_quaternion(data_array[(rgt_hand_num + 6):(rgt_hand_num + 10)])[:3, :3])
        # [gl_pos, gl_rot] = rbt_s.manipulator_dict['arm'].jlc.cvt_loc_tcp_to_gl(loc_pos=npvec3, loc_rotmat=ef_rot)
        # current_jnt_values = rbt_s.manipulator_dict['arm'].jlc._ikt.num_ik(tgt_pos=gl_pos,
        #                                                                    tgt_rot=gl_rot,
        #                                                                    seed_jnt_values=pre_jnt_values,
        #                                                                    tcp_jntid=7)
        # pre_jnt_values = current_jnt_values

        # f = open('gl_pos.txt', 'a', encoding='UTF-8')
        # f.write(f'{gl_pos}')
        # f.write('\n')
        # f.close()
        # f = open('gl_rot.txt', 'a', encoding='UTF-8')
        # f.write(f'{gl_rot}')
        # f.write('\n')
        # f.close()

        # f = open('gl_tcp.txt', 'a', encoding='UTF-8')
        # f.write(str(rbt_s.manipulator_dict['arm'].jlc.get_gl_tcp()))
        # f.write('\n')
        # f.close()

        # 衝突判定
        collided_result = rbt_s.is_collided()
        if collided_result == True:
            print('Collided!!')

        if current_jnt_values is not None:
            rbt_s.fk("arm", current_jnt_values)

        # ロボットモデルを生成して描画
        onscreen.append(rbt_s.gen_meshmodel())
        onscreen[-1].attach_to(base)

    operation_count = operation_count + 1
    return task.cont

if __name__ == '__main__':
    edge_of_panel = 1
    range_of_floor = 10
    gm.gen_frame(length=10, thickness=0.02).attach_to(base)
    gm.gen_sphere(pos=[0.4, 0, 0.95]).attach_to(base)

    def generate_floor_part(x, y):
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
            generate_floor_part(-range_of_floor + i*(edge_of_panel*2), -range_of_floor + j*(edge_of_panel*2))

    threading.Thread(target=data_collect, args=(data_list,)).start()
    taskMgr.doMethodLater(update_frequency, agv_move, "agv_move",
                          extraArgs=None,
                          appendTask=True)
    taskMgr.doMethodLater(update_frequency, arm_move, "arm_move",
                          extraArgs=[rbt_s, onscreen],
                          appendTask=True)
    base.run()
