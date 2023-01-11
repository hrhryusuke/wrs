"""
---robot_dual_v1概要説明---
Axis NEURONから送信されてきたPERCEPTION NEURONの両手の位置姿勢情報を元にシミュレータ上の2台のxarmに動きを同期させる

---v11更新箇所---

---備考---
台車移動・初期時刻での操作者の手とxarmの手先の座標系の同期には非対応
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
rbt_s1 = xarm.XArm7YunjiMobile(enable_cc=True) # ロボットモデルの読込
rbt_s2 = xarm.XArm7YunjiMobile(enable_cc=True)
# rbt_x = xac.XArm7(host="10.2.0.203:18300")
# init_jnt_angles = rbt_x.get_jnt_vlaues()
# rbt_s.fk("arm", init_jnt_angles)
base = wd.World(cam_pos=[4, -4, 2], lookat_pos=[0, 0, .5]) # モデリング空間を生成
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
onscreen1 = [] # ロボット描画情報用配列
onscreen2 = []
onscreen_tcpframe1 = []
onscreen_tcpframe2 = []

operation_count = 0 # 関数動作回数カウンタ
fail_IK_flag = 0 # IK解無しフラッグ

current_jnt_values1 = None
pre_jnt_values1 = None
init_error1 = None
current_jnt_values2 = None
pre_jnt_values2 = None
init_error2 = None

adjustment_rotmat1 = rm.rotmat_from_euler(0, 135, 0) # 全変数調整用
adjustment_rotmat2 = rm.rotmat_from_euler(0, 0, 90) # グリッパ姿勢調整用

rgt_hand_num = 11*16
lft_hand_num = 15*16
# --------------------

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
    global data_array, adjustment_rotmat1, adjustment_rotmat2
    gripper_euler = np.zeros(3)
    agv_pos_and_angle1 = rbt_s1.get_jnt_values("agv")
    agv_pos1 = np.array([agv_pos_and_angle1[0], agv_pos_and_angle1[1], 0])
    agv_pos_and_angle2 = rbt_s2.get_jnt_values("agv")
    agv_pos2 = np.array([agv_pos_and_angle2[0], agv_pos_and_angle2[1], 0])

    # データ型の調整
    b_msg = data_list[0]
    if b_msg is None:
        return "No Data"
    tmp_data_array = np.frombuffer(b_msg, dtype=np.float32)

    # 座標調整
    data_array = np.zeros(int(tmp_data_array.size))
    for i in range(0, tmp_data_array.size):
        if (i % 16) == 2: # センサ座標調整
            adjustmented_pos1 = np.dot(adjustment_rotmat1, np.array(tmp_data_array[i-2: i+1]).T)
            # adjustmented_pos2 = np.dot(adjustment_rotmat2, adjustmented_pos1)
            if i == rgt_hand_num+2:
                data_array[i-2: i+1] = adjustmented_pos1 + agv_pos1
            elif i == lft_hand_num+2:
                data_array[i-2: i+1] = adjustmented_pos1 + agv_pos2
            else:
                data_array[i-2: i+1] = adjustmented_pos1
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

# アーム動作用関数
def arm_move(rbt_s1, rbt_s2, onscreen1, onscreen2, task):
    global data_array, current_jnt_values1, pre_jnt_values1, current_jnt_values2, pre_jnt_values2, fail_IK_flag, operation_count, init_error1, init_error2

    model_remove(onscreen1)
    model_remove(onscreen_tcpframe1)
    model_remove(onscreen2)
    model_remove(onscreen_tcpframe2)

    if data_adjustment(data_list) == "No Data":
        return task.cont
    else:
        data_array = data_adjustment(data_list)

    if operation_count == 0:
        # right hand
        tcp_pos1, tcp_rot1 = rbt_s1.manipulator_dict['arm'].jlc.get_gl_tcp()
        init_error1 = data_array[rgt_hand_num:(rgt_hand_num+3)] - tcp_pos1
        pre_jnt_values1 = rbt_s1.get_jnt_values(component_name="arm")
        # left hand
        tcp_pos2, tcp_rot2 = rbt_s2.manipulator_dict['arm'].jlc.get_gl_tcp()
        init_error2 = data_array[lft_hand_num:(lft_hand_num+3)] - tcp_pos2
        pre_jnt_values2 = rbt_s2.get_jnt_values(component_name="arm")
    else:
        # right hand
        tcp_pos1, tcp_rot1 = rbt_s1.manipulator_dict['arm'].jlc.get_gl_tcp()
        onscreen_tcpframe1.append(gm.gen_frame(pos=tcp_pos1, rotmat=tcp_rot1, length=0.3))
        onscreen_tcpframe1[-1].attach_to(base)
        current_jnt_values1 = rbt_s1.manipulator_dict['arm'].jlc._ikt.num_ik(tgt_pos=data_array[rgt_hand_num:(rgt_hand_num+3)] - init_error1,
                                                                             tgt_rot=rm.rotmat_from_quaternion(data_array[(rgt_hand_num+6):(rgt_hand_num+10)])[:3, :3],
                                                                             seed_jnt_values=pre_jnt_values1,
                                                                             tcp_jntid=7)
        pre_jnt_values1 = current_jnt_values1
        collided_result1 = rbt_s1.is_collided()
        if collided_result1 == True:
            print('Right Hand is Collided!!')
        if current_jnt_values1 is not None:
            rbt_s1.fk("arm", current_jnt_values1)

        # left hand
        tcp_pos2, tcp_rot2 = rbt_s2.manipulator_dict['arm'].jlc.get_gl_tcp()
        onscreen_tcpframe2.append(gm.gen_frame(pos=tcp_pos2, rotmat=tcp_rot2, length=0.3))
        onscreen_tcpframe2[-1].attach_to(base)
        current_jnt_values2 = rbt_s2.manipulator_dict['arm'].jlc._ikt.num_ik(tgt_pos=data_array[lft_hand_num:(lft_hand_num+3)] - init_error2,
                                                                             tgt_rot=rm.rotmat_from_quaternion(data_array[(lft_hand_num+6):(lft_hand_num+10)])[:3, :3],
                                                                             seed_jnt_values=pre_jnt_values2,
                                                                             tcp_jntid=7)
        pre_jnt_values2 = current_jnt_values2
        collided_result2 = rbt_s2.is_collided()
        if collided_result2 == True:
            print('Left Hand is Collided!!')
        if current_jnt_values2 is not None:
            rbt_s2.fk("arm", current_jnt_values2)

        # ロボットモデルを生成して描画
        onscreen1.append(rbt_s1.gen_meshmodel())
        onscreen1[-1].attach_to(base)
        onscreen2.append(rbt_s2.gen_meshmodel())
        onscreen2[-1].attach_to(base)

    # f = open('tcp_rot.txt', 'a', encoding='UTF-8')
    # f.write(f'{tcp_rot1}')
    # f.write('\n')
    # f.close()

    operation_count = operation_count + 1
    return task.cont

if __name__ == '__main__':
    edge_of_panel = 1
    range_of_floor = 10
    gm.gen_frame(length=10, thickness=0.02).attach_to(base)

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

    rbt_s1.fk(component_name="agv", jnt_values=np.array([0, -.35, 0]))
    rbt_s2.fk(component_name="agv", jnt_values=np.array([0, .35, 0]))

    threading.Thread(target=data_collect, args=(data_list,)).start()
    taskMgr.doMethodLater(update_frequency, arm_move, "arm_move",
                          extraArgs=[rbt_s1, rbt_s2, onscreen1, onscreen2],
                          appendTask=True)
    base.run()
