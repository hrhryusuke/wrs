"""
---robot_test_v7概要説明---
Axis NEURONから送信されてきたPERCEPTION NEURONの右手の位置姿勢情報を元にシミュレータ上のxarmに動きを同期させる（手首の回転が微妙？）

---v7更新箇所---
Levenberg-Marquardt法を適用
速い動作を行っても誤差を蓄積しない

---備考---
"""
import socket
import math
from direct.task.TaskManagerGlobal import taskMgr
from ctypes import *
import basis.trimesh.transformations as tr
import robot_sim._kinematics.jlchain_ik as jl
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
rbt_s = xarm.XArm7YunjiMobile(enable_cc=True) # ロボットモデルの読込
# rbt_x = xac.XArm7(host="10.2.0.203:18300")
# init_jnt_angles = rbt_x.get_jnt_vlaues()
# rbt_s.fk("arm", init_jnt_angles)
zoom_rate = 2
base = wd.World(cam_pos=[7/zoom_rate, 5/zoom_rate, 0.5], lookat_pos=[0, 0, 0.5]) # モデリング空間を生成
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

# グリッパ姿勢調整用回転行列
adjustment_rotmat1 = rm.rotmat_from_euler(0, 135, 0) # 全変数調整用
adjustment_rotmat2 = rm.rotmat_from_euler(0, 0, 90) # グリッパ姿勢調整用
# ------------------------------------------------------------

# -----その他の変数-----
data_array = []
attached_list = [] # 球のgmモデルについての情報を格納するリスト
attached_list_tcp = []
onscreen = [] # ロボット描画情報用配列

pre_agv_pos_rgt = np.zeros(3)
pre_rgt_hand_info = np.zeros(6) # 前の時刻の右手の甲の位置姿勢に関する配列
current_rgt_hand_info = np.zeros(6)
displacement = None
operation_count = 0 # 関数動作回数カウンタ
arm_update_weight = 0.1 # armの位置更新の際の重み
fail_IK_flag = 0 # IK解無しフラッグ

current_jnt_values_rgt = None
jnt_displacement = None
threshold_jnt_displacement = 5
pre_jnt_displacement = None
k = np.array([0, 0, 0, 0, 0, 0, 0])
pre_manipulability = None
current_manipulability = None
manipulability_displacement = None

rgt_hand_num = 11*16*0
# --------------------
pre_jnt_values_rgt = None
init_error_rgt = []
test_arg1 = None
test_arg2 = None

# データ受信用関数
def data_collect(data_list):
    while True:
        # データ受信
        data_list[0] = s.recv(main_data_length)
        s.recv(2 * contactRL_length)  # contactLRの分を受信だけする（ズレ調整）

# データ調整用関数
# def data_adjustment(data_list):
#     global data_array
#
#     # データ型の調整
#     b_msg = data_list[0]
#     if b_msg is None:
#         return "No Data"
#     tmp_data_array = np.frombuffer(b_msg, dtype=np.float32)
#
#     # 座標調整
#     data_array = np.zeros(int(tmp_data_array.size))
#     for i in range(0, tmp_data_array.size):
#         if (i % 16) == 0 or (i % 16) == 2:
#             data_array[i] = tmp_data_array[i] * (-1)
#         else:
#             data_array[i] = tmp_data_array[i]
#
#         # if (i % 16) == 6 or (i % 16) == 8 or (i % 16) == 9:
#         #     data_array[i] = tmp_data_array[i] * -1
#
#     return data_array
def data_adjustment(data):
    data[0: 3] = np.dot(adjustment_rotmat1, data[0: 3].T)
    data[3: 7] = rm.quaternion_from_matrix(np.dot(adjustment_rotmat1, rm.rotmat_from_quaternion(data[3: 7])[:3, :3]))

# シミュレータ描画用関数
def display(data_array):
    global k, pre_manipulability, current_manipulability, manipulability_displacement, pre_rgt_hand_info, current_rgt_hand_info, \
        current_jnt_values_rgt, displacement, pre_jnt_displacement, fail_IK_flag, operation_count,pre_jnt_values_rgt, init_error_rgt, test_arg1, attached_list, rbt_s, onscreen, pre_agv_pos_rgt

    for item in onscreen:
        item.detach()
    onscreen.clear()

    # data_array = data_adjustment(data_list)
    # if data_adjustment(data_list) == "No Data":
    #     print('No data')
    #     # return task.cont
    # else:
    #     data_array = data_adjustment(data_list)

    test_euler = None
    if operation_count == 0:
        tcp_pos, tcp_rot = rbt_s.manipulator_dict['arm'].jlc.get_gl_tcp()
        init_error = data_array[rgt_hand_num:(rgt_hand_num + 3)] - tcp_pos

        attached_list.append(gm.gen_sphere(pos=data_array[rgt_hand_num:(rgt_hand_num + 3)] - init_error))
        attached_list[-1].attach_to(base)
        pre_pos[:] = data_array[rgt_hand_num:(rgt_hand_num + 3)] - init_error

        attached_list_tcp.append(gm.gen_sphere(pos=tcp_pos))
        attached_list_tcp[-1].attach_to(base)

        pre_jnt_values = rbt_s.get_jnt_values(component_name="arm")

        pre_rgt_hand_info = np.hstack([data_array[rgt_hand_num:(rgt_hand_num+3)], np.array(list(tr.euler_from_quaternion(data_array[(rgt_hand_num+3):(rgt_hand_num+7)])))])
        pre_manipulability = rbt_s.manipulator_dict['arm'].jlc._ikt.manipulability(tcp_jntid=7)
    else:
        # 球の描画処理
        npvec3 = data_array[rgt_hand_num:(rgt_hand_num + 3)] - init_error - pre_pos
        attached_list[0]._objpdnp.setPos(npvec3[0], npvec3[1], npvec3[2])
        attached_list[0].set_rgba([0, 1, 0, 1])
        pre_pos = npvec3

        tcp_pos, tcp_rot = rbt_s.manipulator_dict['arm'].jlc.get_gl_tcp()
        for item in attached_list_tcp:
            item.detach()
        attached_list_tcp.clear()
        attached_list_tcp.append(gm.gen_sphere(pos=tcp_pos))
        attached_list_tcp[-1].attach_to(base)

        # test_arg1 = rm.rotmat_from_quaternion(data_array[(rgt_hand_num+6):(rgt_hand_num+10)])[:3, :3]
        test_arg2 = rm.rotmat_from_euler(0, math.pi/2, 0)
        # print(test_arg2)

        test_euler = rm.quaternion_to_euler(data_array[(rgt_hand_num + 3):(rgt_hand_num + 7)])
        print(test_euler)
        # 手先姿勢１
        test_euler[0] = test_euler[0] * 1
        test_euler[1] = test_euler[1] - math.pi * 0.5
        test_euler[2] = test_euler[2] * -1
        # print(test_euler)
        # 手先姿勢２
        # test_euler[0] = test_euler[0] * 1
        # test_euler[1] = test_euler[1] - math.pi*0.5
        # test_euler[2] = test_euler[2] * -1 - math.pi*0.5

        # 制御処理
        # 回転の補正なし
        # current_jnt_values = rbt_s.manipulator_dict['arm'].jlc._ikt.num_ik(tgt_pos=npvec3,
        #                                                                    tgt_rot=np.dot(test_arg2, rm.rotmat_from_quaternion(data_array[(rgt_hand_num+6):(rgt_hand_num+10)])[:3, :3]),
        #                                                                    seed_jnt_values=test_arg1,
        #                                                                    tcp_jntid=7)
        # 回転の補正あり
        current_jnt_values = rbt_s.manipulator_dict['arm'].jlc._ikt.num_ik(tgt_pos=data_array[rgt_hand_num:(rgt_hand_num + 3)] - init_error,
                                                                           tgt_rot=rm.rotmat_from_euler(test_euler[0], test_euler[1], test_euler[2]),
                                                                           seed_jnt_values=test_arg1,
                                                                           tcp_jntid=7)

        # jnt_displacement = current_jnt_values - pre_jnt_values
        pre_jnt_values = current_jnt_values

        # 衝突判定
        # collided_result = rbt_s.is_collided()
        # if collided_result == True:
        #     print('Collided!!')

        if current_jnt_values is not None:
            rbt_s.fk("arm", current_jnt_values)

        # f = open('jnt_displacement.txt', 'a', encoding='UTF-8')
        # f.write(f'{jnt_displacement}')
        # f.write('\n')
        # f.close()

        # ロボットモデルを生成して描画
        onscreen.append(rbt_s.gen_meshmodel())
        onscreen[-1].attach_to(base)

        # print(len(onscreen))

    tcp_pos, tcp_rot = rbt_s.manipulator_dict['arm'].jlc.get_gl_tcp()
    operation_count = operation_count + 1
    return tcp_pos - (data_array[rgt_hand_num:(rgt_hand_num + 3)] - init_error), np.linalg.norm(tcp_pos - (data_array[rgt_hand_num:(rgt_hand_num + 3)] - init_error))
    # return tcp_pos - pre_pos, np.linalg.norm(tcp_pos - pre_pos)
    # return task.cont

if __name__ == '__main__':
    data = np.loadtxt('right_hand_info.txt')
    error = None
    error_norm = None

    for i in range(data.shape[0]):
        data_adjustment(data[i, :])
        error, error_norm = display(data[i, :])

    print(f'\nError:{error}, Error Norm:{error_norm}')
    # threading.Thread(target=data_collect, args=(data_list,)).start()
    # taskMgr.doMethodLater(update_frequency, display, "display",
    #                       extraArgs=[attached_list, rbt_s, onscreen, pre_pos],
    #                       appendTask=True)
    # gm.gen_frame(length=1, thickness=0.01).attach_to(base)
    # gm.gen_sphere(pos=[0.4, 0, 0.95]).attach_to(base)
    base.run()
