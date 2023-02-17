"""
---robot_actual_v4概要説明---
Axis NEURONから送信されてきたPERCEPTION NEURONの右手の位置姿勢情報を元にシミュレータ上のxarmに動きを同期させる
coordinate_adjustmentベース

---更新箇所---
LM法を用いた逆運動学の解法を使用

---備考---
"""
import socket
import math
import time
from direct.task.TaskManagerGlobal import taskMgr
import threading
import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import robot_sim.robots.xarm7_shuidi_mobile.xarm7_shuidi_mobile as xarm
import robot_con.xarm_shuidi.xarm_shuidi_client as rbx

# -------------------------各種標準設定-------------------------
# 描画系統
base = wd.World(cam_pos=[5, -5, 5], lookat_pos=[0, 0, 0]) # モデリング空間を生成
rbt_s = xarm.XArm7YunjiMobile(enable_cc=True) # ロボットモデルの読込
# rbt_s.gen_meshmodel().attach_to(base)
rbt_x = rbx.XArmShuidiClient(host="10.2.0.203:18300")
# rbt_s.fk("arm", init_jnt_angles)
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
onscreen = [] # ロボット描画情報用配列
onscreen_tcpframe = []

operation_count = 0 # 関数動作回数カウンタ
fail_IK_flag = 0 # IK解無しフラッグ

current_jnt_values_rgt = None
pre_jnt_values_rgt = None
init_error_rgt = []

adjustment_rotmat1 = rm.rotmat_from_euler(0, 135, 0) # 全変数調整用
adjustment_rotmat2 = rm.rotmat_from_euler(0, 0, 90) # グリッパ姿勢調整用

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
    global data_array, adjustment_rotmat1, adjustment_rotmat2
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

def init_operation(init_pos, init_rot, specified_init_pos, specified_init_rot):
    num_phase = 100
    IO_current_jnt_values = np.zeros(7)
    IO_pre_jnt_values = rbt_s.get_jnt_values(component_name="arm")

    phase_pos_displacement = (specified_init_pos - init_pos) / num_phase
    phase_rot_displacement = (specified_init_rot - init_rot) / num_phase
    for i in range(1, num_phase+1):
        IO_current_jnt_values = rbt_s.manipulator_dict['arm'].jlc._ikt.num_ik(tgt_pos=init_pos + phase_pos_displacement*i,
                                                                              tgt_rot=init_rot + phase_rot_displacement*i,
                                                                              seed_jnt_values=IO_pre_jnt_values,
                                                                              tcp_jntid=7)
        IO_pre_jnt_values = IO_current_jnt_values
        if IO_current_jnt_values is not None:
            rbt_s.fk("arm", IO_current_jnt_values)
            rbt_x.move_jnts(component_name="arm", jnt_values=IO_current_jnt_values)
        else:
            print("Init Operation Failed")
            break

        for item in onscreen:
            item.detach()
        onscreen.clear()
        onscreen.append(rbt_s.gen_meshmodel())
        onscreen[-1].attach_to(base)
        # f = open('init_operate_rotmat.txt', 'a', encoding='UTF-8')
        # f.write(str(init_rot + phase_displacement*i))
        # f.write('\n\n')
        # f.close()
        time.sleep(0.01)

# 実機動作用関数
def operate(rbt_s, rbt_x, onscreen, task):
    global data_array, current_jnt_values_rgt, pre_jnt_values_rgt, fail_IK_flag, operation_count, init_error_rgt

    for item in onscreen:
        item.detach()
    onscreen.clear()
    for item in onscreen_tcpframe:
        item.detach()
    onscreen_tcpframe.clear()

    # data_array = data_adjustment(data_list)
    if data_adjustment(data_list) == "No Data":
        return task.cont
    else:
        data_array = data_adjustment(data_list)

    if operation_count == 0:
        tcp_pos, tcp_rot = rbt_s.manipulator_dict['arm'].jlc.get_gl_tcp()
        init_error = data_array[rgt_hand_num:(rgt_hand_num+3)] - tcp_pos
        pre_jnt_values = rbt_s.get_jnt_values(component_name="arm")
    else:
        tcp_pos, tcp_rot = rbt_s.manipulator_dict['arm'].jlc.get_gl_tcp()
        onscreen_tcpframe.append(gm.gen_frame(pos=tcp_pos, rotmat=tcp_rot, length=0.3))
        onscreen_tcpframe[-1].attach_to(base)

        # 制御処理
        current_jnt_values = rbt_s.manipulator_dict['arm'].jlc._ikt.num_ik(tgt_pos=data_array[rgt_hand_num:(rgt_hand_num + 3)] - init_error,
                                                                           tgt_rot=rm.rotmat_from_quaternion(data_array[(rgt_hand_num+6):(rgt_hand_num+10)])[:3, :3],
                                                                           seed_jnt_values=pre_jnt_values,
                                                                           tcp_jntid=7)
        pre_jnt_values = current_jnt_values

        # 衝突判定
        # collided_result = rbt_s.is_collided()
        # if collided_result == True:
        #     print('Collided!!')

        if current_jnt_values is not None:
            rbt_s.fk("arm", current_jnt_values)
            rbt_x.move_jnts(component_name="arm", jnt_values=current_jnt_values)

        # ロボットモデルを生成して描画
        onscreen.append(rbt_s.gen_meshmodel())
        onscreen[-1].attach_to(base)

    operation_count = operation_count + 1
    return task.cont

if __name__ == '__main__':
    sim_init_pos, sim_init_rot = rbt_s.manipulator_dict['arm'].jlc.get_gl_tcp()
    specified_init_pos = sim_init_pos
    specified_init_rot = np.array([[0, -1, 0],
                                   [-1, 0, 0],
                                   [0, 0, -1]])
    real_init_jnt_values = rbt_x.get_jnt_values()
    rbt_s.fk(jnt_values=real_init_jnt_values)
    real_init_pos, real_init_rot = rbt_s.manipulator_dict['arm'].jlc.get_gl_tcp()
    init_operation(init_pos=real_init_pos, init_rot=real_init_rot, specified_init_pos=specified_init_pos, specified_init_rot=specified_init_rot)

    threading.Thread(target=data_collect, args=(data_list,)).start()
    taskMgr.doMethodLater(update_frequency, operate, "operate",
                          extraArgs=[rbt_s, rbt_x, onscreen],
                          appendTask=True)
    gm.gen_frame(length=1, thickness=0.015).attach_to(base)
    base.run()
