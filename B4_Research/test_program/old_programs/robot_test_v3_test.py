"""
---robot_test_v3概要説明---
Axis NEURONから送信されてきたPERCEPTION NEURONの右手の座標位置情報を元にxarmに動きを同期させる

---v3更新箇所---
前の時刻と現在の右手の甲の相対方向に動作する仕様に変更
IK解なしからの復帰時のメッセージ表示を追加

---備考---
"""

import socket
import math
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
# import robot_con.xarm.xarm_client as xac
import robot_sim.robots.xarm7_shuidi_mobile.xarm7_shuidi_mobile as xarm

# ロボットモデルを読み込み
rbt_s = xarm.XArm7YunjiMobile()
# rbt_x = xac.XArm7(host="10.2.0.203:18300")
# init_jnt_angles = rbt_x.get_jnt_vlaues()
# rbt_s.fk("arm", init_jnt_angles)

# NeuronDataReaderを読み込む
NDR_path = os.path.abspath("../../NeuronDataReader.dll")
NDR = cdll.LoadLibrary(NDR_path)

# モデリング空間を生成
base = wd.World(cam_pos=[5, -5, 5], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base) # 座標系の軸を表示

# サーバー側に接続
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((socket.gethostname(), 7001)) # 7001番のポートに接続
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# データの更新頻度(s)(ノード18個未満：120Hz 18個以上：60Hz)
update_frequency = (1/60)

# データ長の設定
main_data_length = 60*16*4
contactRL_length = 4

# 球のgmモデルについての情報を格納するリスト
attached_list = []

# 関数動作回数カウンタ
operation_count = 0

# update内で右手座標の更新を行う頻度(何回に1回右手座標の更新を行うか)
# update_cal_frequency = 1

# ロボット描画情報用配列
onscreen = []

# 前の時刻の右手の甲の位置に関する配列
pre_agv_pos_rgt = np.zeros(3)

# armの位置更新の際の重み
arm_update_weight = 0.01

# データ情報
data_list = [None]

# IK解なし用フラッグ
fail_IK_flag = 0


# データ受信用関数
def data_collect(data_list):
    while True:
        # データ受信
        data_list[0] = s.recv(main_data_length)
        s.recv(2 * contactRL_length)  # contactLRの分を受信だけする（ズレ調整）


# データ抽出および描画
def update(s, attached_list, rbt_s, onscreen, pre_pos, task):
    global operation_count
    global fail_IK_flag
    # # データ受信
    # b_msg = s.recv(main_data_length)
    # s.recv(2 * contactRL_length) # contactLRの分を受信だけする（ズレ調整）

    # onscreen.append(rbt_s.gen_meshmodel())
    # onscreen[-1].attach_to(base)

    # 配列を空にする
    # if len(onscreen) > 0:
    #     for objcm in onscreen:
    #         objcm.remove()

    # モデルを削除
    for item in onscreen:
        item.detach()

    # データ型を変更
    b_msg = data_list[0]
    if b_msg is None:
        return task.cont
    tmp_data_array = np.frombuffer(b_msg, dtype=np.float32)

    # 座標調整
    data_array = np.zeros(int(tmp_data_array.size))
    for i in range(0, tmp_data_array.size):
        if (i % 16) == 0 or (i % 16) == 2:
            data_array[i] = tmp_data_array[i] * (-1)
        else:
            data_array[i] = tmp_data_array[i]
    # data_array = data_array*(-1) # 座標調整
    # print(data_array.size)

    # 右手の甲の位置データ表示およびテキストファイルへの書き込み
    for i in range(0, 16 * 23, 16):
        # if i == 8*16:
        #     attached_list.append(gm.gen_sphere(pos=data_array[i:(i + 3)], radius=1, rgba=[0,1,0,1]))

        # データの数が規定量に達したらbreak
        if i == data_array.size:
            break
        # print(data_array[i:i + 3])

        # 描画処理
        if operation_count == 0:
            # 各部の座標情報を赤色の球として表示
            attached_list.append(gm.gen_sphere(pos=data_array[i:(i + 3)]))
            attached_list[-1].attach_to(base)
            # print(attached_list)

            # 最初の時刻の右手の座標を記録
            pre_pos[:] = data_array[11*16:(11*16+3)]
        else:
            # 球の位置情報を更新
            # npvec3 = data_array[i:(i + 3)]
            # attached_list[int(i / 16)]._objpdnp.setPos(npvec3[0], npvec3[1], npvec3[2])

            # 更新時点での右手の座標をxarmに反映させる
            if i == 11 * 16:
                # 球の位置情報を更新
                npvec3 = data_array[i:(i + 3)]
                attached_list[int(i / 16)]._objpdnp.setPos(npvec3[0], npvec3[1], npvec3[2])
                # 右手の座標を示す球は緑色に設定
                attached_list[int(i / 16)].set_rgba([0, 1, 0, 1])
                # 現在のアームの位置と回転行列を取得
                tcp_pos, tcp_rotmat = rbt_s.manipulator_dict["arm"].get_gl_tcp()
                # 初期の右手の位置との差分を動きに反映させる
                new_tcp_pos = tcp_pos + arm_update_weight * (npvec3 - pre_pos)
                # この時刻の右手の甲の位置を記録
                pre_pos = data_array[i:(i + 3)]
                # 逆運動学によって目標位置におけるジョイントの角度を計算
                jnt_values = rbt_s.ik("arm", new_tcp_pos, tcp_rotmat)
                # 順運動学で先ほどのIKの計算結果を適用（IKが解けないときはFKを更新しない）
                if not(jnt_values is None):
                    rbt_s.fk("arm", jnt_values)
                    if fail_IK_flag == 1:
                        print("-----------------------\nArm operating restart!!\n-----------------------")
                        fail_IK_flag = 0
                else:
                    fail_IK_flag = 1

                # ロボットモデルを生成して描画
                onscreen.append(rbt_s.gen_meshmodel())
                onscreen[-1].attach_to(base)
            # print(attached_list)

        # print(f"{operation_count}")

    operation_count = operation_count + 1
    return task.cont


threading.Thread(target=data_collect, args=(data_list,)).start()

# update_frequency(s)間隔で情報を更新
taskMgr.doMethodLater(update_frequency, update, "update",
                      extraArgs=[s, attached_list, rbt_s, onscreen, pre_agv_pos_rgt],
                      appendTask=True)
base.run()
