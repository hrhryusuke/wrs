"""
---generate_actual_init_jnt_values概要説明---
Xarmの任意の初期姿勢における関節角度を生成，txtファイルに出力する

---備考---
robot_actual_test_v6以降のバージョンでは，このプログラムで関節角度情報を生成するだけで初期姿勢の変更が可能
"""

import numpy as np
import visualization.panda.world as wd
import robot_sim.robots.xarm7_shuidi_mobile.xarm7_shuidi_mobile as xarm

base = wd.World(cam_pos=[6.5, -6.5, 2], lookat_pos=[0, 0, .5]) # モデリング空間を生成
rbt_s = xarm.XArm7YunjiMobile(enable_cc=True)

if __name__ == '__main__':
    # rbt_s.gen_meshmodel().attach_to(base)
    sim_init_pos, sim_init_rot = rbt_s.manipulator_dict['arm'].jlc.get_gl_tcp()

    # 初期位置および姿勢の指定
    specified_init_pos = sim_init_pos
    specified_init_rot = np.array([[0, -1, 0],
                                   [-1, 0, 0],
                                   [0, 0, -1]])
    specified_init_jnt_values = rbt_s.ik(tgt_pos=specified_init_pos, tgt_rotmat=specified_init_rot)
    rbt_s.fk(jnt_values=specified_init_jnt_values)

    print(specified_init_jnt_values)
    f = open('init_jnt_values_lft.txt', 'w')
    for i in range(specified_init_jnt_values.shape[0]):
        f.write(f'{specified_init_jnt_values[i]}')
        if i != specified_init_jnt_values.shape[0]-1:
            f.write(' ')
    f.close()

    rbt_s.gen_meshmodel().attach_to(base)
    base.run()
