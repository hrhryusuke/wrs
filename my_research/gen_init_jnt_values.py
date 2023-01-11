"""
---gen_init_jnt_values概要説明---
xarmの任意の初期姿勢における関節角度を生成，txtファイルに出力する

---備考---

---更新日---
20220901
"""

import numpy as np
import visualization.panda.world as wd
import robot_sim.robots.xarm_shuidi.xarm_shuidi as xarm

base = wd.World(cam_pos=[5, -5, 2], lookat_pos=[0, 0, .5]) # モデリング空間を生成
rbt_s = xarm.XArmShuidi(enable_cc=True)

if __name__ == '__main__':
    sim_init_pos, sim_init_rot = rbt_s.get_gl_tcp()

    # 初期位置および姿勢の指定
    specified_init_pos = sim_init_pos
    specified_init_pos[2] = sim_init_pos[2] - 0.2
    specified_init_rot = np.array([[0, -1, 0],
                                   [-1, 0, 0],
                                   [0, 0, -1]])
    specified_init_jnt_values = rbt_s.ik(tgt_pos=specified_init_pos, tgt_rotmat=specified_init_rot)

    rbt_s.fk(jnt_values=specified_init_jnt_values)
    rbt_s.gen_meshmodel().attach_to(base)
    print(specified_init_jnt_values)

    np.savetxt("init_jnt_values.txt", specified_init_jnt_values)

    base.run()
