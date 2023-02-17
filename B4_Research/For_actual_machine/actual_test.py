import numpy as np
import visualization.panda.world as wd
import robot_sim.robots.xarm7_shuidi_mobile.xarm7_shuidi_mobile as xarm

# base = wd.World(cam_pos=[6.5, -6.5, 2], lookat_pos=[0, 0, .5]) # モデリング空間を生成
# rbt_s = xarm.XArm7YunjiMobile(enable_cc=True)

if __name__ == '__main__':
    data = np.loadtxt('init_jnt_values_rgt.txt')
    print(data)
    print(type(data))
