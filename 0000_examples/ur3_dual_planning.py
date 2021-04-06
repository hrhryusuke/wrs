import math
import numpy as np
import visualization.panda.world as wd
import modeling.geometricmodel as gm
import modeling.collisionmodel as cm
import robotsim.robots.ur3_dual.ur3_dual as ur3d
import motion.probabilistic.rrt_connect as rrtc

base = wd.World(campos=[2, 1, 3], lookatpos=[0, 0, 1.1])
gm.gen_frame().attach_to(base)
# object
object = cm.CollisionModel("./objects/bunnysim.stl")
object.set_pos(np.array([.55, -.3, 1.3]))
object.set_rgba([.5, .7, .3, 1])
object.attach_to(base)
# robot
component_name = 'lft_arm'
robot_instance = ur3d.UR3Dual()

# possible right goal np.array([0, -math.pi/4, 0, math.pi/2, math.pi/2, math.pi / 6])
# possible left goal np.array([0, -math.pi / 2, -math.pi/3, -math.pi / 2, math.pi / 6, math.pi / 6])

rrtc_planner = rrtc.RRTConnect(robot_instance)
path = rrtc_planner.plan(start_conf=robot_instance.lft_arm.homeconf,
                         goal_conf=np.array([0, -math.pi / 2, -math.pi/3, -math.pi / 2, math.pi / 6, math.pi / 6]),
                         obstacle_list=[object],
                         ext_dist=.2,
                         rand_rate=70,
                         maxtime=300,
                         component_name=component_name)
print(path)
for pose in path:
    print(pose)
    robot_instance.fk(component_name, pose)
    robot_meshmodel = robot_instance.gen_meshmodel()
    robot_meshmodel.attach_to(base)
    robot_instance.gen_stickmodel().attach_to(base)

base.run()
