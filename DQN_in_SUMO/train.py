import gymnasium as gym
import traci
import numpy as np
import time as tm
import pandas as pd
import torch
import random
from torch.utils.tensorboard import SummaryWriter
from dqn import Agent
import save_and_load as sl

# 设置结果保存路径(在output_model文件夹中以当前时间为文件名生成一个文件夹)
saved_directory = sl.assign_train_directory("output_model")
# 保存的每个episode的平均奖励
reward_data = pd.DataFrame(columns=["Episode", "Average_Reward"])
# 设置记录器，可以实时查看reward曲线 (具体做法：运行该程序后，在终端命令行输入"tensorboard --logdir=runs")
writer = SummaryWriter()


# 定义训练的总轮数和每轮运行的仿真步数
NUM_EPISODES = 200
NUM_STEPS = 500

# 配置仿真环境，获取状态和动作维数
# ********************************************************
env = gym.make("CartPole-v1")
state, info = env.reset()
s_dim = len(state)
a_dim = env.action_space.n
# ********************************************************


# 初始化agent (所有超参数都在这里修改, 具体含义见"dqn.py")
agent = Agent(s_dim, a_dim, gamma=0.99, memory_capacity=20000, batch_size=256, learning_rate=0.001,
              tau=0.005, epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=1000)

def _get_queuelength(self):
     intersection_queue = 0
     halt_N =traci.lane.getLastStepLength("-E1_0")+traci.lane.getLastStepLength("-E1_1")+traci.lane.getLastStepLength("-E1_2")+traci.lane.getLastStepLength("-E1_3") 
     halt_S = traci.lane.getLastStepLength("-E3_0")+traci.lane.getLastStepLength("-E3_1")+traci.lane.getLastStepLength("-E3_2")+traci.lane.getLastStepLength("-E3_3") 
     halt_E = traci.lane.getLastStepLength("-E2_0")+traci.lane.getLastStepLength("-E2_1")+traci.lane.getLastStepLength("-E2_2")+traci.lane.getLastStepLength("-E2_3") 
     halt_W = traci.lane.getLastStepLength("-E0_0")+traci.lane.getLastStepLength("-E0_1")+traci.lane.getLastStepLength("-E0_2")+traci.lane.getLastStepLength("-E0_3") 
     intersection_queue = halt_N + halt_S + halt_E + halt_W
     return intersection_queue

def _Set_YellowPhase(self, old_action):
    if old_action == 0 or old_action == 1 or old_action == 2:
          yellow_phase = 1
    elif old_action == 3 or old_action == 4 or old_action == 5:
          yellow_phase = 3
    elif old_action == 6 or old_action == 7 or old_action == 8:
          yellow_phase = 5
    elif old_action == 9 or old_action == 10 or old_action == 11:
          yellow_phase = 7
    traci.trafficlight.setPhase("0",yellow_phase)
        
         
def _Set_GreenPhaseandDuration(self, action):
    if action == 0:
        traci.trafficlight.setPhase("0", 0)
        traci.trafficlight.setPhaseDuration("0", 15)
        self._green_duration = 15
    elif action == 1:
        traci.trafficlight.setPhase("0", 0)
        traci.trafficlight.setPhaseDuration("0", 10)
        self._green_duration = 10
    elif action == 2:
        traci.trafficlight.setPhase("0", 0)
        traci.trafficlight.setPhaseDuration("0", 20)
        self._green_duration = 20
    elif action == 3:
        traci.trafficlight.setPhase("0", 2)
        traci.trafficlight.setPhaseDuration("0", 15)
        self._green_duration = 15
    elif action == 4:
        traci.trafficlight.setPhase("0", 2)
        traci.trafficlight.setPhaseDuration("0", 10)
        self._green_duration = 10
    elif action == 5:
        traci.trafficlight.setPhase("0", 2)
        traci.trafficlight.setPhaseDuration("0", 20)
        self._green_duration = 20
    elif action == 5:
        traci.trafficlight.setPhase("0", 4)
        traci.trafficlight.setPhaseDuration("0", 15)
        self._green_duration = 15
    elif action == 7:
        traci.trafficlight.setPhase("0", 4)
        traci.trafficlight.setPhaseDuration("0", 10)
        self._green_duration = 10
    elif action == 8:
        traci.trafficlight.setPhase("0", 4)
        traci.trafficlight.setPhaseDuration("0", 20)
        self._green_duration = 20
    elif action == 9:
        traci.trafficlight.setPhase("0", 6)
        traci.trafficlight.setPhaseDuration("0", 15)
        self._green_duration = 15
    elif action == 10:
        traci.trafficlight.setPhase("0", 6)
        traci.trafficlight.setPhaseDuration("0", 10)
        self._green_duration = 10
    elif action == 11:
        traci.trafficlight.setPhase("0", 6)
        traci.trafficlight.setPhaseDuration("0", 20)
        self._green_duration = 20

def get_custom_state(state):
    Position_Matrix = []
    Velocity_Matrix = []
    for i in range(16):
            Position_Matrix.append([])
            Velocity_Matrix.append([])
            for j in range(3):
                Position_Matrix[i].append(0)
                Velocity_Matrix[i].append(0)
    Position_Matrix = np.array(Position_Matrix)
    Velocity_Matrix = np.array(Velocity_Matrix)

    Loop1i_0_1 = traci.inductionloop.getLastStepVehicleIDs("Loop0i_0_1" )
    Loop1i_0_2 = traci.inductionloop.getLastStepVehicleIDs("Loop0i_0_2" )
    Loop1i_0_3 = traci.inductionloop.getLastStepVehicleIDs("Loop0i_0_3" )
    Loop1i_1_1 = traci.inductionloop.getLastStepVehicleIDs("Loop0i_1_1" )
    Loop1i_1_2 = traci.inductionloop.getLastStepVehicleIDs("Loop0i_1_2" )
    Loop1i_1_3 = traci.inductionloop.getLastStepVehicleIDs("Loop0i_1_3" )
    Loop1i_2_1 = traci.inductionloop.getLastStepVehicleIDs("Loop0i_2_1" )
    Loop1i_2_2 = traci.inductionloop.getLastStepVehicleIDs("Loop0i_2_2" )
    Loop1i_2_3 = traci.inductionloop.getLastStepVehicleIDs("Loop0i_2_3" )
    Loop1i_3_1 = traci.inductionloop.getLastStepVehicleIDs("Loop0i_3_1" )
    Loop1i_3_2 = traci.inductionloop.getLastStepVehicleIDs("Loop0i_3_2" )
    Loop1i_3_3 = traci.inductionloop.getLastStepVehicleIDs("Loop0i_3_3" )


    if len(Loop1i_0_1) != 0:
       Velocity_Matrix[0,0] = traci.vehicle.getSpeed(Loop1i_0_1[0])
       Loop1i_0_1 = 1
    else:
       Loop1i_0_1 = 0
       
    if len(Loop1i_0_2) != 0:
       Velocity_Matrix[0,1] = traci.vehicle.getSpeed(Loop1i_0_2[0])
       Loop1i_0_2 = 1
    else:
       Loop1i_0_2 = 0
       
    if len(Loop1i_0_3) != 0:
       Velocity_Matrix[0,2] = traci.vehicle.getSpeed(Loop1i_0_3[0])
       Loop1i_0_3 = 1
    else:
       Loop1i_0_3 = 0   
       
    if len(Loop1i_1_1) != 0:
       Velocity_Matrix[1,0] = traci.vehicle.getSpeed(Loop1i_1_1[0])
       Loop1i_1_1 = 1
    else:
       Loop1i_1_1 = 0 
       
    if len(Loop1i_1_2) != 0:
       Velocity_Matrix[1,1] = traci.vehicle.getSpeed(Loop1i_1_2[0])
       Loop1i_1_2 = 1
    else:
       Loop1i_1_2 = 0 
     
    if len(Loop1i_1_3) != 0:
       Velocity_Matrix[1,2] = traci.vehicle.getSpeed(Loop1i_1_3[0])
       Loop1i_1_3 = 1
    else:
       Loop1i_1_3 = 0 
    
    if len(Loop1i_2_1) != 0:
       Velocity_Matrix[2,0] = traci.vehicle.getSpeed(Loop1i_2_1[0])
       Loop1i_2_1 = 1
    else:
       Loop1i_2_1 = 0 
       
    if len(Loop1i_2_2) != 0:
       Velocity_Matrix[2,1] = traci.vehicle.getSpeed(Loop1i_2_2[0])
       Loop1i_2_2 = 1
    else:
       Loop1i_2_2 = 0 
     
    if len(Loop1i_2_3) != 0:
       Velocity_Matrix[2,2] = traci.vehicle.getSpeed(Loop1i_2_3[0])
       Loop1i_2_3 = 1
    else:
       Loop1i_2_3 = 0 
      
    if len(Loop1i_3_1) != 0:
       Velocity_Matrix[3,0] = traci.vehicle.getSpeed(Loop1i_3_1[0])
       Loop1i_3_1 = 1
    else:
       Loop1i_3_1 = 0 
           
    if len(Loop1i_3_2) != 0:
       Velocity_Matrix[3,1] = traci.vehicle.getSpeed(Loop1i_3_2[0])
       Loop1i_3_2 = 1
    else:
       Loop1i_3_2 = 0 
         
    if len(Loop1i_3_3) != 0:
       Velocity_Matrix[3,2] = traci.vehicle.getSpeed(Loop1i_2_3[0])
       Loop1i_2_3 = 1
    else:
       Loop1i_2_3 = 0 
   
    Position_Matrix[0,0] = Loop1i_0_1
    Position_Matrix[0,1] = Loop1i_0_2
    Position_Matrix[0,2] = Loop1i_0_3
    Position_Matrix[1,0] = Loop1i_1_1
    Position_Matrix[1,1] = Loop1i_1_2
    Position_Matrix[1,2] = Loop1i_1_3
    Position_Matrix[2,0] = Loop1i_2_1
    Position_Matrix[2,1] = Loop1i_2_2
    Position_Matrix[2,2] = Loop1i_2_3
    Position_Matrix[3,0] = Loop1i_3_1
    Position_Matrix[3,1] = Loop1i_3_2
    Position_Matrix[3,2] = Loop1i_3_3
    
    Loop2i_0_1 = traci.inductionloop.getLastStepVehicleIDs("Loop1i_0_1" )
    Loop2i_0_2 = traci.inductionloop.getLastStepVehicleIDs("Loop1i_0_2" )
    Loop2i_0_3 = traci.inductionloop.getLastStepVehicleIDs("Loop1i_0_3" )
    Loop2i_1_1 = traci.inductionloop.getLastStepVehicleIDs("Loop1i_1_1" )
    Loop2i_1_2 = traci.inductionloop.getLastStepVehicleIDs("Loop1i_1_2" )
    Loop2i_1_3 = traci.inductionloop.getLastStepVehicleIDs("Loop1i_1_3" )
    Loop2i_2_1 = traci.inductionloop.getLastStepVehicleIDs("Loop1i_2_1" )
    Loop2i_2_2 = traci.inductionloop.getLastStepVehicleIDs("Loop1i_2_2" )
    Loop2i_2_3 = traci.inductionloop.getLastStepVehicleIDs("Loop1i_2_3" )
    Loop2i_3_1 = traci.inductionloop.getLastStepVehicleIDs("Loop1i_3_1" )
    Loop2i_3_2 = traci.inductionloop.getLastStepVehicleIDs("Loop1i_3_2" )
    Loop2i_3_3 = traci.inductionloop.getLastStepVehicleIDs("Loop1i_3_3" )


    if len(Loop2i_0_1) != 0:
       Velocity_Matrix[4,0] = traci.vehicle.getSpeed(Loop2i_0_1[0])
       Loop2i_0_1 = 1
    else:
       Loop2i_0_1 = 0
       
    if len(Loop2i_0_2) != 0:
       Velocity_Matrix[4,1] = traci.vehicle.getSpeed(Loop2i_0_2[0])
       Loop2i_0_2 = 1
    else:
       Loop2i_0_2 = 0
       
    if len(Loop2i_0_3) != 0:
       Velocity_Matrix[4,2] = traci.vehicle.getSpeed(Loop2i_0_3[0])
       Loop2i_0_3 = 1
    else:
       Loop2i_0_3 = 0   
       
    if len(Loop2i_1_1) != 0:
       Velocity_Matrix[5,0] = traci.vehicle.getSpeed(Loop2i_1_1[0])
       Loop2i_1_1 = 1
    else:
       Loop2i_1_1 = 0 
       
    if len(Loop2i_1_2) != 0:
       Velocity_Matrix[5,1] = traci.vehicle.getSpeed(Loop2i_1_2[0])
       Loop2i_1_2 = 1
    else:
       Loop2i_1_2 = 0 
     
    if len(Loop2i_1_3) != 0:
       Velocity_Matrix[5,2] = traci.vehicle.getSpeed(Loop2i_1_3[0])
       Loop2i_1_3 = 1
    else:
       Loop2i_1_3 = 0 
    
    if len(Loop2i_2_1) != 0:
       Velocity_Matrix[6,0] = traci.vehicle.getSpeed(Loop2i_2_1[0])
       Loop2i_2_1 = 1
    else:
       Loop2i_2_1 = 0 
       
    if len(Loop2i_2_2) != 0:
       Velocity_Matrix[6,1] = traci.vehicle.getSpeed(Loop2i_2_2[0])
       Loop2i_2_2 = 1
    else:
       Loop2i_2_2 = 0 
     
    if len(Loop2i_2_3) != 0:
       Velocity_Matrix[6,2] = traci.vehicle.getSpeed(Loop2i_2_3[0])
       Loop2i_2_3 = 1
    else:
       Loop2i_2_3 = 0 
      
    if len(Loop2i_3_1) != 0:
       Velocity_Matrix[7,0] = traci.vehicle.getSpeed(Loop2i_3_1[0])
       Loop2i_3_1 = 1
    else:
       Loop2i_3_1 = 0 
           
    if len(Loop2i_3_2) != 0:
       Velocity_Matrix[7,1] = traci.vehicle.getSpeed(Loop2i_3_2[0])
       Loop2i_3_2 = 1
    else:
       Loop2i_3_2 = 0 
         
    if len(Loop2i_3_3) != 0:
       Velocity_Matrix[7,2] = traci.vehicle.getSpeed(Loop2i_2_3[0])
       Loop2i_2_3 = 1
    else:
       Loop2i_2_3 = 0 
   
    Position_Matrix[4,0] = Loop2i_0_1
    Position_Matrix[4,1] = Loop2i_0_2
    Position_Matrix[4,2] = Loop2i_0_3
    Position_Matrix[5,0] = Loop2i_1_1
    Position_Matrix[5,1] = Loop2i_1_2
    Position_Matrix[5,2] = Loop2i_1_3
    Position_Matrix[6,0] = Loop2i_2_1
    Position_Matrix[6,1] = Loop2i_2_2
    Position_Matrix[6,2] = Loop2i_2_3
    Position_Matrix[7,0] = Loop2i_3_1
    Position_Matrix[7,1] = Loop2i_3_2
    Position_Matrix[7,2] = Loop2i_3_3
    
    Loop3i_0_1 = traci.inductionloop.getLastStepVehicleIDs("Loop2i_0_1" )
    Loop3i_0_2 = traci.inductionloop.getLastStepVehicleIDs("Loop2i_0_2" )
    Loop3i_0_3 = traci.inductionloop.getLastStepVehicleIDs("Loop2i_0_3" )
    Loop3i_1_1 = traci.inductionloop.getLastStepVehicleIDs("Loop2i_1_1" )
    Loop3i_1_2 = traci.inductionloop.getLastStepVehicleIDs("Loop2i_1_2" )
    Loop3i_1_3 = traci.inductionloop.getLastStepVehicleIDs("Loop2i_1_3" )
    Loop3i_2_1 = traci.inductionloop.getLastStepVehicleIDs("Loop2i_2_1" )
    Loop3i_2_2 = traci.inductionloop.getLastStepVehicleIDs("Loop2i_2_2" )
    Loop3i_2_3 = traci.inductionloop.getLastStepVehicleIDs("Loop2i_2_3" )
    Loop3i_3_1 = traci.inductionloop.getLastStepVehicleIDs("Loop2i_3_1" )
    Loop3i_3_2 = traci.inductionloop.getLastStepVehicleIDs("Loop2i_3_2" )
    Loop3i_3_3 = traci.inductionloop.getLastStepVehicleIDs("Loop2i_3_3" )


    if len(Loop3i_0_1) != 0:
       Velocity_Matrix[8,0] = traci.vehicle.getSpeed(Loop3i_0_1[0])
       Loop3i_0_1 = 1
    else:
       Loop3i_0_1 = 0
       
    if len(Loop3i_0_2) != 0:
       Velocity_Matrix[8,1] = traci.vehicle.getSpeed(Loop3i_0_2[0])
       Loop3i_0_2 = 1
    else:
       Loop3i_0_2 = 0
       
    if len(Loop3i_0_3) != 0:
       Velocity_Matrix[8,2] = traci.vehicle.getSpeed(Loop3i_0_3[0])
       Loop3i_0_3 = 1
    else:
       Loop3i_0_3 = 0   
       
    if len(Loop3i_1_1) != 0:
       Velocity_Matrix[9,0] = traci.vehicle.getSpeed(Loop3i_1_1[0])
       Loop3i_1_1 = 1
    else:
       Loop3i_1_1 = 0 
       
    if len(Loop3i_1_2) != 0:
       Velocity_Matrix[9,1] = traci.vehicle.getSpeed(Loop3i_1_2[0])
       Loop3i_1_2 = 1
    else:
       Loop3i_1_2 = 0 
     
    if len(Loop3i_1_3) != 0:
       Velocity_Matrix[9,2] = traci.vehicle.getSpeed(Loop3i_1_3[0])
       Loop3i_1_3 = 1
    else:
       Loop3i_1_3 = 0 
    
    if len(Loop3i_2_1) != 0:
       Velocity_Matrix[10,0] = traci.vehicle.getSpeed(Loop3i_2_1[0])
       Loop3i_2_1 = 1
    else:
       Loop3i_2_1 = 0 
       
    if len(Loop3i_2_2) != 0:
       Velocity_Matrix[10,1] = traci.vehicle.getSpeed(Loop3i_2_2[0])
       Loop3i_2_2 = 1
    else:
       Loop3i_2_2 = 0 
     
    if len(Loop3i_2_3) != 0:
       Velocity_Matrix[10,2] = traci.vehicle.getSpeed(Loop3i_2_3[0])
       Loop3i_2_3 = 1
    else:
       Loop3i_2_3 = 0 
      
    if len(Loop3i_3_1) != 0:
       Velocity_Matrix[11,0] = traci.vehicle.getSpeed(Loop3i_3_1[0])
       Loop3i_3_1 = 1
    else:
       Loop3i_3_1 = 0 
           
    if len(Loop3i_3_2) != 0:
       Velocity_Matrix[11,1] = traci.vehicle.getSpeed(Loop3i_3_2[0])
       Loop3i_3_2 = 1
    else:
       Loop3i_3_2 = 0 
         
    if len(Loop3i_3_3) != 0:
       Velocity_Matrix[11,2] = traci.vehicle.getSpeed(Loop3i_2_3[0])
       Loop3i_2_3 = 1
    else:
       Loop3i_2_3 = 0 
   
    Position_Matrix[8,0] = Loop3i_0_1
    Position_Matrix[8,1] = Loop3i_0_2
    Position_Matrix[8,2] = Loop3i_0_3
    Position_Matrix[9,0] = Loop3i_1_1
    Position_Matrix[9,1] = Loop3i_1_2
    Position_Matrix[9,2] = Loop3i_1_3
    Position_Matrix[10,0] = Loop3i_2_1
    Position_Matrix[10,1] = Loop3i_2_2
    Position_Matrix[10,2] = Loop3i_2_3
    Position_Matrix[11,0] = Loop3i_3_1
    Position_Matrix[11,1] = Loop3i_3_2
    Position_Matrix[11,2] = Loop3i_3_3
    
    Loop4i_0_1 = traci.inductionloop.getLastStepVehicleIDs("Loop3i_0_1" )
    Loop4i_0_2 = traci.inductionloop.getLastStepVehicleIDs("Loop3i_0_2" )
    Loop4i_0_3 = traci.inductionloop.getLastStepVehicleIDs("Loop3i_0_3" )
    Loop4i_1_1 = traci.inductionloop.getLastStepVehicleIDs("Loop3i_1_1" )
    Loop4i_1_2 = traci.inductionloop.getLastStepVehicleIDs("Loop3i_1_2" )
    Loop4i_1_3 = traci.inductionloop.getLastStepVehicleIDs("Loop3i_1_3" )
    Loop4i_2_1 = traci.inductionloop.getLastStepVehicleIDs("Loop3i_2_1" )
    Loop4i_2_2 = traci.inductionloop.getLastStepVehicleIDs("Loop3i_2_2" )
    Loop4i_2_3 = traci.inductionloop.getLastStepVehicleIDs("Loop3i_2_3" )
    Loop4i_3_1 = traci.inductionloop.getLastStepVehicleIDs("Loop3i_3_1" )
    Loop4i_3_2 = traci.inductionloop.getLastStepVehicleIDs("Loop3i_3_2" )
    Loop4i_3_3 = traci.inductionloop.getLastStepVehicleIDs("Loop3i_3_3" )


    if len(Loop4i_0_1) != 0:
       Velocity_Matrix[12,0] = traci.vehicle.getSpeed(Loop4i_0_1[0])
       Loop4i_0_1 = 1
    else:
       Loop4i_0_1 = 0
       
    if len(Loop4i_0_2) != 0:
       Velocity_Matrix[12,1] = traci.vehicle.getSpeed(Loop4i_0_2[0])
       Loop4i_0_2 = 1
    else:
       Loop4i_0_2 = 0
       
    if len(Loop4i_0_3) != 0:
       Velocity_Matrix[12,2] = traci.vehicle.getSpeed(Loop4i_0_3[0])
       Loop4i_0_3 = 1
    else:
       Loop4i_0_3 = 0   
       
    if len(Loop4i_1_1) != 0:
       Velocity_Matrix[13,0] = traci.vehicle.getSpeed(Loop4i_1_1[0])
       Loop4i_1_1 = 1
    else:
       Loop4i_1_1 = 0 
       
    if len(Loop4i_1_2) != 0:
       Velocity_Matrix[13,1] = traci.vehicle.getSpeed(Loop4i_1_2[0])
       Loop4i_1_2 = 1
    else:
       Loop4i_1_2 = 0 
     
    if len(Loop4i_1_3) != 0:
       Velocity_Matrix[13,2] = traci.vehicle.getSpeed(Loop4i_1_3[0])
       Loop4i_1_3 = 1
    else:
       Loop4i_1_3 = 0 
    
    if len(Loop4i_2_1) != 0:
       Velocity_Matrix[14,0] = traci.vehicle.getSpeed(Loop4i_2_1[0])
       Loop4i_2_1 = 1
    else:
       Loop4i_2_1 = 0 
       
    if len(Loop4i_2_2) != 0:
       Velocity_Matrix[14,1] = traci.vehicle.getSpeed(Loop4i_2_2[0])
       Loop4i_2_2 = 1
    else:
       Loop4i_2_2 = 0 
     
    if len(Loop4i_2_3) != 0:
       Velocity_Matrix[14,2] = traci.vehicle.getSpeed(Loop4i_2_3[0])
       Loop4i_2_3 = 1
    else:
       Loop4i_2_3 = 0 
      
    if len(Loop4i_3_1) != 0:
       Velocity_Matrix[15,0] = traci.vehicle.getSpeed(Loop4i_3_1[0])
       Loop4i_3_1 = 1
    else:
       Loop4i_3_1 = 0 
           
    if len(Loop4i_3_2) != 0:
       Velocity_Matrix[15,1] = traci.vehicle.getSpeed(Loop4i_3_2[0])
       Loop4i_3_2 = 1
    else:
       Loop4i_3_2 = 0 
         
    if len(Loop4i_3_3) != 0:
       Velocity_Matrix[15,2] = traci.vehicle.getSpeed(Loop4i_2_3[0])
       Loop4i_2_3 = 1
    else:
       Loop4i_2_3 = 0 
   
    Position_Matrix[12,0] = Loop4i_0_1
    Position_Matrix[12,1] = Loop4i_0_2
    Position_Matrix[12,2] = Loop4i_0_3
    Position_Matrix[13,0] = Loop4i_1_1
    Position_Matrix[13,1] = Loop4i_1_2
    Position_Matrix[13,2] = Loop4i_1_3
    Position_Matrix[14,0] = Loop4i_2_1
    Position_Matrix[14,1] = Loop4i_2_2
    Position_Matrix[14,2] = Loop4i_2_3
    Position_Matrix[15,0] = Loop4i_3_1
    Position_Matrix[15,1] = Loop4i_3_2
    Position_Matrix[15,2] = Loop4i_3_3
   
    

    
    #Create 4 x 1 matrix for phase state
    Phase = []
    if traci.trafficlight.getPhase('0') == 0 or traci.trafficlight.getPhase('0') == 1 or traci.trafficlight.getPhase('0') == 2:
        Phase = [1, 0, 0, 0]
    elif traci.trafficlight.getPhase('0') == 3 or traci.trafficlight.getPhase('0') == 4 or traci.trafficlight.getPhase('0') == 5:
        Phase = [0, 1, 0, 0]
    elif traci.trafficlight.getPhase('0') == 6 or traci.trafficlight.getPhase('0') == 7 or traci.trafficlight.getPhase('0') == 8:
        Phase = [0, 0, 1, 0]
    elif traci.trafficlight.getPhase('0') == 9 or traci.trafficlight.getPhase('0') == 10 or traci.trafficlight.getPhase('0') == 11:
        Phase = [0, 0, 0, 1]
  
    Phase = np.array(Phase)
    Phase = Phase.flatten()
    
    state = np.concatenate((Position_Matrix,Velocity_Matrix), axis=0)
    state = state.flatten()
    state =  np.concatenate((state,Phase), axis=0)
    
    #Create matrix for duration
    Duration_Matrix = [traci.trafficlight.getPhaseDuration('0')]

    Duration_Matrix = np.array(Duration_Matrix)
    Duration_Matrix = Duration_Matrix.flatten()
    state =  np.concatenate((state,Duration_Matrix), axis=0)
   

    return state 

def get_custom_reward(state, action, reward, next_state):


start = tm.time()  # 记录程序开始时间
print("开始训练 ...")


for i_episode in range(NUM_EPISODES):
    total_reward = 0  # 统计当前episode的总奖励
    state, _ = env.reset()  # 在每一轮训练开始时初始化环境，并获取本轮的初始state (该state必须是一维的numpy.array)
    state = get_custom_state(state) 
    state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)  # 将状态变量转为torch.tensor

    step = 1
    for t in range(NUM_STEPS):
        action = agent.choose_action(state)  # 该action为torch.tensor类型
        
  
        # 此处需要替换成自己的代码，包括将动作值输入进仿真，获取奖励值，并判断该episode是否完成
        # *********************************************************************************************************
        observation, reward, terminated, truncated, _ = env.step(action.item())

        total_reward += reward

        # action.item()将torch.tensor转为一个int，将该数据输入环境
        reward = torch.tensor([reward], device=agent.device)  # 将获得的奖励转为torch.tensor
        done = terminated or truncated  # 当前episode是否结束
        # *********************************************************************************************************


        if terminated:
            next_state = None
        else:
            # 如果该episode未结束，需要在仿真更新一步后获取最新的状态
            # *************************************************************************************************
            # traci.simulationStep()
            # 如果当前episode未结束，将next_state转为torch.tensor
            next_state = torch.tensor(observation, dtype=torch.float32, device=agent.device).unsqueeze(0)
            # *************************************************************************************************

        # 存储SARS，前三个量都是torch.tensor，next_state可能是torch.tensor，也可能是None
        agent.store_transition(state, action, reward, next_state)

        # 将下一状态转为当前状态，以备循环下一个step时使用
        state = next_state
        # agent训练一次
        agent.learn()
        # 如果episode结束，就开始下一次episode
        step += 1
        if done:
            break
    average_reward = total_reward / step
    writer.add_scalar("Average Reward", average_reward, i_episode + 1)
    print(f"第{i_episode + 1}轮, 平均奖励: {average_reward: .2f}, "
          f"经验池[{len(agent.memory)}/{agent.memory_capacity}]")
    new_reward_data = pd.DataFrame([[i_episode + 1, average_reward]], columns=["Episode", "Average_Reward"])
    reward_data = pd.concat([reward_data, new_reward_data])

# 保存模型和结果
sl.save_model(agent, reward_data, saved_directory)
writer.close()

print("\n训练完成，", end="")

end = tm.time()  # 记录程序结束时间
# 统计程序运行时间
duration = round(end - start)
if duration < 60:
    print(f"总运行时间: {duration:.0f}s")
elif duration < 3600:
    print(f"总运行时间: {duration // 60}m {duration % 60:.0f}s")
else:
    print(f"总运行时间: {duration // 3600}h {(duration % 3600) // 60}m {duration % 60:.0f}s")