import os
import sys
import optparse
import traci

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  

def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options

if __name__ == "__main__":
    options = get_options()   
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    traci.start([sumoBinary, "-c", "test2.sumocfg"])
    step_list = []
    veh_num = []
    lane_list = []
    for step in range(0,7200):
        traci.simulationStep()
        print("step:",step)
        step_list.append(step+1)
        veh_num = traci.vehicle.getIDCount()
        lane_list = traci.lane.getIDList()
        for lane_id in lane_list:
            if "-E0_0" in lane_id:
              E0_0 = lane_id
              print("-E0_0",E0_0)
              print("排队长度：",traci.lane.getLastStepLength(E0_0))
              print("平均车速：",traci.lane.getLastStepMeanSpeed(E0_0))
            if "-E0_1" in lane_id:
              E0_1 = lane_id
              print("-E0_1",E0_1)
              print("排队长度：",traci.lane.getLastStepLength(E0_1))
              print("平均车速：",traci.lane.getLastStepMeanSpeed(E0_1))
           

        traci.simulationStep()
    traci.close()
