import pickle
from simulator.simulator import Simulator
from planner import Planner
from policy import *
from ilp_policy import UnrelatedParallelMachinesSchedulingPolicy
from task_execution_time import ExecutionTimeModel

import multiprocessing
import time


prediction_model = ExecutionTimeModel()
warm_up_policy = RandomPolicy()
warm_up_time =  24#*365*3
simulation_time = warm_up_time

#policy = HungarianMultiObjectivePolicy(1, 0, 0, delta)
my_planner = Planner(prediction_model,
                    warm_up_policy, warm_up_time,
                    warm_up_policy,
                    predict_multiple=True,
                    hour_timeout=120,
                    debug=True)

simulator = Simulator(my_planner)
simulator_result = simulator.run(simulation_time)


with open('prediction_model.pkl', 'wb') as file:
    pickle.dump(prediction_model, file)

print(simulator_result)