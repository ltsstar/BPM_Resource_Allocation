import pickle
from simulator.simulator import Simulator
from planner import Planner
from policy import *
from ilp_policy import UnrelatedParallelMachinesSchedulingPolicy
from task_execution_time import ExecutionTimeModel

import multiprocessing
import time

prediction_model = ExecutionTimeModel()
with open('prediction_model.pkl', 'rb') as file:
    prediction_model = pickle.load(file)

warm_up_policy = RandomPolicy()
warm_up_time =  0
simulation_time = 24*28

def run_simulator(delta):
    policy = HungarianMultiObjectivePolicy(1, 0, 0, delta)
    my_planner = Planner(prediction_model,
                        warm_up_policy, warm_up_time,
                        policy,
                        predict_multiple=True,
                        #hour_timeout=120,
                        debug=True)

    simulator = Simulator(my_planner)
    simulator_result = simulator.run(simulation_time)
    if simulator_result[1] == "Stopped":
        return ', '.join([str(delta), "Stopped", *map(str, my_planner.get_current_loss()),
                          str(my_planner.policy.num_allocated), str(my_planner.policy.num_postponed)])
    else:
        return ', '.join([str(delta), simulator_result[1], *map(str, my_planner.get_current_loss()),
                          str(my_planner.policy.num_allocated), str(my_planner.policy.num_postponed)])



for i in range(0,10):
    res = run_simulator(i)
    print(res)