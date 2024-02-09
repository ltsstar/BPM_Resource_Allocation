import pickle
from simulator.simulator import Simulator
from planner import Planner
from policy import *
from ilp_policy import UnrelatedParallelMachinesSchedulingPolicy
from task_execution_time import ExecutionTimeModel

import numpy as np
import multiprocessing
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

prediction_model = ExecutionTimeModel()
with open('prediction_model.pkl', 'rb') as file:
    prediction_model = pickle.load(file)

warm_up_policy = RandomPolicy()
warm_up_time =  0
simulation_time = 24

def run_simulator(delta, result_queue):
    policy = HungarianMultiObjectivePolicy(1, 0, 0, delta)
    my_planner = Planner(prediction_model, warm_up_policy, warm_up_time, policy,
                        predict_multiple=True,
                        hour_timeout=120,
                        debug=False)

    simulator = Simulator(my_planner)
    simulator_result = simulator.run(simulation_time)
    if simulator_result[1] == "Stopped":
        result_queue.push([str(delta), "Stopped", *map(str, my_planner.get_current_loss()),
                          str(my_planner.policy.num_allocated), str(my_planner.policy.num_postponed)])
    else:
        result_queue.push([str(delta), simulator_result[1], *map(str, my_planner.get_current_loss()),
                          str(my_planner.policy.num_allocated), str(my_planner.policy.num_postponed)])

def get_alive_proceses(all_procesess):
    result = []
    for p in all_procesess:
        if p.is_alive():
            result.append(p)
    return result

processes = []
result_queue = multiprocessing.Queue()
alive_processes = []

MAX_PROCESSES = 2
for i in np.arange(0, 5, 0.1):
    alive_processes = get_alive_proceses(processes)
    while len(alive_processes) >= MAX_PROCESSES:
        time.sleep(0.5)
        alive_processes = get_alive_proceses(processes)

    p = multiprocessing.Process(target=run_simulator, args=(i, result_queue))
    p.start()
    print(i, 'Started')
    processes.append(p)

    while not result_queue.empty():
        print(result_queue.get())

for p in processes:
    p.join()
while not result_queue.empty():
    print(result_queue.get())