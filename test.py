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

def run_simulator(delta, result_queue):
    policy = HungarianMultiObjectivePolicy(delta)
    my_planner = Planner(prediction_model, warm_up_policy, warm_up_time, policy,
                        predict_multiple=False)

    simulator = Simulator(my_planner)
    simulator_result = simulator.run(simulation_time)
    if simulator_result[1] == "Stopped":
        result_queue.put(', '.join([str(delta), "Stopped", *map(str, my_planner.get_current_loss())]))
    else:
        result_queue.put(', '.join([str(delta), "Succeeded", *map(str, my_planner.get_current_loss())]))

def get_alive_proceses(all_procesess):
    result = []
    for p in all_procesess:
        if p.is_alive():
            result.append(p)
    return result

processes = []
result_queue = multiprocessing.Queue()
alive_processes = []

MAX_PROCESSES = 32
for i in range(-10, 2000, 10):
    while len(alive_processes) > MAX_PROCESSES:
        time.sleep(0.5)
        alive_processes = get_alive_proceses(processes)

    p = multiprocessing.Process(target=run_simulator, args=(i, result_queue))
    p.start()
    processes.append(p)

    for p in processes:
        if not p.is_alive():
            p.join()
            print(result_queue.get())

alive_processes = get_alive_proceses(processes)
while len(alive_processes):
    for p in processes:
        if not p.is_alive():
            p.join()
            print(result_queue.get())
    time.sleep(0.5)
    alive_processes = get_alive_proceses(processes)


for p in processes:
    p.join()

while not result_queue.empty():
    print(result_queue.get())