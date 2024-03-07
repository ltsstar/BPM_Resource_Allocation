import pickle
from simulator.simulator import Simulator
from planner import Planner
from policy import *
from ilp_policy import UnrelatedParallelMachinesSchedulingPolicy
from ilp_policy_non_assign import UnrelatedParallelMachinesSchedulingNonAssignPolicy
from ilp_policy_non_assign_2 import UnrelatedParallelMachinesSchedulingNonAssignPolicy2
from ilp_policy_2_batch import UnrelatedParallelMachinesSchedulingBatchPolicy2
from least_loaded_qualified_person_policy import LeastLoadedQualifiedPersonPolicy
from russel_policies import RoundRobinPolicy
from park_policy import *
from task_execution_time import ExecutionTimeModel
from hungarian_policy import HungarianMultiObjectivePolicy

import numpy as np
import multiprocessing
import time
from datetime import datetime
import csv
import sys

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



def run_simulator(days, objective, delta, result_queue, selection_strategy=None):
    real_start_time = time.time()
    start_time = time.process_time()
    prediction_model = ExecutionTimeModel()
    with open('prediction_model.pkl', 'rb') as file:
        prediction_model = pickle.load(file)

    warm_up_policy = RandomPolicy()
    warm_up_time =  0
    simulation_time = 24*days
    if objective == "Hungarian":
        policy = HungarianMultiObjectivePolicy(1, 0, 0, delta)
    elif objective == "MILP":
        policy = UnrelatedParallelMachinesSchedulingNonAssignPolicy2(1, 0, 0, delta, selection_strategy)
    elif objective == "KBatch":
        # use delta for batch size k
        policy = UnrelatedParallelMachinesSchedulingBatchPolicy2(1, 0, 0, 0, selection_strategy, delta) 
    elif objective == "Park":
        policy = None
    elif objective == "RoundRobin":
        policy = RoundRobinPolicy()
    elif objective == "LLQP":
        policy = LeastLoadedQualifiedPersonPolicy()

    my_planner = Planner(prediction_model, warm_up_policy, warm_up_time, policy,
                        predict_multiple=True,
                        hour_timeout=3600,
                        debug=True)

    simulator = Simulator(my_planner)

    if objective == "Park":
        policy = ParkPolicy(simulator.problem.next_task_distribution, my_planner.predictor, my_planner.task_type_occurrences)
        my_planner.policy = policy

    simulator_result = simulator.run(simulation_time)
    times = (datetime.fromtimestamp(real_start_time).strftime("%Y-%m-%d %H:%M:%S"),
             datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S"),
             str(time.time() - real_start_time),
             str(time.process_time()-start_time)
            )
    if simulator_result[1] == "Stopped":
        res = [objective, *times, str(delta), "Stopped", "", *map(str, my_planner.get_current_loss()),
                          str(my_planner.num_assignments), str(my_planner.policy.num_allocated), str(my_planner.policy.num_postponed)]
    else:
        res = [objective, *times, str(delta), *map(str, simulator_result), *map(str, my_planner.get_current_loss()),
                          str(my_planner.num_assignments), str(my_planner.policy.num_allocated), str(my_planner.policy.num_postponed)]
    if objective == "MILP":
        res += [str(policy.optimal), str(policy.feasible), str(policy.no_solution), selection_strategy]
    else:
        res += ['', '', '', '']
    result_queue.put(res)

def get_alive_proceses(all_procesess):
    result = []
    for p in all_procesess:
        if p.is_alive():
            result.append(p)
    return result

processes = []
result_queue = multiprocessing.Queue()
alive_processes = []

MAX_PROCESSES = int(sys.argv[6])
for i in np.arange(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])):
    alive_processes = get_alive_proceses(processes)
    while len(alive_processes) >= MAX_PROCESSES:
        time.sleep(0.5)
        alive_processes = get_alive_proceses(processes)

    selection_strategy = sys.argv[7]
    p = multiprocessing.Process(target=run_simulator, args=(int(sys.argv[5]), sys.argv[4], i, result_queue, selection_strategy))
    p.start()
    print(i, 'Started')
    processes.append(p)

    while not result_queue.empty():
        res = result_queue.get()
        print(res)
        with open('results.csv', 'a') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(res)

for p in processes:
    p.join()
while not result_queue.empty():
    res = result_queue.get()
    print(res)
    with open('results.csv', 'a') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(res)
