import pickle
from simulator.simulator import Simulator
from planner import Planner
from policy import *
from ilp_policy import UnrelatedParallelMachinesSchedulingPolicy
from task_execution_time import ExecutionTimeModel

import multiprocessing

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
        result_queue.put(', '.join(["Stopped", *map(str, my_planner.get_current_loss())]))
    else:
        result_queue.put(', '.join(["Succeeded", *map(str, my_planner.get_current_loss())]))

for i in range(-10, 2000, 10):
    result_queue = multiprocessing.Queue()
    p1 = multiprocessing.Process(target=run_simulator, args=(i, result_queue))
    p2 = multiprocessing.Process(target=run_simulator, args=(i, result_queue))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    print(i, result_queue.get())
    print(i, result_queue.get())