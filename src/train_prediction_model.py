import pickle
from simulator.simulator import Simulator
from planner import Planner
from russel_policies import *
from ilp_policy import UnrelatedParallelMachinesSchedulingPolicy
from task_execution_time import *

import multiprocessing
import time


prediction_model = ExecutionTimeModelPO()
warm_up_policy = RandomPolicy()
warm_up_time =  365*24*3
simulation_time = warm_up_time

#policy = HungarianMultiObjectivePolicy(1, 0, 0, delta)
my_planner = Planner(prediction_model,
                    warm_up_policy, warm_up_time,
                    warm_up_policy,
                    predict_multiple=True,
                    hour_timeout=120,
                    debug=True)

#instance_file="./data/BPI Challenge 2017 - instance.pickle"
instance_file='./data/po_problem.pickle'
simulator = Simulator(my_planner, instance_file)
simulator.problem.interarrival_time._alpha /= 4.8 #reset
simulator_result = simulator.run(simulation_time+24)


with open('prediction_model_po.pkl', 'wb') as file:
    pickle.dump(prediction_model, file)

print(simulator_result)