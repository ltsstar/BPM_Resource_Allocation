import pickle
from simulator.simulator import Simulator, Reporter
from planner import Planner
from russel_policies import *
from ilp_policy import UnrelatedParallelMachinesSchedulingPolicy
from task_execution_time import *
from simulator.problems import *

import multiprocessing
import time
import sys



#problem_file = "src/simulator/data/BPI Challenge 2017 - clean Jan Jun - problem.pickle"
problem_file = "src/simulator/data/BPI Challenge 2017 - instance 2.pickle"
#problem_file='./data/po_problem.pickle'

sys.path.append('src/simulator')
problem = MinedProblem.from_file(problem_file)
problem.interarrival_time._alpha *= 4.8

prediction_model = ExecutionTimeModelPO()
warm_up_policy = RandomPolicy()
warm_up_time =  365*24*3
simulation_time = warm_up_time
activity_names = list(problem.resource_pools.keys())
#policy = HungarianMultiObjectivePolicy(1, 0, 0, delta)
my_planner = Planner(prediction_model,
                    warm_up_policy, warm_up_time,
                    warm_up_policy,
                    activity_names,
                    predict_multiple=True,
                    hour_timeout=120,
                    debug=True)

simulator = Simulator(problem, Reporter(), my_planner)


simulator_result = simulator.simulate(simulation_time+24)








with open('prediction_model_bpic.pkl', 'wb') as file:
    pickle.dump(prediction_model, file)

print(simulator_result)