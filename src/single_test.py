import pickle
from simulator.simulator import Simulator
from planner import Planner

from policy import *
from ilp_policy import UnrelatedParallelMachinesSchedulingPolicy
from ilp_policy_non_assign_2 import UnrelatedParallelMachinesSchedulingNonAssignPolicy2
from ilp_policy_2_batch import UnrelatedParallelMachinesSchedulingBatchPolicy2
from hungarian_policy import *
from park_policy import *
from russel_policies import *
from least_loaded_qualified_person_policy import *
from task_execution_time import ExecutionTimeModel

import multiprocessing
import time

prediction_model = ExecutionTimeModel()
with open('prediction_model_po.pkl', 'rb') as file:
    prediction_model = pickle.load(file)

warm_up_policy = RandomPolicy()
warm_up_time =  0
simulation_time = 24*70

def run_simulator(delta, instance_file):
    #policy = UnrelatedParallelMachinesSchedulingNonAssignPolicy2(1, 0, 0, delta, 'EIF')
    #policy = HungarianMultiObjectivePolicy(1, 0, 0, delta)
    #policy = LeastLoadedQualifiedPersonPolicy()
    policy = RoundRobinPolicy()
    #policy = UnrelatedParallelMachinesSchedulingBatchPolicy2(1, 0, 0, delta, 'fastest', 50)
    #policy = ShortestQueueAllocation()

    my_planner = Planner(prediction_model,
                        warm_up_policy, warm_up_time,
                        policy,
                        predict_multiple=True,
                        #hour_timeout=120,
                        debug=True)

    simulator = Simulator(my_planner, instance_file)
    simulator.problem.interarrival_time._alpha /= 4.8 #reset

    policy = ParkPolicy(simulator.problem.next_task_distribution, my_planner.predictor, my_planner.task_type_occurrences)
    my_planner.policy = policy

    simulator_result = simulator.run(simulation_time)
    if simulator_result[1] == "Stopped":
        return ', '.join([str(delta), "Stopped", *map(str, my_planner.get_current_loss()),
                          str(my_planner.policy.num_allocated), str(my_planner.policy.num_postponed)])
    else:
        return ', '.join([str(delta), simulator_result[1], *map(str, my_planner.get_current_loss()),
                          str(my_planner.policy.num_allocated), str(my_planner.policy.num_postponed)])



res = run_simulator(3, 'data/po_problem.pickle')
print(res)