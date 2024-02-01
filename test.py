import pickle
from simulator.simulator import Simulator
from planner import Planner
from policy import *
from task_execution_time import ExecutionTimeModel

prediction_model = ExecutionTimeModel()
warm_up_policy = RandomPolicy()
warm_up_time =  24*365
#policy = RandomPolicy()
policy = HungarianPolicy()
policy = GreedyParallelMachinesSchedulingPolicy()
my_planner = Planner(prediction_model, warm_up_policy, warm_up_time, policy,
                     predict_multiple=False)

simulator = Simulator(my_planner)
result = simulator.run(warm_up_time)

with open('prediction_model.pkl', 'wb') as file:
    pickle.dump(prediction_model, file)

print(result)
