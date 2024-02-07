import pickle
from simulator.simulator import Simulator
from planner import Planner
from policy import *
from task_execution_time import ExecutionTimeModel

prediction_model = ExecutionTimeModel()
with open('prediction_model.pkl', 'rb') as file:
    prediction_model = pickle.load(file)

warm_up_policy = RandomPolicy()
warm_up_time =  0
simulation_time = 360*24
policy = HungarianPolicy()

my_planner = Planner(prediction_model, warm_up_policy, warm_up_time, policy,
                     predict_multiple=False)

simulator = Simulator(my_planner)
result = simulator.run(simulation_time)

print(result)
