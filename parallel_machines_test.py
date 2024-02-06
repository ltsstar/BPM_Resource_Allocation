import pickle
from simulator.simulator import Simulator
from planner import Planner
from policy import *
from task_execution_time import ExecutionTimeModel

prediction_model = ExecutionTimeModel()
with open('prediction_model.pkl', 'rb') as file:
    prediction_model = pickle.load(file)
warm_up_policy = RandomPolicy()
simulation_time = 24*4
warm_up_time = 0

#policy = GreedyParallelMachinesSchedulingPolicy()
policy = HungarianPolicy()
my_planner = Planner(prediction_model, warm_up_policy, warm_up_time, policy,
                     predict_multiple=False)

simulator = Simulator(my_planner)

def my_function():
    result = simulator.run(simulation_time)
    print(result)


if __name__ == "__main__":
    my_function()