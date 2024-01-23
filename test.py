from simulator.simulator import Simulator
from planner import Planner
from policy import RandomPolicy
from task_execution_time import ExecutionTimeModel

prediction_model = ExecutionTimeModel()
warm_up_policy = RandomPolicy()
warm_up_time =  23*365/4
policy = RandomPolicy()
my_planner = Planner(prediction_model, warm_up_policy, warm_up_time, policy)

simulator = Simulator(my_planner)
result = simulator.run(24*365)
print(result)