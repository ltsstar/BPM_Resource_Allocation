from simulator import Simulator


class MyPlanner:

    def plan(self, available_resources, unassigned_tasks, resource_pool):
        assignments = []
        # assign the first unassigned task to the first available resource, the second task to the second resource, etc.
        for task in unassigned_tasks:
            for resource in available_resources:
                if resource in resource_pool[task.task_type]:
                    available_resources.remove(resource)
                    assignments.append((task, resource))
                    break

        return assignments

    def report(self, event):
        print(event)


my_planner = MyPlanner()
simulator = Simulator(my_planner)
result = simulator.run(7*24)
print(result)
