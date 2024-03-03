from policy import Policy

class LeastLoadedQualifiedPersonPolicy(Policy):
    def __init__(self):
        self.num_allocated = 0
        self.num_postponed = 0

    def allocate(self, unassigned_tasks, available_resources, resource_pool, trd,
                 occupations, fairness, task_costs, working_resources, current_time):
        relevant_resources = set(available_resources) | set(working_resources.keys())

        unassigned_resources = relevant_resources.copy()

        # unassigned tasks is already sorted by instance
        selected = []
        for task in unassigned_tasks:
            qualified_resources = set(resource_pool[task.task_type]).intersection(unassigned_resources)
            qualified_resources_occupation = dict([(resource,occupations[resource] if resource in occupations else 0)
                                                   for resource in qualified_resources])
            if qualified_resources_occupation:
                least_loaded_qualified_resource = sorted(qualified_resources_occupation.items(),
                                                        key=lambda item : item[1])[0][0]
                selected.append((task, least_loaded_qualified_resource))
                unassigned_resources.remove(least_loaded_qualified_resource)
    
        selected_size = len(selected)
        task_assignment =  self.prune_invalid_assignments(selected, available_resources, resource_pool, unassigned_tasks)
        assignment_size = len(task_assignment)
        self.num_allocated += selected_size
        self.num_postponed += selected_size - assignment_size

        return task_assignment
            