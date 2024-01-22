class Policy:
    def allocate(self, trd):
        pass


class RandomPolicy(Policy):
    def allocate(self, unassigned_tasks, available_resources, resource_pool, trd):
        assignments = []
        # assign the first unassigned task to the first available resource, the second task to the second resource, etc.
        for task in unassigned_tasks:
            for resource in available_resources:
                if resource in resource_pool[task.task_type]:
                    available_resources.remove(resource)
                    assignments.append((task, resource))
                    break
        return assignments