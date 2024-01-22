import scipy
import numpy as np

class OneStageAssignment:
    def compute(trd):
        i, j, k = (0, 0, 0)
        task_indices, resource_indices = {}, {}
        index_taskresources = {}
        for task, resources in trd.items():
            task_indices[task] = i
            for resource, duration in resources.items():
                if resource not in resource_indices:
                    resource_indices[resource] = j
                    j += 1
                index_taskresources[(resource_indices[resource],i)] = (task, resource)
            i += 1

        mat = np.full((j,i), np.inf, dtype=np.dtype("float32"))
        for task, resources in trd.items():
            for resource, duration in resources.items():
                mat[resource_indices[resource]][task_indices[task]] = duration

        row_ind, col_ind = scipy.optimize.linear_sum_assignment(mat)
        selected = []
        for row, col in zip(row_ind, col_ind):
            if mat[row][col] == np.inf:
                if mat[row].argmin() != np.inf:
                    tr = index_taskresources[(row, mat[row].argmin())]
                else:
                    continue
            else:
                tr = index_taskresources[(row, col)]
            selected.append(tr)
        
        return selected