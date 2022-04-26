# Author: Jakub Mazurkiewicz
import numpy as np

class GradientDescent:
    def __init__(self, nabla_func, start_point: np.array, step_size: float, max_iteration: int):
        self.local_min = start_point
        self.points = [np.array(start_point)]
        self._calc(nabla_func, start_point, step_size, max_iteration)

    def _calc(self, nabla_func, start_point: np.array, step_size: float, max_iteration: int):
        point = np.array(start_point)
        for _ in range(max_iteration):
            point -= step_size * np.array(nabla_func(point))
            self.points.append(point.copy())
        self.local_min = point

    def get_points(self) -> list:
        return self.points

    def get_local_min(self) -> float:
        return self.local_min
