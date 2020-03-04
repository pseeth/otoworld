import numpy as np


class Polygon:
    def __init__(self, n, r, x_center=0, y_center=0, theta=0):
        """

		Args:
			n: Number of vertices of the polygon
			r: Radius of the polygon
		"""

        self.n = n
        self.r = r
        self.x_center = x_center
        self.y_center = y_center
        self.theta = theta

    def generate(self):
        numbers = np.array([i for i in range(self.n)])

        x_points = (
            self.r * (2 * np.cos((2 * np.pi * numbers) / self.n + self.theta)) + self.x_center
        )
        y_points = (
            self.r * (2 * np.sin((2 * np.pi * numbers) / self.n) + self.theta) + self.y_center
        )

        return x_points, y_points


if __name__ == "__main__":
    n = np.array([i for i in range(6)])

    x = 2 * 2 * np.cos((2 * np.pi * n) / 6)
    y = 2 * 2 * np.sin((2 * np.pi * n) / 6)
    print(x)
    print(y)
