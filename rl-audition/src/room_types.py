import numpy as np

"""
Currently, we are only working with rooms that are convex and 2D. 
"""

class Polygon:
    def __init__(self, n, r, x_center=0, y_center=0, theta=0):
        """
        This class represents a polygon room. 
        pyroomacoustics.Room.from_corners() is called to create a polygon room. 

		Args:
			n (int): Number of vertices of the polygon
			r (int): Radius of the polygon
			x_center (int): x coordinate of center of polygon
			y_center (int): y coordinate of center of polygon
			theta (int): polygon construction parameter
		"""
        self.n = n
        self.r = r
        self.x_center = x_center
        self.y_center = y_center
        assert(theta >= 0.0)
        self.theta = theta
        self.corners = True

    def generate(self):
        """
        This function generates a polygon and returns the points

        Returns:
            x_points (List[np.array]): x-coordinates of points that makeup the room
            y_points (List[np.array]): y-coordinates of points that makeup the room
        """
        numbers = np.array([i for i in range(self.n)])

        x_points = (
            self.r * (2 * np.cos((2 * np.pi * numbers) / self.n + self.theta)) + self.x_center
        )
        y_points = (
            self.r * (2 * np.sin((2 * np.pi * numbers) / self.n) + self.theta) + self.y_center
        )

        return [x_points, y_points]


class ShoeBox:
    def __init__(self, x_length=10, y_length=10):
        """
        This class represents a shoe box (rectangular) room. It is a wrapper 
        for the pyroomacoustics.Room.ShoeBox class. We are sticking with
        2D rooms (4 walls) for now, though PRA supports 3D rooms (6 walls)

        Args:
            x_length (float): the horizontal length of the room
            y_length (float): the vertical length of the room
        """
        self.x_length = x_length
        self.y_length = y_length
        self.corners = False

    def generate(self):
        """
        This function generates a shoebox and returns the points

        Returns:
            List[int]: x_length and y_length
        """
        return [self.x_length, self.y_length]
