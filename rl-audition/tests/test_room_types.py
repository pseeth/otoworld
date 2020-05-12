# NOTE: to run, need to cd into tests/, then run pytest
import sys
sys.path.append("../src")

from pyroomacoustics import ShoeBox, Room

import room_types
import constants

def test_polygon_num_sides():
    """
    Test Polygon room class with different number of sides
    """
    for num_sides in range(3, 11):
        room = room_types.Polygon(n=num_sides, r=2)
        points = room.generate()

        # only x and y-coordinates for 2d polygon 
        assert(len(points) == 2)

        # create pra room 
        pra_room = Room.from_corners(points, fs=constants.RESAMPLE_RATE)

        # assert center is inside of room
        assert(pra_room.is_inside([room.x_center, room.y_center]))

        # apparently polygons don't need to be convex 
        # pra_room.convex_hull()
        # assert(len(pra_room.obstructing_walls) == 0)

def test_shoebox():
    """
    Testing (our) ShoeBox room class
    """
    # 2d shoebox room is a rectangle (4 walls)
    room = room_types.ShoeBox()
    points = room.generate()

    # sanity, rectangles only have two different lengths
    assert(len(points) == 2)

    # ensure class variables equal points returned by generate
    assert(room.x_length == points[0])
    assert(room.y_length == points[1])

    # not sure how this would fail
    assert((room.x_length * room.y_length) == (points[0] * points[1]))

    # create pra room 
    pra_room = ShoeBox(points)

    # test whether it is a convex hull (this should be ensured by pra)
    pra_room.convex_hull()
    assert(len(pra_room.obstructing_walls) == 0)
