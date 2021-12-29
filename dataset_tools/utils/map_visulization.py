import carla
import argparse
import matplotlib.pyplot as plt


class MapVisualization:
    def __init__(self, world: carla.World):
        self.world = world
        self.map = self.world.get_map()
        self.fig, self.ax = plt.subplots()


    def destroy(self):
        self.world = None

    @staticmethod
    def lateral_shift(transform, shift):
        """Makes a lateral shift of the forward vector of a transform"""
        transform.rotation.yaw += 90
        return transform.location + shift * transform.get_forward_vector()

    def draw_line(self, points: list):
        x = []
        y = []
        for p in points:
            x.append(p.x)
            y.append(-p.y)
        self.ax.plot(x, y, color='darkslategrey', markersize=2)
        return True

    def draw_spawn_points(self):
        spawn_points = self.map.get_spawn_points()
        for i in range(len(spawn_points)):
            p = spawn_points[i]
            x = p.location.x
            y = -p.location.y
            self.ax.text(x, y, str(i),
                         fontsize=6,
                         color='darkorange',
                         va='center',
                         ha='center',
                         weight='bold')

    def draw_roads(self):
        precision = 0.1
        topology = self.map.get_topology()
        topology = [x[0] for x in topology]
        topology = sorted(topology, key=lambda w: w.transform.location.z)
        set_waypoints = []
        for waypoint in topology:
            waypoints = [waypoint]
            nxt = waypoint.next(precision)
            if len(nxt) > 0:
                nxt = nxt[0]
                while nxt.road_id == waypoint.road_id:
                    waypoints.append(nxt)
                    nxt = nxt.next(precision)
                    if len(nxt) > 0:
                        nxt = nxt[0]
                    else:
                        break
            set_waypoints.append(waypoints)

        for waypoints in set_waypoints:
            # waypoint = waypoints[0]
            road_left_side = [self.lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints]
            road_right_side = [self.lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints]
            # road_points = road_left_side + [x for x in reversed(road_right_side)]
            # self.add_line_strip_marker(points=road_points)

            if len(road_left_side) > 2:
                self.draw_line(points=road_left_side)
            if len(road_right_side) > 2:
                self.draw_line(points=road_right_side)

    def show_spawn_points(self):
        self.map = self.world.get_map()
        self.draw_roads()
        self.draw_spawn_points()
        self.ax.invert_xaxis()
        self.ax.invert_yaxis()
        plt.axis('equal')
        plt.show()
