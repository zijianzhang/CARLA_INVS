#!/usr/bin/python

import carla


class Map(object):
    def __init__(self, args):
        self.pretrain_model = True
        self.client = carla.Client(args.host, args.port)
        self.world = self.client.get_world()
        self.initial_spectator(args.spectator_point)
        self.tmp_spawn_points = self.world.get_map().get_spawn_points()
        if self.pretrain_model:
            self.initial_spawn_points = self.tmp_spawn_points
        else:
            self.initial_spawn_points = self.check_spawn_points(args.initial_spawn_ROI)
        self.additional_spawn_points = self.check_spawn_points(args.additional_spawn_ROI)
        self.destination = self.init_destination(self.tmp_spawn_points, args.ROI)
        self.ROI = args.ROI
        try:
            self.hd_id = args.hd_id
            self.av_id = args.av_id
        except:
            self.hd_id = []
            self.av_id = []

    def initial_spectator(self, spectator_point):
        spectator = self.world.get_spectator()
        spectator_point_transform = carla.Transform(carla.Location(spectator_point[0][0],
                                                                   spectator_point[0][1],
                                                                   spectator_point[0][2]),
                                                    carla.Rotation(spectator_point[1][0],
                                                                   spectator_point[1][1],
                                                                   spectator_point[1][2]))
        spectator.set_transform(spectator_point_transform)

    def check_spawn_points(self, check_spawn_ROI):
        tmp_spawn_points = []
        tmpx, tmpy = [], []
        for tmp_transform in self.tmp_spawn_points:
            tmp_location = tmp_transform.location
            for edge in check_spawn_ROI:
                if edge[0] < tmp_location.x < edge[1] \
                        and edge[2] < tmp_location.y < edge[3]:
                    tmp_spawn_points.append(tmp_transform)
                    tmpx.append(tmp_location.x)
                    tmpy.append(tmp_location.y)
                    continue
        # self.plot_points(tmpx,tmpy)
        return tmp_spawn_points

    # def plot_points(self, tmpx, tmpy):
    #     plt.figure(figsize=(8, 7))
    #     ax = plt.subplot(111)
    #     ax.axis([-50, 250, 50, 350])
    #     ax.scatter(tmpx, tmpy)
    #     for index in range(len(tmpx)):
    #         ax.text(tmpx[index], tmpy[index], index)
    #     plt.show()

    def init_destination(self, spawn_points, ROI):
        destination = []
        tmpx, tmpy = [], []
        for p in spawn_points:
            if not self.inROI([p.location.x, p.location.y], ROI):
                destination.append(p)
                tmpx.append(p.location.x)
                tmpy.append(p.location.y)
        # self.plot_points(tmpx,tmpy)
        return destination

    def sign(self, a, b, c):
        return (a[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (a[1] - c[1])

    def inROI(self, x, ROI):
        d1 = self.sign(x, ROI[0], ROI[1])
        d2 = self.sign(x, ROI[1], ROI[2])
        d3 = self.sign(x, ROI[2], ROI[3])
        d4 = self.sign(x, ROI[3], ROI[0])

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0) or (d4 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0) or (d4 > 0)
        return not (has_neg and has_pos)

    def shuffle_spawn_points(self, spawn_points, start=False):
        # random.shuffle(spawn_points)
        if self.pretrain_model:
            # self.av_id = [4,5,27,20,97,22,14,77,47]
            # self.hd_id = [19,21,29,31,44,48,87,96] + [i for i in range(50,70)]

            cav = [spawn_points[i] for i in self.av_id]
            hd = [spawn_points[i] for i in self.hd_id]
            if len(cav) == 0 and len(hd) == 0:
                return spawn_points[:60], spawn_points[-20:]
            else:
                return hd, cav
