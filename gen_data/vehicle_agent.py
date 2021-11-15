#!/usr/bin/python3
import sys
from pathlib import Path
sys.path.append(Path(__file__).resolve().parent.parent.as_posix() ) #repo path
sys.path.append(Path(__file__).resolve().parent.as_posix() ) #file path
from threading import Thread

import carla
import weakref
import numpy
from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from queue import Queue
from params import *

class VehicleAgent(BehaviorAgent):
    def __init__(self, vehicle):
        self.id = vehicle.id
        BehaviorAgent.__init__(self, vehicle)


class CavControlThread(Thread):
    #   继承父类threading.Thread
    def __init__(self, vehicle_agent: VehicleAgent, world, destination, num_min_waypoints, apply_vehicle_control):
        Thread.__init__(self)
        self.v = vehicle_agent
        self.id = vehicle_agent.id
        self.w = world
        self.d = destination
        self.n = num_min_waypoints
        self.c_cmd = apply_vehicle_control
        self.control = None
        self.v.set_target_speed(15.0)
        self.start()

    def run(self):
        #   把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        self.control = self.v.run_step()
        self.c_cmd = self.c_cmd(self.id, self.control)

    def return_control(self):
        #   threading.Thread.join(self) # 等待线程执行完毕
        self.join()
        try:
            return self.c_cmd
        except Exception:
            print('This is an issue')


class CavCollectThread(Thread):
    def __init__(self, parent_id, sensor_attribute, sensor_transform, args):
        Thread.__init__(self)
        self.recording = False
        self.args = args
        gamma_correction = 2.2
        self.client = carla.Client(self.args.host, self.args.port)
        world = self.client.get_world()
        self.sensor = None
        self._parent = world.get_actor(parent_id)
        self._camera_transforms = sensor_transform  # (sensor_transform, Attachment.Rigid)
        bp_library = world.get_blueprint_library()
        bp = bp_library.find(sensor_attribute[0])
        if sensor_attribute[0].startswith('sensor.camera'):
            bp.set_attribute('image_size_x', str(self.args.image_width))
            bp.set_attribute('image_size_y', str(self.args.image_height))
            if bp.has_attribute('gamma'):
                bp.set_attribute('gamma', str(gamma_correction))
            for attr_name, attr_value in sensor_attribute[3].items():
                bp.set_attribute(attr_name, attr_value)
        elif sensor_attribute[0].startswith('sensor.lidar'):
            bp.set_attribute('range', '10')
            bp.set_attribute('channels', '16')
            bp.set_attribute('points_per_second', '22400')
            # bp.set_attribute('range', '100')
            # bp.set_attribute('channels', '64')
            # bp.set_attribute('points_per_second', '2240000')
            bp.set_attribute('rotation_frequency', '20')
            bp.set_attribute('sensor_tick', str(0.05))
            bp.set_attribute('dropoff_general_rate', '0.0')
            bp.set_attribute('dropoff_intensity_limit', '1.0')
            bp.set_attribute('dropoff_zero_intensity', '0.0')
            # bp.set_attribute('noise_stddev', '0.0')
        sensor_attribute.append(bp)
        self.sensor_attribute = sensor_attribute
        self.data_queue = Queue()

    def run(self):
        self.set_sensor()

    def set_sensor(self):
        self.sensor = self._parent.get_world().spawn_actor(
            self.sensor_attribute[-1],
            self._camera_transforms[0],
            attach_to=self._parent)
        # attachment_type=self._c#amera_transforms[1])
        filename = Path(self.args.raw_data_path,
                        "{}_{}".format(self._parent.type_id, self._parent.id),
                        "{}_{}".format(self._parent.type_id, self._parent.id))
        filename = filename.as_posix()
        # '%s_%d'%(self._parent.type_id, self._parent.id),
        # '%s_%d'%(self.sensor.type_id, self.sensor.id)).as_posix()
        # filename = self.args.raw_data_path + \
        #             self._parent.type_id + '_' + str(self._parent.id) + '/' + \
        #             self.sensor.type_id + '_' + str(self.sensor.id)

        weak_self = weakref.ref(self)
        self.sensor.listen(lambda sensor_data: CavCollectThread.data_callback(weak_self, sensor_data, filename, self.data_queue))
        # self.sensor.stop()
        # print(filename)

    @staticmethod
    def data_callback(weak_self, data, filename, data_queue: Queue):
        data_queue.put((data, filename))

    def get_sensor_id(self):
        self.join()
        return self.sensor.id

    def save_to_disk(self):
        sensor_frame = self.data_queue.get(True, 1.0)
        sensor_data = sensor_frame[0]
        filename = sensor_frame[1]
        # if sensor_data.frame < RAW_DATA_START or sensor_data.frame > RAW_DATA_END:
        #     return
        if sensor_data.frame % self.args.sample_frequence != 0:
            return
        if self.sensor.type_id == 'sensor.camera.semantic_segmentation':
            sensor_data.convert(self.sensor_attribute[1])
            # sensor_data.save_to_disk(filename + '/seg' + '/%010d' % sensor_data.frame)
            carla_image_data_array = numpy.ndarray(
                shape=(sensor_data.height, sensor_data.width, 4),
                dtype=numpy.uint8, buffer=sensor_data.raw_data)
            os.makedirs(filename + '/seg', exist_ok=True)
            numpy.savez_compressed(filename + '/seg' + '/%010d' % sensor_data.frame, a=carla_image_data_array)
        # elif self.sensor.type_id == 'sensor.camera.depth':
        #     sensor_data.convert(self.sensor_attribute[1])
        #     sensor_data.save_to_disk(filename + '/%010d' % sensor_data.frame + '_depth')
        else:
            sensor_data.save_to_disk(filename + '/%010d' % sensor_data.frame)
