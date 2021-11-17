#!/usr/bin/python3
import copy
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
    def __init__(self, parent_id, sensor_attribute_list, sensor_transform_list, args):
        Thread.__init__(self)
        self.recording = False
        self.data_queue = Queue()
        self.args = args
        self.client = carla.Client(self.args.host, self.args.port)
        self.world = self.client.get_world()

        self._parent = self.world.get_actor(parent_id)

        self.sensors_transforms = sensor_transform_list  # (sensor_transform, Attachment.Rigid)
        self.sensors_attribute_list = sensor_attribute_list
        self.sensor_attribute = None
        self.sensor_id_list = []
        self.sensor_list = []

    def run(self):
        self.spawn_sensors()

    def spawn_sensors(self):
        for i in range(len(self.sensors_attribute_list)):
            sensor_attribute_raw = self.sensors_attribute_list[i]
            sensor_transform = self.sensors_transforms[i]
        # for sensor_attribute_raw, sensor_transform in zip(self.sensors_attribute_list, self.sensors_transforms):
            gamma_correction = 2.2
            bp_library = self.world.get_blueprint_library()
            bp = bp_library.find(sensor_attribute_raw[0])
            if sensor_attribute_raw[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', '1382')
                bp.set_attribute('image_size_y', '512')
                bp.set_attribute('fov', '90')
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in sensor_attribute_raw[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif sensor_attribute_raw[0].startswith('sensor.lidar'):
                # bp.set_attribute('range', '10')
                # bp.set_attribute('channels', '16')
                # bp.set_attribute('points_per_second', '22400')
                bp.set_attribute('range', '100')
                bp.set_attribute('channels', '64')
                bp.set_attribute('points_per_second', '1300000')
                bp.set_attribute('rotation_frequency', '10')
                bp.set_attribute('upper_fov', '2.0')
                bp.set_attribute('lower_fov', '-24.8')
                # bp.set_attribute('sensor_tick', str(0.05))
                # bp.set_attribute('dropoff_general_rate', '0.0')
                # bp.set_attribute('dropoff_intensity_limit', '1.0')
                # bp.set_attribute('dropoff_zero_intensity', '0.0')
                bp.set_attribute('noise_stddev', '0.02')
            sensor_attribute_raw.append(bp)
            sensor_actor = self._parent.get_world().spawn_actor(
                bp,
                sensor_transform[0],
                attach_to=self._parent)
            self.set_sensor(sensor_actor)

    def set_sensor(self, sensor_actor: carla.Sensor):
        self.sensor_list.append(sensor_actor)
        self.sensor_id_list.append(sensor_actor.id)
        filename = Path(self.args.raw_data_path,
                            "{}_{}".format(self._parent.type_id, self._parent.id),
                            "{}_{}".format(self._parent.type_id, self._parent.id))
        filename = filename.as_posix()
        sensor_type = copy.deepcopy(str(sensor_actor.type_id))
        print(self._parent.id, sensor_type, sensor_actor.id)
        weak_self = weakref.ref(self)
        sensor_actor.listen(lambda sensor_data: CavCollectThread.data_callback(weak_self,
                                                                               sensor_data,
                                                                               sensor_type,
                                                                               filename,
                                                                               self.data_queue))
        # print("id_list: {}".format(self.sensor_id_list))
        # self.sensor.stop()
        # print(filename)

    @staticmethod
    def data_callback(weak_self, sensor_data, type_id, filename, data_queue: Queue):
        data_queue.put((sensor_data, type_id, filename))

    def get_sensor_id_list(self):
        self.join()
        return self.sensor_id_list

    def save_to_disk(self):
        print("Save vehicle {} sensor data:".format(self._parent.id))
        for _ in range(len(self.sensor_list)):
            sensor_frame = self.data_queue.get(True, 1.0)
            sensor_data = sensor_frame[0]
            sensor_type_id = sensor_frame[1]
            filename = sensor_frame[2]

            print("\tFrame: {} type: {} | {}".format(sensor_data.frame, sensor_data, sensor_type_id))

            if sensor_type_id == 'sensor.camera.semantic_segmentation':
                sensor_data.convert(carla.ColorConverter.CityScapesPalette)
                # sensor_data.save_to_disk(filename + '/seg' + '/%010d' % sensor_data.frame)
                carla_image_data_array = numpy.ndarray(
                    shape=(sensor_data.height, sensor_data.width, 4),
                    dtype=numpy.uint8, buffer=sensor_data.raw_data)
                os.makedirs(filename + '/seg', exist_ok=True)
                numpy.savez_compressed(filename + '/seg' + '/%010d' % sensor_data.frame, a=carla_image_data_array)
            else:
                sensor_data.save_to_disk(filename + '/%010d' % sensor_data.frame)

