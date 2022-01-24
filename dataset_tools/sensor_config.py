#!/usr/bin/python3

import carla


class CameraConfig:
    image_width = 1242
    image_height = 375
    view_fov = 90


class VehicleSensorConfig:

    sensor_attribute = [['sensor.camera.rgb', carla.ColorConverter.Raw, 'Camera RGB', {}],
                        ['sensor.camera.semantic_segmentation', carla.ColorConverter.CityScapesPalette,
                         'Camera Semantic Segmentation (CityScapes Palette)', {}],
                        ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {}]]
    sensor_transform = [(carla.Transform(carla.Location(x=0, z=1.65)), carla.AttachmentType.Rigid),
                        (carla.Transform(carla.Location(x=0, z=1.65)), carla.AttachmentType.Rigid),
                        (carla.Transform(carla.Location(x=-0.27, z=1.73)), carla.AttachmentType.Rigid)]


class RoadSensorConfig:
    sensor_attribute = [['sensor.camera.rgb', carla.ColorConverter.Raw, 'Camera RGB', {}],
                        ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {}]]
    sensor_transform = [(carla.Transform(carla.Location(x=0, z=1.7)), carla.AttachmentType.Rigid),
                        (carla.Transform(carla.Location(x=0, z=1.7)), carla.AttachmentType.Rigid)]


class RoadSensorHighConfig:
    sensor_attribute = [['sensor.camera.rgb', carla.ColorConverter.Raw, 'Camera RGB', {}],
                             ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {}]]
    sensor_transform = [(carla.Transform(carla.Location(x=0, z=2.5)), carla.AttachmentType.Rigid),
                             (carla.Transform(carla.Location(x=0, z=2.5)), carla.AttachmentType.Rigid)]
