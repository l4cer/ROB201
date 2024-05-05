from place_bot.entities.lidar import LidarParams
from place_bot.entities.odometer import OdometerParams

from place_bot.simu_world.simulator import Simulator

from my_robot import MyRobot

from worlds.my_world import MyWorld


if __name__ == "__main__":
    lidar_params: LidarParams = LidarParams()
    lidar_params.noise_enable = True

    odometer_params: OdometerParams = OdometerParams()

    my_robot: MyRobot = MyRobot(lidar_params=lidar_params,
                                odometer_params=odometer_params)

    my_world: MyWorld = MyWorld(robot=my_robot)
    simulator: Simulator = Simulator(the_world=my_world,
                                     use_keyboard=True)

    simulator.run()
