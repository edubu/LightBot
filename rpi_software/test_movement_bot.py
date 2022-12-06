from light_bot import LightBot
import time

robot = LightBot()

rest_pos = {
    'm1': 0,
    'm2': 0,
    'm3': 0,
    'm4': 0,
    'm5': 0
}

rest_pos2 = {
    'm1': -30,
    'm2': -30,
    'm3': -30,
    'm4': -30,
    'm5': -30
}

# You can directly set new positions to motors by providing
# the Robot goto_position method with a dictionary such as
# {motor_name: position, motor_name: position...}
time.sleep(1)
robot.set_joints(list(rest_pos2.values()))
