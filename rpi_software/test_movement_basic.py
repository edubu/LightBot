from pypot.robot import robot, config
import time
import os

# debug_filename = os.getcwd() + "/LightBot/configuration/light_bot.json"
robot = config.from_json(os.getcwd() + "/configuration/light_bot.json")
# robot = config.from_json(debug_filename)

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

for m in robot.motors:
    m.compliant = False

# You can directly set new positions to motors by providing
# the Robot goto_position method with a dictionary such as
# {motor_name: position, motor_name: position...}
robot.goto_position(rest_pos2, duration=0.022, wait=True)
time.sleep(5)