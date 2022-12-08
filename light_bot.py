import time
import os
import requests

# debug_filename = os.getcwd() + "/LightBot/configuration/light_bot.json"
#robot = config.from_json(os.getcwd() + "/configuration/light_bot.json")
#robot = config.from_json(debug_filename)

import logging
logger = logging.getLogger(__name__)

class LightBot():
    def __init__(self, port=8080):
        self.api_url = f'http://poppy.local:{port}'
        
        self.motor_names = self.get_motor_names()
        self.joint_limits = self.get_joint_limits()
        
        # parameters found by source code
        self.max_speed = 360 # degrees per second
        self.sync_freq = 50 # motor synchronization rate in HZ
        
        # set all motors as non-compliant
        self.set_motors_compliance([False for i in range(len(self.motor_names))])
            
        # set to base position
        self.base_angles = [0.0, 96.0, 20.67, 20, 21.55]
        self.set_joint_angles(self.base_angles)
        
        # flash motor LEDs to green to indicate ready state
        self.__setup_flash_leds()
    
    
    # GETTERS
    """
        Returns joint limits in form
        {
            'motor_name1': [lower_limit, upper_limit]
            ...
            'motor_name5': [lower_limit, upper_limit]
        }
    """
    def get_joint_limits(self):
        motor_names = self.get_motor_names()
        
        reg_low = 'lower_limit'
        reg_up = 'upper_limit'
        
        low_limits_uri = f'/motors/registers/{reg_low}/list.json'
        up_limits_uri = f'/motors/registers/{reg_up}/list.json'
        
        res = requests.get(self.api_url + low_limits_uri)
        lower_limits = res.json()
        
        res = requests.get(self.api_url + up_limits_uri)
        upper_limits = res.json()
        
        joint_limits = {}
        for mname in motor_names:
            lower_limit = lower_limits['lower_limit'][mname]
            upper_limit = upper_limits['upper_limit'][mname]
            joint_limits[mname] = [lower_limit, upper_limit]
        
        return joint_limits
    
    def get_motor_names(self) -> list:
        motor_list_uri = "/motors/list.json"
        response = requests.get(self.api_url + motor_list_uri)
        response = response.json()
        
        return response['motors']
    
    def get_joint_angles(self) -> list:
        register = 'present_position'
        motor_names = self.get_motor_names()
        joint_angles_uri = f'/motors/registers/{register}/list.json'
        response = requests.get(self.api_url + joint_angles_uri)
        response = response.json()
        
        joint_angles = []
        for mname in motor_names:
            joint_angles.append(response[register][mname])
        
        return joint_angles
    
    
    # ------- SETTERS ----------
    """
        Description: Sets joint angles of robotic arm and blocks until joint angles are set
        Inputs: list of integer joint angles for each joint
        Returns: True if joint angles were correctly set
    """
    def set_joint_angles(self, angles: list, speed=180) -> bool:        
        logger.debug("Validating joint positions")
        try:
            self.__validate_joint_angles(angles)
        except Exception as e:
            logger.warning(e)
            quit(1)
        logger.debug("Joint angles validated")
        
        move_joint_uri = '/motors/goto.json' # motor set position endpoint
        
        # set angle depending on fist value
        if angles[-1] == 1:
            angles[-1] -37.0
        elif angles[-1] == 0:
            angles[-1] = 31.52
        
        # set the angles of the joints on the robot
        move_command = {}
        move_command['motors'] = self.motor_names
        move_command['positions'] = angles
        
        # calculate wait time based on desired speed of robot
        wait_time = self.__calculate_wait_time(angles, speed=speed)
        move_command['duration'] = str(wait_time)
        move_command['wait'] = True
    
        res = requests.post(self.api_url + move_joint_uri, json=move_command)
        
        
        
    def set_motor_compliance(self, motor_name, isCompliant):
        comp_uri = f'/motors/{motor_name}/registers/compliant/value.json'
        
        res = requests.post(self.api_url + comp_uri, json=isCompliant)
    
    def set_motors_compliance(self, compliance_states):
        for i in range(len(self.motor_names)):
            self.set_motor_compliance(self.motor_names[i], compliance_states[i])        
            
    def set_motor_led(self, motor_name, led_state):
        led_uri = f'/motors/{motor_name}/registers/led/value.json'
        
        res = requests.post(self.api_url + led_uri, json=led_state)
    
    def set_motors_led(self, led_states):
        for i in range(len(self.motor_names)):
            self.set_motor_led(self.motor_names[i], led_states[i])
    
    
    def __calculate_wait_time(self, angles, speed=180):
        curr_pos = self.get_joint_angles()
        diffs = [abs(angles[i] - curr_pos[i]) for i in range(len(self.motor_names))]
        
        max_speed = 360
        if speed > max_speed:
            speed = max_speed
        return 1/(speed/max(diffs))
        
        
    def __validate_joint_angles(self, angles):
        n = len(angles)
        if n != len(self.motor_names):
            raise Exception("Invalid amount of joint angles input")
        
        # changing angles to stay within joint limits
        for i, mname in enumerate(self.motor_names):
            limits = self.joint_limits[mname]
            if angles[i] > max(limits):
                #print("Upper limit set")
                angles[i] = max(limits)
            if angles[i] < min(limits):
               #print("Lower Limit Set")
                angles[i] = min(limits)
        
    
    def __setup_flash_leds(self):
        self.set_motors_led(['red' for i in range(len(self.motor_names))])
        time.sleep(1)
        self.set_motors_led(['green' for i in range(len(self.motor_names))])
