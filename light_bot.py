from pypot.robot import robot, config
import time
import os
import logging

# debug_filename = os.getcwd() + "/LightBot/configuration/light_bot.json"
#robot = config.from_json(os.getcwd() + "/configuration/light_bot.json")
#robot = config.from_json(debug_filename)

logger = logging.getLogger(__name__)

class LightBot():
    def __init__(self, config_filename=None):
        if not config_filename:
            config_filename = os.getcwd() + "/configuration/light_bot.json"

        logger.info(f'Building LightBot from {config_filename}')
        self.robot = config.from_json(config_filename)
        logger.info("LightBot successfuly built")
        
        # parameters found by source code
        self.max_speed = 360 # degrees per second
        self.sync_freq = 50 # motor synchronization rate in HZ
        
        # set all motors as non-compliant
        self.set_compliance(isCompliant=False)
            
        # # set to base position
        self.base_angles = [-14.52, 7.77, 20.67, -63.2, 21.55]
        self.set_joints(self.base_angles)
    
    
    # GETTERS
    def get_joint_angles(self):
        curr_pos = [m.present_position for m in self.robot.motors]
        return curr_pos
    
    """
        Description: Sets joint angles of robotic arm and blocks until joint angles are set
        Inputs: list of integer joint angles for each joint
        Returns: True if joint angles were correctly set
    """
    def set_joints(self, angles: list, speed=180) -> bool:        
        logger.debug("Validating joint positions")
        try:
            self.__validate_joint_angles(angles)
        except Exception as e:
            logger.warning(e)
            quit(1)
        logger.debug("Joint angles validated")
                    
        # set the angles of the joints on the robot
        pos_dict = {}
        for i in range(len(angles)):
            pos_dict[self.robot.motors[i].name] = angles[i]
        print("Setting to ", pos_dict)
        
        # calculate wait time based on desired speed of robot
        wait_time = self.__calculate_wait_time(angles, speed=speed)
        self.robot.goto_position(pos_dict, duration=wait_time, wait=True)
        
    def set_compliance(self, isCompliant):
        for m in self.robot.motors:
            m.compliant = isCompliant
    
    
    def __calculate_wait_time(self, angles, speed=180):
        curr_pos = [m.present_position for m in self.robot.motors]
        diffs = [abs(angles[i] - curr_pos[i]) for i in range(len(self.robot.motors))]
        
        if speed > self.max_speed:
            speed = self.max_speed
        return 1/(speed/max(diffs))
        
        
    def __validate_joint_angles(self, angles):
        n = len(angles)
        if n != len(self.robot.motors):
            raise Exception("Invalid amount of joint angles input")
        
        # changing angles to stay within joint limits
        for i in range(n):
            motor = self.robot.motors[i]
            limits = [motor.upper_limit, motor.lower_limit]
            if angles[i] > max(limits):
                print("Upper limit set")
                angles[i] = max(limits)
            if angles[i] < min(limits):
                print("Lower Limit Set")
                angles[i] = min(limits)
                
        
        
        