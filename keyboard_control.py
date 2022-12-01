from light_bot import LightBot
import time
import threading
from pynput import keyboard


class ActiveKeys(object):
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.key_states = [False for i in range(2 * 5)]
    
    def set_key_state(self, keyIdx, isActive):
        self.lock.acquire()
        try:
            self.key_states[keyIdx] = isActive
        finally:
            self.lock.release()
    
    def get_key_states(self):
        self.lock.acquire()
        key_state = list(self.key_states)
        self.lock.release()
        return key_state
        

robot = LightBot()
active_keys = ActiveKeys()
curr_angles = robot.get_joint_angles()

up_key_chars = ['q', 'w', 'e', 'r', 't']
down_key_chars = ['a', 's', 'd', 'f', 'g']
isRunning = True

def on_press(key):
    try:
        n = len(up_key_chars)
        for i in range(n):
            if key.char == up_key_chars[i]:
                active_keys.set_key_state(keyIdx=(2*i), isActive=True)
            elif key.char == down_key_chars[i]:
                active_keys.set_key_state(keyIdx=((2*i) + 1), isActive=True)
                
    except AttributeError:
        print('special key {0} pressed'.format(key))

def on_release(key):
    #print('{0} released'.format(key))
    try:
        n = len(up_key_chars)
        for i in range(n):
            if key.char == up_key_chars[i]:
                active_keys.set_key_state(keyIdx=(2*i), isActive=False)
            elif key.char == down_key_chars[i]:
                active_keys.set_key_state(keyIdx=((2*i) + 1), isActive=False)
    except Exception as e:
        print(e)
        
    if key == keyboard.Key.esc:
        # stop listener
        isRunning = False
        return False

# Start keyboard listener
listener = keyboard.Listener(on_press=on_press,on_release=on_release)
listener.start()

# start listening and setting angles
try:
    update_freq = 10 # HZ
    period = 1/float(update_freq)
    print("Period is at: ", period)
    while isRunning:
        key_states = active_keys.get_key_states()
        hasChanged = False
        for i in range(0, len(key_states), 2):
            up_state = key_states[i]
            down_state = key_states[i + 1]
            
            if up_state:
                hasChanged = True
                curr_angles[i] += 15
            if down_state:
                hasChanged = True
                curr_angles[i] -= 15
        
        if hasChanged:
            robot.set_joints(curr_angles)
        time.sleep(period)
    
    listener.join()

except Exception as e:
    print(e)
    

    
