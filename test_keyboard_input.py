from pynput import keyboard
import threading

def on_press(key):
    try:
        print('alphanumeric key {0} pressed'.format(key.char))
    except AttributeError:
        print('special key {0} pressed'.format(key))
        pass

def on_release(key):
    print('{0} released'.format(key))
    if key == keyboard.Key.esc:
        # stop listener
        return False

# Collect events until released
with keyboard.Listener(on_press=on_press,on_release=on_release) as listener:
    listener.join()
