from light_bot import LightBot
import logging
import time

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def main():
    bot = LightBot()
    time.sleep(1)
    try:
        logging.info("Starting movement printing")
        bot.set_motors_compliance([True for i in range(len(bot.motor_names))])
        while(True):
            print(bot.get_joint_angles())
            
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
