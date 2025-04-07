# module_a.py
import config


def run():
    if config.DEBUG_MODE:
        print("Module A running in DEBUG mode.")
    else:
        print("Module A running normally.")
