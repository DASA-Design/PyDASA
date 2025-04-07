# main.py
import module_a
import module_b
import config

print("Initial DEBUG_MODE:", config.DEBUG_MODE)
module_a.run()

module_b.enable_debug()

print("Updated DEBUG_MODE:", config.DEBUG_MODE)
module_a.run()
