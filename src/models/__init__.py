import tensorflow as tf
from os import environ

# Set GPU devices
environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"  # Remove spaces after commas

# Configure GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
try:
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
        
    # limit GPU memory fraction, you can use this instead:
    # for gpu in physical_devices:
    #     tf.config.experimental.set_virtual_device_configuration(
    #         gpu,
    #         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]  # memory in MB
    #     )
        
except RuntimeError as e:
    print(e)

# Enable soft placement
tf.config.set_soft_device_placement(True)
