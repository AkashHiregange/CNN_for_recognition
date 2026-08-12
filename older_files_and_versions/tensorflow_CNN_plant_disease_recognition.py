import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
import os

import config

proj_config = config.ProjectConfig()

base = os.getcwd()

train_dir = os.path.join(base,'Train')
test_dir = os.path.join(base,'Test')
val_dir = os.path.join(base,'Validation')
train_data = tf.keras.utils.image_dataset_from_directory(train_dir, labels='inferred', label_mode='categorical', 
                                                         class_names = ['Healthy', 'Powdery', 'Rust'], batch_size = 32, 
                                                         image_size = (256, 256), shuffle = True, seed = seed, 
                                                         validation_split = 0)
test_data = tf.keras.utils.image_dataset_from_directory(test_dir, labels='inferred', label_mode='categorical', 
                                                         class_names = ['Healthy', 'Powdery', 'Rust'], batch_size = 32, 
                                                         image_size = (256, 256), shuffle = True, seed = seed, 
                                                         validation_split = 0)
validation_data = tf.keras.utils.image_dataset_from_directory(val_dir, labels='inferred', label_mode='categorical', 
                                                         class_names = ['Healthy', 'Powdery', 'Rust'], batch_size = 32, 
                                                         image_size = (256, 256), shuffle = True, seed = seed, 
                                                         validation_split = 0)