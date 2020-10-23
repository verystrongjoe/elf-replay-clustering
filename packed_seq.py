import numpy as np
import os
import torch

model_name = "model_45252"
style = "simple"   # or hit_run
frame_skip = 50

load_dir = f"D:/workspace/clustering/{model_name}/save_npz/{style}/{frame_skip}"
load_files = os.listdir(load_dir)
samples = np.load(load_dir + "/" + load_files[0])

print(samples['state'].sum(1))