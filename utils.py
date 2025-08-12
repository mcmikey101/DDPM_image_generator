import torch
import os
from PIL import Image
import numpy as np
import cv2

def get_values_by_index(vals, t, x_shape):
    batch_size = t.shape[0]
    out = torch.gather(vals, -1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
        
def save_model(model, model_save_dir):
    os.makedirs(os.path.dirname(model_save_dir), exist_ok=True)
    torch.save(model.state_dict(), model_save_dir)

def load_model(model, path, map_location=None):
    state_dict = torch.load(path, map_location=map_location)
    model.load_state_dict(state_dict)
    return model

def save_sample(sample, sample_save_dir):
    sample = sample.detach().cpu().squeeze(1)
    arr = cv2.cvtColor(np.array(sample.permute(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    img = Image.fromarray(arr)

    os.makedirs(os.path.dirname(sample_save_dir), exist_ok=True)
    img.save(os.path.join(sample_save_dir, str(len(os.listdir(sample_save_dir)) + 1) + ".jpg"))
