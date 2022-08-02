""" 
Lin Zhang - ETH Zurich
"""
import os
import glob
from PIL import Image

def process_data(basedir, class_name):

    savedir = os.path.join(basedir, "val", class_name)
    os.makedirs(savedir, exist_ok=True)

    print("Val save path: %s" % savedir)

    img_paths = os.path.join(basedir, "training", class_name) + "/*.jpeg"
    img_paths = glob.glob(img_paths)
    img_paths = sorted(img_paths)
    img_paths = img_paths[:round(len(img_paths)*0.1)]

    for i, img_path in enumerate(img_paths):
        new_path = os.path.join(savedir, os.path.split(img_paths[i])[-1])
        os.rename(img_path, new_path)
            
process_data("data/", "cats")
process_data("data/", "birds")