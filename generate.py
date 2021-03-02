import os
import logging

import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(format='[%(levelname)s]\t: %(message)s',
                    level=logging.INFO)

"""
small_random (400000 * 100): 1 * 32 * 32 ; 1.1 KB/sample
medium_random (100000 * 100): 3 * 64 * 64; 13 KB/sample
large_random (100000 * 10): 3 * 160 * 160; 76 KB/sample
"""

def write_image(data_path, num_class, num_per_class, img_size, grey=False):
    if os.path.exists(data_path):
        logging.warning(f"Path {data_path} exsited!")
        return
    os.makedirs(data_path, exist_ok=True)
    logging.warning(f"Generating dataset in {data_path}...")
    for class_idx in tqdm(range(num_class)):
        class_path = os.path.join(data_path, str(class_idx))
        os.makedirs(class_path, exist_ok=True)
        for sample_idx in range(num_per_class):
            img_shape = (img_size,img_size,3)
            if grey:
                img_shape = (img_size,img_size)
            data = np.random.random(img_shape)
            rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
            im = Image.fromarray(rescaled)
            im.save(os.path.join(class_path, f'{sample_idx}.png'))

def main():
    write_image("./data/small_random", 20, 10000, 32, grey=True)
    write_image("./data/medium_random", 20, 5000, 64)
    write_image("./data/large_random", 10, 5000, 160)


if __name__ == '__main__':
    main()