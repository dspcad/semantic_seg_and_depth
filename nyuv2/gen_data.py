import h5py
import numpy as np
from PIL import Image, ImageOps

import scipy.io

from tqdm import tqdm
import matplotlib.pyplot as plt


def rotate_image(image):
    return image.rotate(-90, expand=True)


if __name__ == "__main__":
    mat_path = '/home/hhwu/project/semantic_seg_and_depth/dataset/nyu_depth_v2_labeled.mat'
    cls_path = '/home/hhwu/project/semantic_seg_and_depth/dataset/classMapping40.mat'
  
    h5_file = h5py.File(mat_path, mode='r')
    color_maps = h5_file['images']
    depth_maps = h5_file['depths']
    label_maps = h5_file['labels']

    #h5_40_cls = h5py.File(cls_path, mode='r')
    h5_40_cls = scipy.io.loadmat(cls_path)
    #print(h5_40_cls.keys())
    print(h5_40_cls['className'])
    #print(len(h5_40_cls['mapClass'][0]))
    tbl_40_cls = h5_40_cls['mapClass'][0]
    #tbl_40_cls = np.concatenate([[0],tbl_40_cls])
    tbl_40_cls = np.insert(tbl_40_cls,0,0)
    print(tbl_40_cls)
    print(len(tbl_40_cls))


    for idx in tqdm(range(len(color_maps))):
        cur_color_map = color_maps[idx]
        cur_color_map = np.moveaxis(cur_color_map, 0, -1)
        color_image = Image.fromarray(cur_color_map, mode='RGB')
        color_image = rotate_image(color_image)

        depth_image = depth_maps[idx].T
        #plt.imshow(cur_depth_map.T)
        #plt.show()

        label_image = label_maps[idx].T
        h, w = label_image.shape
        #print(f"h: {h}   w: {w}")
        for i in range(h):
            for j in range(w):
                label_image[i][j] = tbl_40_cls[label_image[i][j]]

        #label_image = Image.fromarray(cur_label_map, mode='I;16')
        #label_image = rotate_image(label_image)

        
        color_image.save(f"train/train_img_{idx}.png")
        np.save(f"label_depth/depth_{idx}.npy", depth_image)
        depth_image = np.load(f"label_depth/depth_{idx}.npy")

        #plt.imshow(depth_image)
        #plt.show()
        np.save(f"label_semantic_seg/semantic_seg_{idx}.npy", label_image)     
        #depth_image.save(f"label_depth/depth_{idx}.png")       
