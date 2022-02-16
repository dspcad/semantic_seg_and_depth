import h5py
import numpy as np
from PIL import Image, ImageOps

import scipy.io

from tqdm import tqdm
import matplotlib.pyplot as plt


def rotate_image(image):
    return image.rotate(-90, expand=True)


if __name__ == "__main__":
    mat_path   = '/home/hhwu/project/semantic_seg_and_depth/dataset/nyu_depth_v2_labeled.mat'
    cls_path   = '/home/hhwu/project/semantic_seg_and_depth/dataset/classMapping40.mat'
    split_path = '/home/hhwu/project/semantic_seg_and_depth/dataset/splits.mat'
  
    h5_file = h5py.File(mat_path, mode='r')
    color_maps = h5_file['images']
    depth_maps = h5_file['depths']
    label_maps = h5_file['labels']

    #h5_40_cls = h5py.File(cls_path, mode='r')
    h5_40_cls = scipy.io.loadmat(cls_path)
    split_data = scipy.io.loadmat(split_path)
    print(split_data.keys())
    #print(len(split_data['trainNdxs'].flatten()))
    #print(len(split_data['testNdxs'].flatten()))



    tbl_40_cls = h5_40_cls['mapClass'][0]
    tbl_40_cls = np.insert(tbl_40_cls,0,0)
    print(h5_40_cls['className'])
    print(tbl_40_cls)
    print(len(tbl_40_cls))


    #################################
    #    Generate Training Data     #
    #################################
    print("Generating the training data...")
    num=0
    for idx in tqdm(split_data['trainNdxs'].flatten()):
        idx = idx-1
        cur_color_map = color_maps[idx]
        cur_color_map = np.moveaxis(cur_color_map, 0, -1)
        color_image = Image.fromarray(cur_color_map, mode='RGB')
        color_image = rotate_image(color_image)


        #print(depth_maps[idx].shape)

        #plt.imshow(color_image)
        #plt.show()
        depth_image = np.rot90(depth_maps[idx],3)
        #print(depth_image.shape)
        #plt.imshow(depth_image)
        #plt.show()

        label_image = np.rot90(label_maps[idx],3)
        h, w = label_image.shape
        for i in range(h):
            for j in range(w):
                #print(f"org: {i},{j}:   {label_image[i][j]}")
                label_image[i][j] = tbl_40_cls[label_image[i][j]]
                #print(f"                {label_image[i][j]}")

        #label_image = Image.fromarray(cur_label_map, mode='I;16')
        #label_image = rotate_image(label_image)

        
        color_image.save(f"train/images/train_img_{num}.png")
        np.save(f"train/label_depth/depth_{num}.npy", depth_image)
        #depth_image = np.load(f"train/label_depth/depth_{num}.npy")

        #plt.imshow(depth_image)
        #plt.show()
        np.save(f"train/label_semantic_seg/semantic_seg_{num}.npy", label_image)     
        num+=1


    #################################
    #       Generate Val Data       #
    #################################
    print("Generating the val data...")
    num=0
    for idx in tqdm(split_data['testNdxs'].flatten()):
        idx = idx-1
        cur_color_map = color_maps[idx]
        cur_color_map = np.moveaxis(cur_color_map, 0, -1)
        color_image = Image.fromarray(cur_color_map, mode='RGB')
        color_image = rotate_image(color_image)

        depth_image = np.rot90(depth_maps[idx],3)
        #plt.imshow(cur_depth_map.T)
        #plt.show()

        label_image = np.rot90(label_maps[idx],3)
        h, w = label_image.shape
        #print(f"h: {h}   w: {w}")
        for i in range(h):
            for j in range(w):
                label_image[i][j] = tbl_40_cls[label_image[i][j]]

        #label_image = Image.fromarray(cur_label_map, mode='I;16')
        #label_image = rotate_image(label_image)

        
        color_image.save(f"val/images/train_img_{num}.png")
        np.save(f"val/label_depth/depth_{num}.npy", depth_image)
        #depth_image = np.load(f"val/label_depth/depth_{num}.npy")

        #plt.imshow(depth_image)
        #plt.show()
        np.save(f"val/label_semantic_seg/semantic_seg_{num}.npy", label_image)     
        num+=1



