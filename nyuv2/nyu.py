import h5py
import numpy as np
from PIL import Image


#import sys
#np.set_printoptions(threshold=sys.maxsize)
import torch
torch.set_printoptions(profile="full")


def rotate_image(image):
    return image.rotate(-90, expand=True)

class LabeledDataset:
    """Python interface for the labeled subset of the NYU dataset.

    To save memory, call the `close()` method of this class to close
    the dataset file once you're done using it.
    """

    def __init__(self, path,transform=None):
        """Opens the labeled dataset file at the given path."""
        self.file = h5py.File(path, mode='r')
        print(self.file.keys())
        self.color_maps = self.file['images']
        self.depth_maps = self.file['depths']
        self.label_maps = self.file['labels']
        self.id_maps    = self.file['namesToIds']

    
        #strlist = [''.join([chr(v[0]) for v in self.file[obj_ref]]) for obj_ref in self.file['names'][0]]
        #print(strlist)
        #print(self.id_maps)
        #print(''.join(chr(v[0]) for v in self.file[self.id_maps[0]]))



        self.transform = transform

    def close(self):
        """Closes the HDF5 file from which the dataset is read."""
        self.file.close()

    def __len__(self):
        return len(self.color_maps)

    def __getitem__(self, idx):
        color_map = self.color_maps[idx]
        color_map = np.moveaxis(color_map, 0, -1)
        color_image = Image.fromarray(color_map, mode='RGB')
        color_image = rotate_image(color_image)

        depth_map = self.depth_maps[idx]
        depth_image = Image.fromarray(depth_map, mode='F')
        depth_image = rotate_image(depth_image)

        label_map = self.label_maps[idx]
        label_image = Image.fromarray(label_map, mode='I;16')
        #label_image = Image.fromarray(label_map, mode='I')
        label_image = rotate_image(label_image)


        if self.transform:
            color_image = self.transform(color_image)
            depth_image = self.transform(depth_image)
            label_image = self.transform(label_image)

        #print(label_image)
        return color_image, label_image
