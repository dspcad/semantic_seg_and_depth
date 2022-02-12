import numpy as np
from PIL import Image
import os, torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

class NyuV2Dataset:
    """Python interface for the labeled subset of the NYU dataset.

    To save memory, call the `close()` method of this class to close
    the dataset file once you're done using it.
    """

    def __init__(self, image_dir_path, label_depth_dir_path, label_semantic_seg_dir_path, transform=None):
        """Opens the labeled dataset file at the given path."""

        self.image_dir_path              = image_dir_path
        self.label_depth_dir_path        = label_depth_dir_path
        self.label_semantic_seg_dir_path = label_semantic_seg_dir_path


    
        #strlist = [''.join([chr(v[0]) for v in self.file[obj_ref]]) for obj_ref in self.file['names'][0]]
        #print(strlist)
        #print(self.id_maps)
        #print(''.join(chr(v[0]) for v in self.file[self.id_maps[0]]))


        #print(self.image_dir_path)
        #print(len(os.listdir(self.image_dir_path)))
        self.transform = transform


    def __len__(self):
        return len(os.listdir(self.image_dir_path))

    def __getitem__(self, idx):
        img_fname = f"train_img_{idx}.png"
        depth_fname = f"depth_{idx}.npy"
        semantic_seg_fname = f"semantic_seg_{idx}.npy"
        image        = Image.open(os.path.join(self.image_dir_path, img_fname))
        depth        = np.load(os.path.join(self.label_depth_dir_path, depth_fname))
        semantic_seg = np.int16(np.load(os.path.join(self.label_semantic_seg_dir_path, semantic_seg_fname)))


        #plt.imshow(image)
        #plt.show()
        if self.transform:
            image        = self.transform(image)

        #print(label_image)
        return image, semantic_seg


###########################
#     Test the code       #
###########################
if __name__ == '__main__':
    transform_train = transforms.Compose([transforms.ToTensor()])
    nyu_v2_train = NyuV2Dataset("/home/hhwu/project/semantic_seg_and_depth/nyuv2/train/", "/home/hhwu/project/semantic_seg_and_depth/nyuv2/label_depth/", "/home/hhwu/project/semantic_seg_and_depth/nyuv2/label_semantic_seg/",transform=transform_train)

    nyu_v2_train_sampler = torch.utils.data.RandomSampler(nyu_v2_train)
    nyu_v2_train_dataloader = DataLoader(nyu_v2_train, batch_size=1, shuffle=False, sampler=nyu_v2_train_sampler, num_workers=0)

    nyu_v2_progress_bar = tqdm(nyu_v2_train_dataloader)
    nyu_v2_itr = iter(nyu_v2_progress_bar)

    img,label = next(nyu_v2_itr)

