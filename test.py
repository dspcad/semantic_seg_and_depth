from nyuv2.nyu import NyuV2Dataset
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torchvision
from tqdm import tqdm
import time

from torchvision.utils import save_image
from utils import save_model

from torchmetrics import JaccardIndex
import argparse, os



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the FCN semantic segmentation model with NYU depth V2')
    parser.add_argument('--testBatchSize', type=int, default=8, help='testing batch size')
    parser.add_argument('--data_path', type=str, default='/home/hhwu/project/semantic_seg_and_depth/nyuv2/', help="path to the data folder")
    parser.add_argument('--dataset', type=int, default=0, help='0: semantic segmentation, 1: depth estimation')
    parser.add_argument('--model_path', type=str, default='/home/hhwu/project/semantic_seg_and_depth/saved_model/ep010.pth', help="path to the saved model folder")
    
    opt = parser.parse_args()
    print(opt)


    opt.train_data_path               = opt.data_path + 'train/'
    opt.train_label_semantic_seg_path = opt.data_path + 'label_semantic_seg/'
    opt.train_label_depth_path        = opt.data_path + 'label_depth/'


    transform_train = transforms.Compose([transforms.ToTensor()])
    nyu_v2_train = NyuV2Dataset(opt.train_data_path, opt.train_label_depth_path, opt.train_label_semantic_seg_path,transform=transform_train)
    nyu_v2_val_dataloader = DataLoader(nyu_v2_train, batch_size=8, shuffle=False, num_workers=8)

    distributed = False



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torchvision.models.segmentation.fcn_resnet50(pretrained=False, num_classes=41).to(device)
    model.load_state_dict(torch.load(opt.model_path)['model'])

    eval_res = JaccardIndex(num_classes=41, ignore_index=0, compute_on_step=False).cuda()
    with torch.no_grad():
        for img, target in tqdm(nyu_v2_val_dataloader):
            img = img.cuda()
            target = target.cuda()
            pred = model(img)['out']

            eval_res.update(pred,target)
            
        print(eval_res.compute())

