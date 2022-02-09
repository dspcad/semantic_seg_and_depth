from nyuv2 import *
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torchvision
from tqdm import tqdm
import time

from torchvision.utils import save_image


if __name__ == "__main__":

    transform_train = transforms.Compose([transforms.ToTensor()])

    nyu_v2_train = LabeledDataset('/home/hhwu/project/nyuv2-python-toolbox/dataset/nyu_depth_v2_labeled.mat',transform_train)
    nyu_v2_train_sampler = torch.utils.data.RandomSampler(nyu_v2_train)


    nyu_v2_train_dataloader = DataLoader(nyu_v2_train, batch_size=1, shuffle=False, sampler=nyu_v2_train_sampler, num_workers=8)

    nyu_v2_progress_bar = tqdm(nyu_v2_train_dataloader)
    nyu_v2_itr = iter(nyu_v2_progress_bar)


    model = torchvision.models.segmentation.fcn_resnet50(pretrained=False, num_classes=41)
    # set computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model to eval() model and load onto computation devicce
    model.eval().to(device)
    print(model)

    

    params = [p for p in model.parameters() if p.requires_grad]
    learning_rate = 0.001
    semantic_seg_optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0001)
    semantic_seg_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(semantic_seg_optimizer, T_max=len(nyu_v2_train_dataloader), eta_min=1e-6)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0) # Set loss function

    cnt=0
    h=480
    w=640
    for epoch in range(20):
        depth_data_available = True
        t_data_0 = time.time()
        while depth_data_available:
            ########################################
            #          Depth Estimation            #
            ########################################
            try:
                img,label = next(nyu_v2_itr)
                t_data_1 = time.time()
                img = img.cuda()
                label = label.cuda()
                #print(len(data_label))
    
                #img   = img[0]
                label = label[0]
                #print(img.shape)
                #print(label.shape)
    
                #_,h,w = img.shape
    
                for i in range(h):
                    for j in range(w):
                        label[0][i][j] = 0 if label[0][i][j] <=0 or label[0][i][j]>40 else label[0][i][j]
                #            img[0][i][j] = 0
                #            img[1][i][j] = 0
                #            img[2][i][j] = 0
    
                #save_image(img, f"test_{cnt}.png")

                t_net_0 = time.time()
                pred = model(img)['out']
                #print(type(outputs))
                #print(outputs['out'].shape)

                seg_loss = criterion(pred,label.long())
                #print(seg_loss.item())
                semantic_seg_optimizer.zero_grad()
                seg_loss.backward()
                semantic_seg_optimizer.step()
                semantic_seg_scheduler.step()
                t_net_1 = time.time()



                if hasattr(nyu_v2_progress_bar,'set_postfix'):
                    nyu_v2_progress_bar.set_postfix(loss = '%.3f' % float(seg_loss), lr= '%.8f' % float(semantic_seg_scheduler.get_last_lr()[0]), cur_loss= '%.3f' % float(seg_loss),
                                            data_time = '%.3f' % float(t_data_1 - t_data_0),
                                            net_time = '%.3f' % float(t_net_1 - t_net_0))



                t_data_0 = time.time()
    
            except StopIteration:
                depth_data_available = False  # 
                print('all depth obj detect samples iterated')
                del depth_itr
                break
    
