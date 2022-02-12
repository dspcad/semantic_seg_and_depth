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

if __name__ == "__main__":

    transform_train = transforms.Compose([transforms.ToTensor()])
    nyu_v2_train = NyuV2Dataset("/home/hhwu/project/semantic_seg_and_depth/nyuv2/train/", "/home/hhwu/project/semantic_seg_and_depth/nyuv2/label_depth/", "/home/hhwu/project/semantic_seg_and_depth/nyuv2/label_semantic_seg/",transform=transform_train)
    nyu_v2_train_sampler = torch.utils.data.RandomSampler(nyu_v2_train)


    batch_size = 1
    distributed = False
    save_path = "/home/hhwu/project/semantic_seg_and_depth/saved_model/"
    nyu_v2_train_dataloader = DataLoader(nyu_v2_train, batch_size=batch_size, shuffle=False, sampler=nyu_v2_train_sampler, num_workers=8)
    nyu_v2_val_dataloader = DataLoader(nyu_v2_train, batch_size=8, shuffle=False, num_workers=8)



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torchvision.models.segmentation.fcn_resnet50(pretrained=False, num_classes=41).to(device)
    print(model)

    

    params = [p for p in model.parameters() if p.requires_grad]
    learning_rate = 1e-5
    semantic_seg_optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0001)
    #semantic_seg_optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=0.0001)
    semantic_seg_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(semantic_seg_optimizer, T_max=len(nyu_v2_train_dataloader))
    #semantic_seg_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(semantic_seg_optimizer, T_max=len(nyu_v2_train_dataloader), eta_min=1e-6)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0) # Set loss function

    seg_ave_loss = 0.0
    h=480
    w=640
    for epoch in range(101):
        depth_data_available = True
        t_data_0 = time.time()

        nyu_v2_progress_bar = tqdm(nyu_v2_train_dataloader)
        nyu_v2_itr = iter(nyu_v2_progress_bar)
        b_seg_idx  = 0

        cnt=0
        while depth_data_available:
            ########################################
            #          Depth Estimation            #
            ########################################
            try:
                img, target = next(nyu_v2_itr)
                t_data_1 = time.time()
                img = img.cuda()
                target = target.cuda()
                global_step = epoch * len(nyu_v2_train_dataloader) + b_seg_idx
                                
                target = target.reshape(batch_size,h,w)
                #save_image(img, f"test_{cnt}.png")

                t_net_0 = time.time()
                pred = model(img)['out']

                #print(type(outputs))
                #print(outputs['out'].shape)

                #print(f"label: {target.shape}")
                #print(f"pred: {pred.shape}")
                seg_loss = criterion(pred,target.long())
                #print(seg_loss.item())
                semantic_seg_optimizer.zero_grad()
                seg_loss.backward()
                semantic_seg_optimizer.step()
                semantic_seg_scheduler.step()
                t_net_1 = time.time()


                seg_ave_loss = (seg_ave_loss*global_step + float(seg_loss) )/(global_step+1)



                if hasattr(nyu_v2_progress_bar,'set_postfix'):
                    nyu_v2_progress_bar.set_postfix(loss = '%.3f' % float(seg_ave_loss), lr= '%.8f' % float(semantic_seg_scheduler.get_last_lr()[0]), cur_loss= '%.3f' % float(seg_loss),
                                            data_time = '%.3f' % float(t_data_1 - t_data_0),
                                            net_time = '%.3f' % float(t_net_1 - t_net_0))



                t_data_0 = time.time()
                b_seg_idx += 1
                
                #if cnt==5:
                #    break
 
                #cnt += 1    

            except StopIteration:
                depth_data_available = False  # 
                print('all depth obj detect samples iterated')
                del nyu_v2_itr
                break
        

        if epoch%10==0:
            save_model(model, semantic_seg_optimizer, epoch,save_path, distributed)

            model.eval()
            eval_res = JaccardIndex(num_classes=41, ignore_index=0).cuda()
            with torch.no_grad():
                for img, target in tqdm(nyu_v2_val_dataloader):
                    img = img.cuda()
                    target = target.cuda()
                    pred = model(img)['out']

                    #print(f"pred: {pred.shape}")
                    #print(f"target: {target.shape}")
                    #print(f"target: {target.shape}   {target[0][i][j]}")
                    eval_res.update(pred,target)
                    
                print(eval_res.compute())

            model.train()

