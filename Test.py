import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from os import listdir
from scipy import misc
from lib.pvt import PolypPVT
from utils.dataloader import test_dataset
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--pth_path', type=str, default=f'/content/drive/MyDrive/BTP/pretrained_path/PolypPVT-best.pth')
    opt = parser.parse_args()
    model = PolypPVT()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()
   
    for _data_name in ['TestA', 'TestB', 'Val']:   #'CVC-300'

        ##### put data_path here #####    #
        data_path = './dataset/TestDataset/{}'.format(_data_name)
        ##### save_path #####
        save_path = '/content/drive/MyDrive/BTP/result_map/PolypPVT/{}/'.format(_data_name)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        num1 = len(os.listdir(gt_root))
        test_loader = test_dataset(image_root, gt_root, 352)
        for i in range(num1):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            P1,P2 = model(image)
            res = F.upsample(P1+P2, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            cv2.imwrite(save_path+name, res*255)
        print(_data_name, 'Finish!')
        
#     from google.colab.patches import cv2_imshow


    # Read the image
    # for img in
    # img1 = cv2.imread('/content/drive/MyDrive/BTP/GlaS/images/testA_1.png')  
    # cv2_imshow(img1)

    path = '/content/drive/MyDrive/BTP/GlaS/images/'
    i = 0
    for f in os.listdir(path):
      if i == 5:
        break
      ipath = path + f
      mpath = path[0:-7] + 'masks/' + f
      ppath = path[0:-12] + 'result_map/PolypPVT/GlaS/' + f
      img1 = cv2.imread(ipath)
      half1 = cv2.resize(img1, (0, 0), fx = 0.4, fy = 0.4)
      img2 = cv2.imread(mpath)
      half2 = cv2.resize(img2, (0, 0), fx = 0.4, fy = 0.4)
      img3 = cv2.imread(ppath)
      half3 = cv2.resize(img3, (0, 0), fx = 0.4, fy = 0.4)
      print("              Image                                   Ground Truth                              Predicted")
      Hori = np.concatenate((half1, half2, half3), axis=1)
      cv2.imshow(Hori)
      i=i+1
        ############### visualize in tabular manner and show  final dice score for test data   ########################
        
#        inputs, masks = next(iter(val_loader))
#        output = ((torch.sigmoid(model(inputs.to('cuda')))) >0.5).float()
#        _, ax = plt.subplots(2,3, figsize=(15,10))
#        for k in range(2):
#            ax[k][0].imshow(inputs[k].permute(1,2,0))
#            ax[k][1].imshow(output[k][0].cpu())
#            ax[k][2].imshow(masks[k])
