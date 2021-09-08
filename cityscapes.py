import os.path
import sys 
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms 

import numpy as np
import torch.utils.data as data
from PIL import Image

from .data_loader import register_data_params, register_dataset_obj
from .data_loader import DatasetParams

ignore_label = 255 #all of the ignored classes are set to pixel value 255
id2label = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
            3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
            7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
            14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
            18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
            28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
            220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
            0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
classes = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
        'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        'bicycle'] #19 classes in total





def remap_labels_to_train_ids(arr): #krataw ta 18 prwta noumera (exw 19 klaseis)
    out = ignore_label * np.ones(arr.shape, dtype=np.uint8)
    for id, label in id2label.items():
        out[arr == id] = int(label)
    return out

@register_data_params('cityscapes')
class CityScapesParams(DatasetParams):
    num_channels = 3
    image_size   = 1024
    mean         = 0.5
    std          = 0.5
    num_cls      = 19
    target_transform = None


@register_dataset_obj('cityscapes')
class Cityscapes(data.Dataset):
  #root: /content/gdrive/MyDrive/cityscapes

    def __init__(self, root, split= 'train',remap_labels=True, transform=(None, None),
                 target_transform=None): 
                 
        self.root = root
        sys.path.append(root)
        self.split = split
        self.remap_labels = remap_labels
        self.ids = self.collect_ids()
        self.transform1= transform[0]
        self.transform2= transform[1]
        
        self.target_transform = target_transform
        self.num_cls = 19
        
        self.id2label = id2label
        self.classes = classes
        

    def collect_ids(self): 
      
        #root: /content/gdrive/MyDrive/cityscapes
        
        im_dir = os.path.join(self.root, 'leftImg8bit', self.split) 
      
        
        
        
        
        ids = []
        

        for dirpath, dirnames, filenames in os.walk(im_dir): #os.walk returns a generator, that creates 
        #a tuple of values (current_path, directories in current_path, files in current_path)
        #every time the generator is called it will follow each directory recursively until no further
        #sub-directories are available from the initial directory that walk was called upon 
      
        #os.walk: parameters: top - each directory rooted at directory, yields 3-tuples (dirpath, dirnames, filenames)
        
        #dirpath is the path to the directory
        #dirnames is a list of the names of the subdirectories in dirpath excluding '.' and '..'
          
          for filename in filenames:
            if filename.endswith('.png'):
              
              ids.append('_'.join(filename.split('_')[:3]))
       
        return ids



    def img_path(self, id): #an dwsw ws dataroot to cityscapes folder
        fmt = 'leftImg8bit/{}/{}/{}_leftImg8bit.png' 
        subdir = id.split('_')[0] 
        path = fmt.format(self.split, subdir, id) 
        return os.path.join(self.root, path) 
      

    def label_path(self, id):
        fmt = 'gtFine_lab/{}/{}/{}_gtFine_labelIds.png' 
        subdir = id.split('_')[0]
        path = fmt.format(self.split, subdir, id)
        return os.path.join(self.root, path)


  
    
    def __getitem__(self, index):
        
        id = self.ids[index]
        
        img = Image.open(self.img_path(id)).convert('RGB') 
        plt.imshow(img)
        plt.savefig('cityscapes_get_item_img.png')

      
      
        if self.transform1 is not None: 
            img1 = self.transform1(img) 
            
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img1_after_transf= img1.clone()
        img1_after_transf= img1_after_transf.numpy().transpose([1,2,0]) * std + mean  #(512, 1024, 3)
        plt.figure()
        plt.imshow(img1_after_transf)
        plt.savefig('cityscapes_get_item_after_transf_img1.png')
 
        
        if self.transform2 is not None: 
            img2 = self.transform2(img)
        mean2= [0.5, 0.5, 0.5]
        std2= [0.5, 0.5, 0.5]
        img2_after_transf= img2.clone()
        img2_after_transf= img2_after_transf.numpy().transpose([1,2,0]) * std2 + mean2
        plt.figure()
        plt.imshow(img2_after_transf)
        plt.savefig('cityscapes_get_item_after_transf_img2.png')
 

      
        
        target = Image.open(self.label_path(id)).convert('L') 
        if self.remap_labels:
            target = np.asarray(target) 
            target = remap_labels_to_train_ids(target) 
            target = Image.fromarray(np.uint8(target), 'L') 
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img1, img2, target
        #img1: image from the 1st transformation
        #img2: image from the 2nd transformation
        #target: label 
        
        

    def __len__(self):
        return len(self.ids)


if __name__ == '__main__':
    cs = Cityscapes('/x/CityScapes')
