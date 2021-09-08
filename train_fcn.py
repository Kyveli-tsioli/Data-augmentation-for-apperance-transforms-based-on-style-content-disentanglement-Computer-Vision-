import logging
import os.path
import sys
from collections import deque
import random #added this
import matplotlib.pyplot as plt #added this

from utils_MUNIT import * #added this


sys.path.append("../..")  # Adds higher directory to python modules path.
sys.path.append("..")  # Adds higher directory to python modules path.
sys.path.append('/content/gdrive/MyDrive/segmentation') 


import click #creates command line interface 
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from tensorboardX import SummaryWriter #enables to import the tensorboard class
#we will create instances of summarywriter and then add our model's evaluation features
#like loss, number of correct predictions, accuracy etc to it
#tensorboard takes the output tensors and displays the plot of all the metrics 
from PIL import Image

from data.data_loader import get_fcn_dataset as get_dataset 
from models import get_model
from transforms import augment_collate
from util import config_logging
from util import to_tensor_raw
from util import make_variable


def to_tensor_raw(im):
    return torch.from_numpy(np.array(im, np.int64, copy=False))

def supervised_loss(score, label, weights=None): #cross entropy loss
    loss_fn_ = torch.nn.NLLLoss2d(weight=weights, size_average=True, 
            ignore_index=255) #the negative log likelihood loss, for classification with C classes
    #weight argument: 1D tensor of size C assigning weights to each of the classes, this is useful for unbalanced datasets
    #the final softmax layer of the net have set a label ignore parameter for value 255 to ignore these classes
    loss = loss_fn_(F.log_softmax(score), label) #obtain log-probabilities by adding a log softmax layer in the last layer of the network
    #use CrossEntropyLoss instead if we prefer not to add an extra layer 
    
    #size average: by default (True) the losses are averaged over each loss element in the batch (per pixel loss??)
    return loss
 

@click.command()
@click.option('--outdir', default='.', type=str)
@click.option('--dataset', required=True, multiple=True)
@click.option('--datadir', default="", type=click.Path(exists=True))
@click.option('--batch_size', '-b', default=1)
@click.option('--lr', '-l', default=0.001)
@click.option('--p_synthetic','-p', default=0.8)
@click.option('--step', type=int)
@click.option('--iterations', '-i', default=100000)
@click.option('--momentum', '-m', default=0.9)
@click.option('--snapshot', '-s', default=5000)
@click.option('--downscale', type=int)
@click.option('--augmentation/--no-augmentation', default= False) 
@click.option('--fyu/--torch', default=False)
@click.option('--crop_size', default=720)
@click.option('--weights', type=click.Path(exists=True))
@click.option('--model', default='fcn8s')
@click.option('--num_cls', default=19, type=int)
@click.option('--gpu', default='0')


def main(outdir, dataset, datadir, batch_size, lr, p_synthetic, step, iterations,
        momentum, snapshot, downscale, augmentation, fyu, crop_size, 
        weights, model, gpu, num_cls):

  os.makedirs(outdir.split('/')[0] + '/' + outdir.split('/')[1], exist_ok=True) #recursive directory creation function

  if weights is not None:
      raise RuntimeError("weights don't work because eric is bad at coding")
  config_logging()
  
  logdir = 'runs/{:s}/{:s}'.format(model, '-'.join(dataset)) #runs/fcn8s/cityscapes
  writer = SummaryWriter(log_dir=logdir)
  net = get_model(model, num_cls=num_cls) 
  #print("net",net) #prints architecture of vgg16 and vgg_head and bilinear upsampling
  net.cuda()
  transform1=[] 
  transform2=[] 
  target_transform = [] 
  downscale=1 if downscale is None else downscale 
  
  
  
  #was transforms.Scale and changed it to Resize                                 
  transform1.extend([
      torchvision.transforms.Resize((512 // downscale,)), #was 1024
      net.transform1
      ])
  
  transform2.extend([
      torchvision.transforms.Resize((512 // downscale,)), #was 1024
      net.transform2 
      ]) 

  
  target_transform.extend([
      torchvision.transforms.Resize((512 // downscale,), interpolation=Image.NEAREST), #was 1024 and changed it to 512
      to_tensor_raw
      ])

  transform1 = torchvision.transforms.Compose(transform1) 
  transform2 = torchvision.transforms.Compose(transform2) 
  
  target_transform = torchvision.transforms.Compose(target_transform)
  

  
  dataset_ = get_dataset(dataset[0], os.path.join(datadir, dataset[0]), transform= (transform1, transform2),
                          target_transform=target_transform)
                          


  #print("length of dataset",len(dataset_)) #2975
  #print(len(dataset[0])) #10 ???
  
  

  if weights is not None:
      weights = np.loadtxt(weights)
      
  opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum,
                        weight_decay=0.0005)
  
  if augmentation: #simple dataset augmentation, only during training, flip kai crop
      collate_fn = lambda batch: augment_collate(batch, crop=crop_size, flip=True)
  else:
      collate_fn = torch.utils.data.dataloader.default_collate
      
  
  loader = torch.utils.data.DataLoader(dataset_, batch_size=batch_size,
                                          shuffle=True, num_workers=2,
                                          collate_fn= collate_fn, 
                                          pin_memory=True)
   
  #LOAD CHECKPOINTS
  from trainer_MUNIT import MUNIT_Trainer
  hyperparameters= get_config('gta2cityscapes_folder_MUNIT.yaml')

  class Opts:
    pass
  opts= Opts()
  opts.gpu_ids= [0]

  trainer= MUNIT_Trainer(hyperparameters, opts)
  trainer.resume('../segmentation/check', hyperparameters)
  trainer.eval() #eval mode to save memory (from weights and activations)
  

  #VISUALIZE AN ORIGINAL IMAGE
  test= loader.dataset[0][1] #[-1,1] #torch.Size([3, 512, 1024])
  test_tr= test.numpy().transpose([1,2,0])  #(512, 1024, 3)
  test_tr_rgb= (test_tr +1) * 0.5 *255 #bring it to [0,255]
  test_tr_color = np.asarray(test_tr_rgb, np.uint8)
 
  

  #PRODUCE A SYNTHETIC IMAGE
  content_code_real, _=trainer.gen_b.encode(loader.dataset[0][1].unsqueeze(0)) 
  #print("content_code_real size", content_code_real.size()) #torch.Size([1, 256, 128, 256]) 


  style_encoder= trainer.gen_b.enc_style(loader.dataset[0][1].unsqueeze(0)) #[-1,1]
  #print("style encoder", style_encoder.size()) #torch.Size([1, 8, 1, 1])

  display_size= batch_size #na simfwnei me batch size 
  style_dim=8
  styles_real= torch.randn(display_size,style_dim, 1, 1) 
  #styles_real= np.sqrt(10)*torch.randn(display_size,style_dim, 1, 1) 
  #styles_real= np.sqrt(500)*torch.randn(display_size,style_dim, 1, 1) 
  
  
  
  synthetic=trainer.gen_b.decode(content_code_real,styles_real) #torch.Size([1, 3, 512, 1024])
  synthetic_sq= synthetic.squeeze(0) #torch.Size([3, 512, 1024])
  synthetic_sq_cpu=synthetic_sq.cpu()
  synthetic_sq_tr= synthetic_sq_cpu.detach().numpy().transpose([1,2,0])
  synthetic_sq_tr_rgb= (synthetic_sq_tr +1) * 0.5 *255
  synthetic_color = np.asarray(synthetic_sq_tr_rgb, np.uint8)
 
  
  

  iteration = 0
  counter=0 #how many times synthetic image is constructed during training 
  losses = deque(maxlen=10)

  while True:
    
      for im1, im2, label in loader: 
          print("label", label.shape) #torch.Size([batch_size, 512, 1024]) #augmentation = False
          print("im2", im2.shape) #torch.Size([batch_size, 3, 512, 1024])
          #label torch.Size([batch_size, 512, 512]) if augmentation= True
          #im2 torch.Size([batch_size, 3, 512, 512]) if augmentation= True

          if torch.cuda.is_available():
            im1, label = im1.cuda(), label.cuda()

          # Clear out gradients
          opt.zero_grad()

          
          generate_prob=np.random.uniform(low = 0.0, high = 1.0, size = None)  
          #p_synthetic=0.8 #80% probability of producing a synthetic image 
        
      
          if (generate_prob < p_synthetic):
            counter +=1 

            
            
            
            styles_real_random= torch.randn(display_size,style_dim, 1, 1) 
            content_code_real,_= trainer.gen_b.encode(im2) #takes the current image from the dataloader
            synthetic_im2=trainer.gen_b.decode(content_code_real, styles_real)
            label_im2= label

        
            synthetic_im2_sq= synthetic_im2.squeeze(0)
            synthetic_im2_cpu=synthetic_im2_sq.cpu()
            synthetic_im2_tr= synthetic_im2_cpu.detach().numpy().transpose([1,2,0])
            synthetic_im2_tr_rgb= (synthetic_im2_tr +1) * 0.5 *255
            synthetic_im2_color = np.asarray(synthetic_im2_tr_rgb, np.uint8)
            
  
            
  
            im2 = make_variable(synthetic_im2,requires_grad=False) #check this, maybe we need the im1
            label = make_variable(label_im2, requires_grad=False) 
            
          else:
            im2 = make_variable(im2,requires_grad=False)
            label = make_variable(label, requires_grad=False)

         
          # forward pass and compute loss
          preds = net(im2) 
          loss = supervised_loss(preds, label)

          # backward pass
          loss.backward()
          
          # step gradients
          opt.step()

          losses.append(loss.item()) 

         

          # log results
          if iteration % 10 == 0:
              logging.info('Iteration {}:\t{}'
                              .format(iteration, np.mean(losses)))
              writer.add_scalar('loss', np.mean(losses), iteration)

          if iteration % snapshot == 0:
              torch.save(net.state_dict(),
                          '{}-iter{}.pth'.format(outdir, iteration))

          iteration += 1
          if iteration >= iterations:
              logging.info('Optimization complete.')
              print("counter",counter)
              break

      if iteration >= iteration:
          break

  #test= loader.dataset[0][1] 
  #print(net.forward(test))
        
         
if __name__ == '__main__':
    main()
