"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer
import argparse
from torch.autograd import Variable
from trainer import MUNIT_Trainer, UNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os #module that involves a function to create folders on the system, allows to interact with the operating system
import sys
import tensorboardX
import shutil



#train.py file: contains the for loop for iterations
parser = argparse.ArgumentParser() #module that automatically generates help and usage messages and issues errors when users give the program invalid arguments
#added the add_argument() method to specify which command-line options the program is willing to accept 
parser.add_argument('--config', type=str, default='configs/edges2handbags_folder.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument('--vgg_path', type=str, default='.', help="outputs path")i
parser.add_argument("--resume", action="store_true")
#parser.add_argument('--checkpoint_dir',default='.')
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
opts = parser.parse_args() #parse_args() method returns some data from the options specified 

cudnn.benchmark = True #causes cuDNN to benchmark multiple convolution algorithms and select the fastest.


# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']
#config['vgg_model_path'] = opts.output_path
config['vgg_model_path'] = opts.vgg_path


# Setup model and data loader
if opts.trainer == 'MUNIT':
    trainer = MUNIT_Trainer(config) #initialises network
elif opts.trainer == 'UNIT':
    trainer = UNIT_Trainer(config)
else:
    sys.exit("Only support MUNIT|UNIT")
trainer.cuda()
train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)
print(display_size)
#torch.stack concatenates a sequence of tensors along a new dimension, all tensors need to be of the same size 
train_display_images_a = torch.stack([train_loader_a.dataset[i] for i in range(display_size//2)]).cuda()
train_display_images_b = torch.stack([train_loader_b.dataset[i] for i in range(display_size//2)]).cuda()
test_display_images_a = torch.stack([test_loader_a.dataset[i] for i in range(display_size//2)]).cuda()
test_display_images_b = torch.stack([test_loader_b.dataset[i] for i in range(display_size//2)]).cuda()

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
#os.path.join combines path names into one complete path
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

# Start training
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0 #checkpoints: the weights of the model 
#stops training after the specified number of iterations has been reached 
while True: 
    for it, (images_a, images_b) in enumerate(zip(train_loader_a, train_loader_b)):
        trainer.update_learning_rate()
        images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()
        #In order to enable automatic differentiation, PyTorch keeps track 
        #of all operations involving tensors for which the gradient may need to be computed (i.e., require_grad is True). 
        #The operations are recorded as a directed graph. The detach() method constructs a new view on a tensor which is declared 
        #not to need gradients, i.e., it is to be excluded from further tracking of operations, and therefore the subgraph involving 
        #this view is not recorded.

#This can be easily visualised using the torchviz package. Here is a simple fragment showing a set operations for which the gradient can be computed with respect to the input tensor x.
        with Timer("Elapsed time in update: %f"):
            # Main training code
            trainer.dis_update(images_a, images_b, config)
            trainer.gen_update(images_a, images_b, config)
            torch.cuda.synchronize()

        # Dump training stats in log file (save)
        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, train_writer)

        # Write images
        if (iterations + 1) % config['image_save_iter'] == 0:
            with torch.no_grad(): #temporarily sets all of the requires_grad flags to false
                test_image_outputs = trainer.sample(test_display_images_a, test_display_images_b)
                train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
            write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
            write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
            # HTML
            write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')

        if (iterations + 1) % config['image_display_iter'] == 0:
            with torch.no_grad():
                image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
            write_2images(image_outputs, display_size, image_directory, 'train_current')

        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')

