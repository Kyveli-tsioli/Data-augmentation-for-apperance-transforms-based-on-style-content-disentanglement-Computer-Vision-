python ./train_fcn.py \
    --outdir results/cityscapes/cityscapes_fcn8s \
    --model fcn8s \
    --num_cls 19 \
    --lr 1e-3 \
    --batch_size 6 \
    --momentum 0.9 \
    --crop_size 600 \
    --iterations 100000 \
    #--datadir '/media/echiou/DATA/Documents/gta_cityscapes/' \ #change this
    --datadir '/content/gdrive/MyDrive/cityscapes'
    #pou exw ton fakelo cityscapes 
    #kanei ena for loop se olous tous fakelous tou cityscapes kai tha fortwsei tis eikones
    
    --dataset cityscapes