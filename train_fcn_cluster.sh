#@IgnoreInspection BashAddShebang
#   This is the most basic QSUB file needed for this cluster.
#   Further examples can be found under /share/apps/examples
#   Most software is NOT in your PATH but under /share/apps
#
#   For further info please read http://hpc.cs.ucl.ac.uk
#   For cluster help email cluster-support@cs.ucl.ac.uk
#
#   NOTE hash dollar is a scheduler diredctive not a comment.


# These are flags you must include - Two memory and one runtime.
# Runtime is either seconds or hours:min:sec

#$ -S /bin/bash
#$ -l tmem=8G
# -l h_vmem=2G -- with pytorch, omit this flag
#$ -l h_rt=25:00:00
#$ -l gpu=true
# -pe gpu 2
# -R y
# -l hostname=gonzo*
#These are optional flags but you problably want them in all jobs

#$ -j y
#$ -N train_fcn8s
#$ -o /cluster/project9/echiou_domainAdaptation/segmentation


hostname
date

export PATH=/home/echiou/miniconda2/bin:$PATH

source activate python3.6_pytorch0.4.1_cuda92

python /cluster/project9/echiou_domainAdaptation/segmentation/train_fcn.py \
    --outdir results/cityscapes/cityscapes_fcn8s \
    --model fcn8s \
    --num_cls 19 \
    --lr 1e-3 \
    --batch_size 6 \
    --momentum 0.9 \
    --crop_size 600 \
    --iterations 100000 \
    --datadir '/cluster/project9/echiou_domainAdaptation/datasets/gta_cityscapes/' \
    --dataset cityscapes
