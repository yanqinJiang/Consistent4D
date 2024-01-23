#!\bin\bash
name=$1
GPU_ID=$2

export CUDA_VISIBLE_DEVICES=$GPU_ID
python train.py --dataroot ./datasets/${name} --name ${name}_pix2pix --model pix2pix --direction BtoA --batch_size 1 --seq_length 16 --lr 0.002 \
    --n_epochs 100 --n_epochs_decay 100