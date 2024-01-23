#\bin\bash
name=$1
GPU_ID=$2
epoch=$3

export CUDA_VISIBLE_DEVICES=$GPU_ID

python test.py --dataroot ./test_datasets/${name} --name ${name}__pix2pix --model test --netG unet_256 --direction BtoA --dataset_mode single --norm batch \
--seq_length 16 --batch_size 1 --epoch ${epoch}