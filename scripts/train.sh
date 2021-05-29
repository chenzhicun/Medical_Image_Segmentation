export CUDA_VISIBLE_DEVICES=4
cd ..
python train.py --model unet\
    --epochs 50\
    --batch_size 1\
    --learning_rate 0.0001\
    --validation_percent 10.0\
    --exp_id fix_bug\
    --mask_threshold 0.5\
    --argument