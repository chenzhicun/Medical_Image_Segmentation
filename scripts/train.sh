export CUDA_VISIBLE_DEVICES=4
cd ..
python train.py -e 50 -b 1 -l 0.0001 -v 10 --exp_id test_argument --model unet