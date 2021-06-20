# Medical_Image_Segmentation
This repo is our team project for SJTU CS420. Generally speaking, this repo tries to do segementation for medical image.We implement five segmentation model:
1. U-net
2. MultiRes U-net
3. Nested U-net
4. Attention U-net
5. Ce-net

## Environment Require
```
torch
torchvision
numpy
tqdm
imgaug
pillow
```

## Results
![results shown in table](figures/result_table.png)

![results shown in figure](figures/result_model.png)

## How to add extra model

1. Add model in model directory
2. Register model in utils/get_model.py
3. Modify the parser in config/model_config.py and write your script train.sh predict.sh in /scripts
4. You need to modify parser in predict.py
