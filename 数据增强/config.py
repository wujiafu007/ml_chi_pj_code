label_path='./fabric_data/label_json/'
temp_path='./fabric_data/temp/'
trgt_path='./fabric_data/trgt/'
data_path='./fabric_data/data/'
data_da_path='./fabric_data/data_da/'
train_data_path='./fabric_data/train/'
test_data_path='./fabric_data/test/'
tensor_path='./fabric_data/tensor/'
res_save_path='./fabric_data/result/'
image_size=(224,224)
random_seed=508
trainset_ratio=1
epoch=15
learning_rate=2e-5
batch_size=8
k_fold=5
da_times=4
weight=()

# da_strategy=[
# 1:pad
# 2:resize
# 3:centercrop
# 4:fivecrop
# 5:grayscale
# 6:color jitter
# 7:gaussian blur
# 8:random perspective
# 9:random rotation
# 10:random affine
# 11:random crop
# 12:randomresizedcrop
# 13:random invert
# 14:random posterize
# 15:random solarize
# 16:random adjust sharpness
# 17:random autocontrast
# 18:random equalize
# 19:auto guament
# 20:rand augment
# 21:trivial augment wide
# 22:random horizontalflip
# 23:random vertical flip
# ]
