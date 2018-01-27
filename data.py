import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit, cross_val_score, train_test_split
from skimage.filters import gaussian

def transform(df):
    band_1 = np.array([np.array(row).reshape(75, 75) for row in df['band_1']])
    band_2 = np.array([np.array(row).reshape(75, 75) for row in df['band_2']])
    band_3 = band_1 * band_2
    band_4 = gaussian(band_1 + band_2 / 2)
    
    x = np.concatenate([band_2[:, :, :, np.newaxis], band_1[:, :, :, np.newaxis],\
                       band_3[:, :, :, np.newaxis], band_4[:, :, :, np.newaxis]], axis=-1)
    angle = np.array([np.array(row) for row in df['inc_angle']])
    
    return x, angle 

    
def augment(images):
    image_mirror_lr = []
    image_mirror_ud = []
    for i in range(0,images.shape[0]):
        band_1 = images[i,:,:,0]
        band_2 = images[i,:,:,1]
        band_3 = images[i,:,:,2]
        band_4 = images[i,:,:,3]

        # mirror left-right
        band_1_mirror_lr = np.flip(band_1, 0)
        band_2_mirror_lr = np.flip(band_2, 0)
        band_3_mirror_lr = np.flip(band_3, 0)
        band_4_mirror_lr = np.flip(band_4, 0)
        image_mirror_lr.append(np.stack((band_1_mirror_lr, band_2_mirror_lr, band_3_mirror_lr, band_4_mirror_lr), axis=-1))
        
        # mirror up-down
        band_1_mirror_ud = np.flip(band_1, 1)
        band_2_mirror_ud = np.flip(band_2, 1)
        band_3_mirror_ud = np.flip(band_3, 1)
        band_4_mirror_ud = np.flip(band_4, 1)
        image_mirror_ud.append(np.stack((band_1_mirror_ud, band_2_mirror_ud, band_3_mirror_ud, band_4_mirror_ud), axis=-1))
        
    mirrorlr = np.array(image_mirror_lr)
    mirrorud = np.array(image_mirror_ud)
    images = np.concatenate((images, mirrorlr, mirrorud))
    return images

    
def split_data(imgs, angls, lbls, seed):
    rs = ShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, val_idx = next(rs.split(imgs))
    train_im_1, train_im_2, train_ang, train_y = imgs[tr_idx, :, :, :2], imgs[tr_idx, :, :, 2:], angls[tr_idx], lbls[tr_idx]
    val_im_1, val_im_2, val_ang, val_y = imgs[val_idx, :, :, :2], imgs[val_idx, :, :, 2:], angls[val_idx], lbls[val_idx]
    
    return [train_im_1, train_im_2, train_ang, train_y], [val_im_1, val_im_2, val_ang, val_y]
    
#def make_gen(x_data_img, x_data_angle, y_data, batch_size):
#    num_images = len(x_data_img)
#
#    if len(y_data) == 1:
#        y_data = y_data[0]

#    while True:
#        idx1 = np.random.randint(0, num_images, batch_size)
#        #idx2 = np.random.randint(0, num_images, batch_size)
#
#        batch_x = [x_data_img[idx1, 2, :, :, :],\
#                   x_data_angle[idx1]]
#        
#        batch_y = y_data[idx1]
#
#        yield batch_x, batch_y
        
def prepare_data_train(filename='/net/store/scratch/even/valid_until_28_feb_2018/ksokolov/statoil-kaggle/data/train.json'):
    train = pd.read_json(filename)
    train.inc_angle = train.inc_angle.replace('na', 0)
    train_X, angles = transform(train)
    train_y = np.array(train['is_iceberg'])
    train_X = augment(train_X)
    train_y = np.concatenate((train_y, train_y, train_y))
    angles = np.concatenate((angles, angles, angles))

    return train_X, angles, train_y
    
def prepare_data_test(filename='/net/store/scratch/even/valid_until_28_feb_2018/ksokolov/statoil-kaggle/data/test.json'):
    test = pd.read_json(filename)
    test.inc_angle = test.inc_angle.fillna(0)
    ids = test['id']
    test_X, angles = transform(test)

    return test_X[:, :, :, :2], test_X[:, :, :, 2:], angles, ids
    
    
