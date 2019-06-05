import numpy as np
import scipy.misc
from random import shuffle
from math import ceil
import os
import h5py as hdf


def load_images_from_folder(folder):
    images = []

    for filename in os.listdir(folder):

        # img= imread(os.path.join(folder,filename),mode='RGB')
        img = get_img(os.path.join(folder, filename), (256, 256, 3))
        if img is not None:
            images.append(img)
    return np.array(images, dtype='float32')


def get_img(src, img_size=False):
    img = scipy.misc.imread(src, mode='RGB')  # misc.imresize(, (256, 256, 3))
    if not (len(img.shape) == 3 and img.shape[2] == 3):
        img = np.dstack((img, img, img))
    if img_size != False:
        img = scipy.misc.imresize(img, img_size)
    return img


def create_dataset(save_path, data_path, resize_img=True, shuffle_data=True):
    hdf5_file = hdf.File(os.path.join(save_path, 'dataset.hdf5'))

    # read addresses a from the 'train' folder
    addrs = glob.glob(data_path)

    # to shuffle data
    if shuffle_data:
        c = list(addrs)
        shuffle(c)
        addrs = c

    train_addrs = addrs
    train_shape = 0
    if resize_img == True:
        train_shape = (len(train_addrs), 256, 256, 3)
    else:
        train_shape = (len(train_addrs), 256, 256, 3)
    # print(train_shape.shape)
    hdf5_file.create_dataset("train_img", train_shape, np.float32)

    for i in range(len(train_addrs)):
        addr = train_addrs[i]
        img = get_img(addr, (256, 256, 3))
        # plt.imshow(img)
        # plt.show()
        hdf5_file["train_img"][i, ...] = img[None]

    print("Datsset created successfully")

    hdf5_file.close()


def show_images(path):
    hdf5_file = hdf.File(path, "r")
    # subtract the training mean

    # Total number of samples
    data_num = hdf5_file["train_img"].shape[0]

    batch_size = 1
    batches_list = list(range(int(ceil(float(data_num) / batch_size))))
    shuffle(batches_list)
    # loop over batches
    for n, i in enumerate(batches_list):
        i_s = i * batch_size  # index of the first image in this batch
        i_e = min([(i + 1) * batch_size, data_num])  # index of the last image in this batch

        images = hdf5_file["train_img"][i_s:i_e, ...]
        print(images[0].shape)

        print("%d / %d" % (n + 1, len(batches_list)))

        plt.imshow(images[0] / 255)
        plt.show()
        if n == 5:  # break after 5 batches
            break
    hdf5_file.close()

#
# if __name__== "__main__":
#     #create_dataset(save_path="C:/Users/Junaid/Desktop/first_github/data",data_path="C:/Users/Junaid/Desktop/first_github/*.jpg",resize_img=True,shuffle_data=True)
#     print("df")
#     show()
#     #show_images('C:/Users/Junaid/Desktop/first_github/data/dataset.hdf5')
