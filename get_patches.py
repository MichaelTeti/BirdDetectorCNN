import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
import os
import h5py

ps = 100


def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata

    global coords
    coords.append((ix, iy))

    if ix > 2000 and iy > 1500:
        fig.canvas.mpl_disconnect(cid)
        plt.close(1)
    return


os.chdir('/home/voxelrx/Downloads/M1')
for filename in os.listdir(os.getcwd()):
    if filename.endswith('.JPG'):
        print(filename)
        im = np.float32(imread(filename))
        im = im[275:, :, :]
        patches = np.zeros([1, ps*2, ps*2, 3])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(im, cmap = 'gray')
        fig.set_size_inches(100, 100)

        coords = []
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        coords = np.asarray(coords)
        coords = np.int32(np.floor(coords[:-1, :]))

        for i in range(coords.shape[0]):
            c, r = coords[i, :]
            patch = im[r-ps:r+ps, c-ps:c+ps, :]
            print(patch.shape)
            patches = np.concatenate((patches[1:, :, :, :], patch[np.newaxis, :, :, :]), 0)

labels = np.ones([patches.shape[0], 1])


os.chdir('/home/voxelrx/Downloads/absence')
num_per_im = np.round(patches.shape[0]/len(os.listdir(os.getcwd())))

for fname in os.listdir(os.getcwd()):
    if fname.endswith('.JPG'):
        img = np.float32(imread(fname))
        img = img[275:, :, :]

        for i in range(num_per_im):
            randr = np.random.randint(ps, img.shape[0]-ps-1, 1)
            ranc = np.random.randint(ps, img.shape[1]-ps-1, 1)
            patch = img[randr-ps:randr+ps, randc-ps:randc+ps, :]
            patches = np.concatenate((patches, patch[np.newaxis, :, :, :]), 0)
            labels = np.concatenate((labels, np.zeros([1, 1])), 0)

f = h5py.File('lila_data.h5', 'a')
f.create_dataset('imgs', data=patches)
f.create_dataset('labels', data=labels)
f.close()
