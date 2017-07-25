import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imshow
import os
import h5py
import glob
import cv2


ps = 75
patches = []
photo_dir = '/home/voxelrx/birds/M1All/'


def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    global coords
    coords.append((ix, iy))
    if ix > 2000 and iy > 1500:
        fig.canvas.mpl_disconnect(cid)
        plt.close(1)
    return


os.chdir(photo_dir)
dirs = os.listdir(os.getcwd())
num_dirs = len(dirs)

for dir_ in dirs:
    os.chdir(photo_dir + dir_)
    counts = np.loadtxt('totals.csv', delimiter=',')
    bird_nums = counts[:, 1]
    id_num = counts[:, 0]
    files = glob.glob('*.JPG')
    assert(len(files) == bird_nums.size), 'please check data folder'
    files.sort(key=lambda f: int(filter(str.isdigit, f)))

    for j in xrange(1, len(bird_nums)-1):
        if bird_nums[j] == 0:
            continue
        filename = files[j]
        assert(int(filename[4:8]) == id_num[j]), 'files loading out of order'
        im = np.uint8(imread(filename))
        prev = np.uint8(imread(files[j-1]))
        nextim = np.uint8(imread(files[j+1]))
        im = im[615:, ...]
        prev = prev[615:, ...]
        nextim = nextim[615:, ...]
        print('There are %d birds in this picture'%(bird_nums[j]))
        raw_input('Press Enter to continue')

        for img_num in xrange(1000):
            cv2.namedWindow('Image', cv2.WINDOW_AUTOSIZE)
            if img_num & 1:
                print('current image')
                cv2.imshow('Image', im[..., ::-1])
            else:
                print('previous image')
                cv2.imshow('Image', prev[..., ::-1])
            if cv2.waitKey(650) > 0:
                break

        #cv2.destroyAllWindows()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(np.uint8(im))
        fig.set_size_inches(100, 100)

        coords = []
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        coords = np.asarray(coords)
        coords = np.int32(np.floor(coords))

        for i in xrange(coords.shape[0]):
            c, r = coords[i, :]
            if im.shape[0] - r <= ps or r <= ps:
                continue
            elif im.shape[1] - c <= ps or c <= ps:
                continue
            patch = im[r-ps:r+ps, c-ps:c+ps, :]
            patchprev = prev[r-ps:r+ps, c-ps:c+ps, :]
            patchnext = nextim[r-ps:r+ps, c-ps:c+ps, :]
            patchstack = np.concatenate((patchprev, patch, patchnext), 2)
            patches.append(patchstack[None, ...])
labels = np.ones([patches.shape[0], 1])

raw_input('''You have selected all of the birds in this file. To continue
and select non birds, press Enter''')
os.chdir(photo_dir)
subdirs = os.listdir(os.getcwd())
img_nums = np.zeros([len(subdirs)])
for i in range(len(subdirs)):
    os.chdir(photo_dir + subdirs[i])
    img_nums[i] = len(glob.glob('*.JPG'))

most_imgs = np.argmax(img_nums) + 1
os.chdir(photo_dir + str(most_imgs))
files = glob.glob('*.JPG')
num_imgs = labels.shape[0]
rand_ex = np.random.randint(1, len(files)-1, num_imgs)
for j in xrange(num_imgs):
    rand_num = rand_ex[j]
    im = np.uint8(imread(files[rand_num]))
    prev = np.uint8(imread(files[rand_num-1]))
    nextim = np.uint8(imread(files[rand_num+1]))
    im = im[615:, ...]
    prev = prev[615:, ...]
    nextim = nextim[615:, ...]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(np.uint8(im))
    fig.set_size_inches(100, 100)
    coords = []
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    coords = np.asarray(coords)
    coords = np.int32(np.floor(coords))

    for i in xrange(coords.shape[0]):
        c, r = coords[i, :]
        if im.shape[0] - r <= ps or r <= ps:
            continue
        elif im.shape[1] - c <= ps or c <= ps:
            continue
        patch = im[r-ps:r+ps, c-ps:c+ps, :]
        patchprev = prev[r-ps:r+ps, c-ps:c+ps, :]
        patchnext = nextim[r-ps:r+ps, c-ps:c+ps, :]
        patchstack = np.concatenate((patchprev, patch, patchnext), 2)
        patches.append(patchstack[None, ...])
        labels = np.concatenate((labels, np.zeros([1, 1])), 0)


f = h5py.File(photo_dir[19:-1] + '.h5', 'a')
f.create_dataset('imgs', data=np.asarray(patches))
f.create_dataset('labels', data=labels)
f.close()
