import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imshow
import os
import h5py
import glob
import cv2


ps = 85
photo_dir = '/home/voxelrx/birds/M3All/'

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
    print(dir_)
    if os.path.isfile(photo_dir + dir_ + '/' + photo_dir[20:-1] + dir_ + '.h5'):
        continue

    patches = []
    labels = []
    os.chdir(photo_dir + dir_)
    counts = np.loadtxt('totals.csv', delimiter=',')
    bird_nums = counts[:, 1]
    id_num = counts[:, 0]
    files = glob.glob('*.JPG')
    assert(len(files) == bird_nums.size), 'please check data folder'
    files.sort(key=lambda f: int(filter(str.isdigit, f)))
    num_ones = 0

    for j in xrange(1, len(bird_nums)-1):
        if bird_nums[j] == 0:
            continue
        filename = files[j]
        assert(int(filename[4:8]) == id_num[j]), 'files loading out of order'
        im = np.uint8(imread(filename))
        prev = np.uint8(imread(files[j-1]))
        nextim = np.uint8(imread(files[j+1]))

        im = im[450:, ...]
        prev = prev[450:, ...]
        nextim = nextim[450:, ...]

        print('There are %d birds in this picture'%(bird_nums[j]))

        for img_num in xrange(1000):
            cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
            if img_num & 1:
                print('current image')
                cv2.imshow('Image', im[..., ::-1])
            else:
                print('previous image')
                cv2.imshow('Image', prev[..., ::-1])
            if cv2.waitKey(650) > 0:
                break

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
            labels.append(1.0)
            num_ones += 1
            print(np.asarray(patches).shape)

    num_zeros = np.sum(np.float32(bird_nums==0))
    zeros_per_im = np.int32(np.ceil(num_ones/num_zeros))
    num_zeros = 0
    print('Adding patches without birds')

    for j in xrange(1, len(bird_nums)-1):
        if bird_nums[j] != 0:
            continue
        elif num_zeros == num_ones:
            break

        filename = files[j]

        im = np.uint8(imread(filename))
        prev = np.uint8(imread(files[j-1]))
        nextim = np.uint8(imread(files[j+1]))

        im = im[450:, ...]
        prev = prev[450:, ...]
        nextim = nextim[450:, ...]

        randcols = np.int32(np.random.randint(ps, im.shape[1]-ps-1, zeros_per_im))
        randrows = np.int32(np.random.randint(ps, im.shape[0]-ps-1, zeros_per_im))

        for i in xrange(randrows.size):
            rr = randrows[i]
            rc = randcols[i]
            patch = im[rr-ps:rr+ps, rc-ps:rc+ps, :]
            patchprev = prev[rr-ps:rr+ps, rc-ps:rc+ps, :]
            patchnext = nextim[rr-ps:rr+ps, rc-ps:rc+ps, :]
            patchstack = np.concatenate((patchprev, patch, patchnext), 2)
            patches.append(patchstack[None, ...])
            labels.append(0.0)
            num_zeros += 1

    os.chdir(photo_dir)
    print('Saving file...')
    f = h5py.File(photo_dir[20:-1] + dir_ + '.h5', 'a')
    f.create_dataset('imgs', data=np.asarray(patches))
    f.create_dataset('labels', data=np.asarray(labels))
    f.close()
