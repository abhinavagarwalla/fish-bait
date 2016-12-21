

"==========loading libraies========="

print "loading libraries"
import numpy as np
import cv2
import os
import scipy.io as io

"==========Constants=========="


FISH_CLASSES = ['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT']
CHANNELS =3
HEIGHT = 720
WIDTH = 1280
TRAIN_DIR = 'train/'
TEST_DIR = 'test/'
ROWS = WIDTH/10
COLS = HEIGHT/10


"======= Load Images ========="


def get_images(fish):
    fish_dir = TRAIN_DIR+'{}'.format(fish)
    images = [fish+'/'+ i for i in os.listdir(fish_dir)]
    return images

def read_image(src):
    im = cv2.imread(src, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (COLS, ROWS), interpolation=cv2.INTER_CUBIC)
    return im
	

files =[]
target =[]
print "loading images"
count = 0
for fish in FISH_CLASSES:
    fish_files = get_images(fish)
    for i in fish_files:
    	data = read_image(TRAIN_DIR+i)
    	data = np.asarray(data)
    	data = data.astype(float)
    	data -= np.min(data)
    	data /= np.max(data)
    	tar = count
    	files.append(data)
    	target.append(tar)

    print "done: "+ fish
    count = count+1


"============ save mat ============="

input_data={}
files = np.asarray(files)
files = files.astype(np.float32)

input_data['data'] = files
input_data['label'] = target
print "saving input.mat file",
f = io.savemat(("input.mat"),input_data)
print " done"



"======= load and save test ======="
test_files = [im for im in os.listdir(TEST_DIR)]
test = np.ndarray((len(test_files), ROWS, COLS, CHANNELS), dtype=np.float32)

for i, im in enumerate(test_files): 
    data = read_image(TEST_DIR+im)
    data = np.asarray(data)
    data = data.astype(float)
    data -= np.min(data)
    data /= np.max(data)
    test[i] = data

output_data={}
output_data['output_data'] = test
print "saving test.mat file",
f = io.savemat(("test.mat"),output_data)
print "  done"