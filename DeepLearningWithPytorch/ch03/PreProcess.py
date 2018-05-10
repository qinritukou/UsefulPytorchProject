from glob import glob
import os
import numpy as np

path = '/Users/developer/data/dogvscat'

# Read all the files inside our folder
files = glob(os.path.join(path, '*/*.jpg'))

print(f'Total no of images {len(files)}')

no_of_images = len(files)

# Create a shuffled index which can be used to create a validation data set
shuffle = np.random.permutation(no_of_images)

# Create a validation directory for holding validation images.
os.mkdir(os.path.join(path, 'valid'))

# Create directoryies with label names
for t in ['train', 'valid']:
    for folder in ['dog/', 'cat/']:
        os.mkdir(os.path.join(path, t, folder))


# Copy a small subset of images into the validation folder
for i in shuffle[:2000]:
    folder = files[i].split('/')[-1].split('.')[0]
    image = files[i].split('/')[-1]
    filepath = os.path.join(path, 'valid', folder, image)
    os.rename(files[i], filepath)

# Copy a small subset of images into the training folder.
for i in shuffle[2000:]:
    folder = files[i].split('/')[-1].split('.')[0]
    image = files[i].split('/')[-1]
    os.rename(files[i], os.path.join(path, 'train', folder, image))
