from itertools import accumulate
import cv2
from glob import glob
import numpy as np
from os.path import basename, dirname

def get_variations(image):
    return (
        image,
        cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(image, cv2.ROTATE_180),
        cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
        cv2.flip(image, 0),
        cv2.rotate(cv2.flip(image, 0), cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(cv2.flip(image, 0), cv2.ROTATE_180),
        cv2.rotate(cv2.flip(image, 0), cv2.ROTATE_90_COUNTERCLOCKWISE),
    )

def files_list(root):
    return glob(root + '/**/*.jpg', recursive=True)

def one_hot_encode(category, categories):
    category_index = categories.index(category)
    y_vector = np.zeros(8)
    y_vector[category_index] = 1
    return y_vector

def augment(files):
    for filepath in files:
        label = basename(dirname(filepath))
        image = np.float32(cv2.cvtColor(cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2GRAY)) / 255
        for v in get_variations(image):
            yield (label, v)

def addlabel(files):
    for filepath in files:
        label = basename(dirname(filepath))
        image = np.float32(cv2.cvtColor(cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2GRAY)) / 255
        yield (label, image)

def randomize(list, seed):
    generator = np.random.default_rng(seed)
    length = len(list)
    permutation = generator.permutation(length)
    for i in permutation:
        yield list[i]

def count_each_shape(files, categories):
    names = list(map(lambda f: basename(dirname(f)), files))
    return list(map(lambda category: (category, names.count(category)), categories))

def split_dataset(data_root, categories, sets_proportions):
    files = files_list(data_root)
    factor = len(files) / sum(sets_proportions)
    sets_sizes = list(map(lambda p : round(p * factor), sets_proportions))
    sets = []
    for s in sets_sizes:
        sets.append((
            [], # X
            [] # y
        ))
    print(count_each_shape(files, categories))
    # access files out of order to prevent having all the same shapes in the test set
    for (i, (label, image)) in enumerate(randomize(list(addlabel(files)), 305)):
        # fill all sets up to their calculated size
        set = sets[list(map(lambda tot_size : i < tot_size, accumulate(sets_sizes))).index(True)]
        
        y_vector = one_hot_encode(label, categories)
        set[0].append(image.reshape(-1)) # X
        set[1].append(y_vector) # y
    return map(lambda s : (np.array(s[0]), np.array(s[1])), sets)