import numpy as np
import os
from PIL import Image

class dataset():
    def __init__(self, root) :
        self.root = root
        self.data = np.array([], dtype=np.int32)
        self.target = np.array([], dtype=np.int32)
        self.target_names = np.array([], dtype=np.int32)
        self.feature_names = np.array([], dtype=np.int32)

        nb_class = 0
        nb_images = 0


        for i in os.listdir(self.root) :
            for j in os.listdir(self.root + str(i)) :
                self.target_names = np.append(self.target_names,j)
                nb_class += 1
                for k in os.listdir(self.root + str(i) + '/' + str(j)) :
                    self.target = np.append(self.target,nb_class)
                    nb_images += 1
                    img = Image.open(self.root + str(i) + '/' + str(j) + '/' + str(k)).convert('L')
                    self.data = np.append(self.data,np.asarray(img))
        self.data = np.reshape(self.data,(nb_images,1600))

        for i in range(40) :
            for j in range(40) :
                self.feature_names = np.append(self.feature_names,'pixels ' + str(i) + '_' + str(j))
        



