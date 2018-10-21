import os, random, cv2
import numpy as np

def one_hot(class_numbers, num_classes=None):
    if num_classes is None:
        num_classes = np.max(class_numbers) + 1
    return np.eye(num_classes, dtype=float)[class_numbers]

class DataSet:
    def __init__(self, img_size, train, data_dir='.', validation=True):
        self.class_names = []
        self.filenames = []
        self.class_names = []
        self.num_classes = 0

        self.validation_filenames = []

        self.img_size = img_size
        self.img_shape = (img_size, img_size)
        self.img_size_flat = img_size ** 2

        for folder in next(os.walk(data_dir))[1]:
            if '.' == folder[0]:
                continue
            
            if 'validation' in folder:
                for file in os.listdir(data_dir + folder):
                    self.validation_filenames.append(data_dir + folder + '/' + file)
                continue
            
            self.class_names.append(folder)
            self.num_classes += 1

            for file in os.listdir(data_dir + folder):
                self.filenames.append(data_dir + folder + '/' + file)

        self.class_names.sort()
        random.shuffle(self.filenames)

        self._num_data = len(self.filenames)
        self.num_train = int(self._num_data * train)
        self.num_test = self._num_data - self.num_train

        self.train_filenames = self.filenames[:self.num_train]
        self.test_filenames = self.filenames[self.num_train:]

        if validation:
            self.x_val = []
            self.x_val_flat = []
            self.y_val = []
            self.y_val_cls = []

            for file in self.validation_filenames:
                tmp_cls = file.split('/')[-1][0].upper()
                self.y_val_cls.append(self.class_names.index(tmp_cls))
                self.y_val.append(one_hot(self.class_names.index(tmp_cls), self.num_classes))
                img = cv2.cvtColor(cv2.resize(cv2.imread(file), self.img_shape), cv2.COLOR_BGR2GRAY)
                self.x_val.append(img)
                self.x_val_flat.append(img.flatten())

            self.x_val_flat = np.vstack(self.x_val_flat)
            self.y_val = np.vstack(self.y_val)
            self.y_val_cls = np.array(self.y_val_cls)

        self.x_train = []
        self.x_train_flat = []
        self.y_train = []
        self.y_train_cls = []

        for file in self.train_filenames:
            tmp_cls = file.split('/')[-2]
            self.y_train_cls.append(self.class_names.index(tmp_cls))
            self.y_train.append(one_hot(self.class_names.index(tmp_cls), self.num_classes))
            img = cv2.cvtColor(cv2.resize(cv2.imread(file), self.img_shape), cv2.COLOR_BGR2GRAY)
            self.x_train.append(img)
            self.x_train_flat.append(img.flatten())

        self.x_train_flat = np.vstack(self.x_train_flat)
        self.y_train = np.vstack(self.y_train)
        #print(self.y_train_cls)
        self.y_train_cls = np.array(self.y_train_cls)

        
        self.x_test = []
        self.x_test_flat = []
        self.y_test = []
        self.y_test_cls = []

        for file in self.test_filenames:
            tmp_cls = file.split('/')[-2]
            self.y_test_cls.append(self.class_names.index(tmp_cls))
            self.y_test.append(one_hot(self.class_names.index(tmp_cls), self.num_classes))
            img = cv2.cvtColor(cv2.resize(cv2.imread(file), self.img_shape), cv2.COLOR_BGR2GRAY)
            self.x_test.append(img)
            self.x_test_flat.append(img.flatten())
            

        self.x_test_flat = np.vstack(self.x_test_flat)
        self.y_test = np.vstack(self.y_test)
        self.y_test_cls = np.array(self.y_test_cls)

    
    def random_batch(self, batch_size=32):
        idx = np.random.randint(low=0, high=self.num_train, size=batch_size)

        x_batch = self.x_train_flat[idx]
        y_batch = self.y_train[idx]
        y_batch_cls = self.y_train_cls[idx]

        return x_batch, y_batch, y_batch_cls