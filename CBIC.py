import numpy as np
import os
from tqdm import tqdm
import cv2
from sklearn.neighbors import KNeighborsClassifier

Test = "TestSet"
Train = "TrainingSet"
Validation = "ValidationSet"

CATEGORIES = ["airplanes", "bonsai", "chair", "ewer", "faces", "flamingo", "guitar", "leopards", "motorbikes",
              "starfish"]


#######################################################################################################################

class DataSet:

    def create_data(self, folder):

        data = []
        area = []
        perimeter = []
        for category in CATEGORIES:
            path = os.path.join(folder, category)  # create path
            class_num = CATEGORIES.index(category)  # get the classification
            for img in tqdm(os.listdir(path)):  # iterate over each image
                try:
                    img_array = cv2.imread(os.path.join(path, img))  # convert to array
                    new_array = cv2.resize(img_array, (64, 64))  # resize to normalize data size
                    data.append([new_array, class_num])  # add this to our training_data
                except Exception as e:
                    pass

        Images = []
        Labels = []

        for features, label in data:
            Images.append(features)
            Labels.append(label)

        Images = np.array(Images)

        Images = Images / 255.0
        Images = np.array(Images)
        Labels = np.array(Labels)

        Images = Images.reshape(-1, 12288)

        return Images, Labels


#######################################################################################################################
class KNearN:

    def knn(self, training_Images, training_Labels, validation_Images, validation_Labels, neighbours):
        knn = KNeighborsClassifier(n_neighbors=neighbours, p=2)
        KNN = knn.fit(training_Images, training_Labels)
        score = knn.score(validation_Images, validation_Labels)
        return KNN, score


#############################################################################################################

class CBIC:

    def start(self):
        training_data, training_labels = DataSet.create_data(self, Train)
        print(training_data.shape)
        testing_data, testing_labels = DataSet.create_data(self, Test)
        print(testing_data.shape)
        validation_data, validation_labels = DataSet.create_data(self, Validation)
        print(validation_data.shape)
        neighbours = 1
        for i in range(5):
            KNN, score = KNearN.knn(self, training_data, training_labels, validation_data, validation_labels,
                                    neighbours)
            print(score)
            neighbours = neighbours + 2


CBIC().start()
