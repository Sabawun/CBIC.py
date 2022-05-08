import numpy as np
import os
from tqdm import tqdm
import cv2
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

Test = "TestSet"
Train = "TrainingSet"
Validation = "ValidationSet"

CATEGORIES = ["airplanes", "bonsai", "chair", "ewer", "faces", "flamingo", "guitar", "leopards", "motorbikes",
              "starfish"]


class DataSet:

    def create_data(self, folder):  # data set created, image pixel values are used as features.
        # Images kept in rgb, resized to 64x64, than later resized to 64*64*3 This code has been tried with using
        # edges as features though the results were extremely awful, further feature extraction techniques were also
        # tried/looked at, though pixels as features worked relatively better than others. The reason for the
        # accuracy numbers being between 51-58 is due to the limitation of KNN.
        data = []
        for category in CATEGORIES:
            path = os.path.join(folder, category)  # create path
            class_num = CATEGORIES.index(category)  # get the classification
            for img in tqdm(os.listdir(path)):  # iterate over each image
                try:
                    img_array = cv2.imread(os.path.join(path, img))  # convert to array
                    new_array = cv2.resize(img_array, (64, 64))  # resize to normalize data size (can be tried
                    # with 120x120 as well)
                    data.append([new_array, class_num])  # add this to our training_data
                except Exception as e:
                    pass

        Images = []
        Labels = []

        for features, label in data:
            Images.append(features)
            Labels.append(label)

        Images = np.array(Images)

        Images = Images / 255.0  # Normalizing values between 0-1
        Images = np.array(Images)
        Labels = np.array(Labels)

        Images = Images.reshape(-1, 12288)  # change this to 43200 if 120x120

        return Images, Labels


class KNearN:
    # The function returns a KNN model as well as the accuracy score against the validation set
    def knn(self, training_Images, training_Labels, validation_Images, validation_Labels, neighbours):
        knn = KNeighborsClassifier(n_neighbors=neighbours, p=2)
        KNN = knn.fit(training_Images, training_Labels)
        score = knn.score(validation_Images, validation_Labels)
        return KNN, score


class CBIC:

    def start(self):
        training_data, training_labels = DataSet.create_data(self, Train)  # Creating Training Set
        # print(training_data.shape) # size is 787x12288
        testing_data, testing_labels = DataSet.create_data(self,
                                                           Test)  # Creating Testing Set (though this is not required)
        # print(testing_data.shape) # size is 105x12288
        validation_data, validation_labels = DataSet.create_data(self, Validation)  # Creating Validation Set
        # print(validation_data.shape) # size is 105x12288
        plot_name = "training against validation"
        print("Score for K with Validation Set")
        CBIC.plot(self, training_data, training_labels, validation_data, validation_labels, plot_name)
        plot_name = "training against testing"
        print("")
        print("Score for K with Testing Set")  # Even though this is not needed since this set should be only used
        # for predicting user query
        CBIC.plot(self, training_data, training_labels, testing_data, testing_labels, plot_name)
        print("")
        CBIC.best(self, training_data, training_labels, validation_data, validation_labels)  # calling function with the
        # best K value to do prediction

    def plot(self, X_train, y_train, X_test, y_test, plot_name):
        y_points = []
        x_points = []
        neighbours = 1
        for i in range(5):  # K values are 1,3,5,7,9
            KNN, score = KNearN.knn(self, X_train, y_train, X_test, y_test, neighbours)
            print("Score for K=" + str(neighbours) + " is : " + str(score))
            y_points.append(score)
            x_points.append(neighbours)
            neighbours = neighbours + 2
        # plotting accuracy against k values
        y_points = np.array(y_points)
        x_points = np.array(x_points)

        plt.plot(x_points, y_points)
        plt.title(plot_name)
        plt.xlabel('K Value')
        plt.ylabel('Accuracy %')
        plt.show()

    def best(self, training_data, training_labels, validation_data, validation_labels):
        N = 3  # This gave the best result (Maybe a better way to do this automatically,
        # but outside scope of this assignment)
        KNN, score = KNearN.knn(self, training_data, training_labels, validation_data, validation_labels,
                                N)
        print("Best Score on both validation & testing is for K=" + str(N) + " which is : " + str(score) +
              " on the validation set")

        CBIC.pred(self, KNN)  # for user query

    def pred(self, KNN):
        print("")
        print("Please copy the (absolute) path of the testing image you would like to run the prediction on or press "
              "q to quit: ")
        path = input()
        while path != "q":
            test_image = cv2.imread(path)
            resize_array = cv2.resize(test_image, (64, 64))  # can be 120x120 if training set is 120x120
            resize_array = resize_array.reshape(-1, 12288)  # should be 43200 if image size is 120x120
            resize_array = resize_array / 255.0
            print(CATEGORIES[int(KNN.predict(resize_array))])  # prints the name of the predicted category
            print(
                "Please copy the (absolute) path of the testing image you would like to run the prediction on or "
                "press q to quit: ")
            path = input()


CBIC().start()  # Code Starts From Here

# SABAWUN AFZAL KHATTAK 2328284
