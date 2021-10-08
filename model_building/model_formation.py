from sklearn.ensemble  import RandomForestClassifier
from sklearn import metrics
from Data_splitting.train_test_split import seperate_ind_feat
import numpy as np

class get_train_test_data:
    """
        Class Name : get_train_test_data
        Description: This class is used to access training and testing data and then using this data to train the Machine Learning Model
        Written by : Harsh KM
        Version    : 1.0
        Revisions  : 0
    """

    def __init__(self):
        pass

    def retrieve_data(self):
        """
                Method_Name : retrieve_access
                Description : This method is used to access the training and testing data
                Output      : Dataframe
                On_Failure  : Raise Exception
                Written By  : Harsh KM
                Version     : 1.0
                Revisions   : 0
        """

        try:
            self.x_train,self.y_train,self.x_test = seperate_ind_feat().x_y_feat()
        except Exception as e:
            return e

    def rfc_model(self):
        """
               Method_Name : rfc_model
               Description : Using Random Forest Algorithm to build a model based on training data
               Output      : Model Accuracy
               On_Failure  : Raise Exception
               Written By  : Harsh KM
               Version     : 1.0
               Revisions   : 0
               """

        try:

            self.rfc = RandomForestClassifier()
            self.rfc.fit(self.x_train,self.y_train.values.ravel())
            self.y_predict = self.rfc.predict(self.x_test)
            print(self.y_predict)
        except Exception as e:
            return e

a = get_train_test_data()
a.retrieve_data()
a.rfc_model()