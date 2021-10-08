import pandas as pd
import numpy as np


class seperate_ind_feat():

    """
    class name: seperate_ind_feat
    Description: This class shows the train test split. This has been done with a different approach as the test set
    doesn't have is_goal values
    Wriiten by: Harsh KM
    Version: 1.0
    Revision: 0
    """

    def __init__(self):
        self.data_src_train = "D:/ML Projects/CR_goal/Data/train.csv"
        self.data_src_test = "D:/ML Projects/CR_goal/Data/test.csv"
    def x_y_feat(self):

        """
         Method_Name : x_y_feat
         Description : Splitting the dataset into dependent and independent features.
         Output      : DataFrame
         On Failure  : Raises Exception
         Written by  :  Harsh KM
         Version     : 1.0
         Revision    : 0
        """
        try:
            self.train = pd.read_csv(self.data_src_train)
            self.test = pd.read_csv(self.data_src_test)
            self.x_train = pd.DataFrame(self.train.drop('is_goal',axis=1))
            self.y_train = pd.DataFrame(self.train.is_goal)
            self.x_test = pd.DataFrame(self.test.drop('is_goal',axis=1))
            return self.x_train, self.y_train, self.x_test
        except Exception as e:
            print(e)

s = seperate_ind_feat()
s.x_y_feat()


