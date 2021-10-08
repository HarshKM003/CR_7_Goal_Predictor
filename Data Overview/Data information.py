import pandas as pd
import numpy as np

class get_data:

    """This class is for getting the data from source.
    Written by: Harsh K M
    Version: 1.0
    Revisions: 0
    """

    def __init__(self):
        self.data_src = "D:/ML Projects/CR_goal/Data/yds_data.csv"
        self.data_src_train = "D:/ML Projects/CR_goal/Data/train.csv"
        self.data_src_test = "D:/ML Projects/CR_goal/Data/test.csv"
        self.delimiter = ","

    def acquire_data_from_src(self):
        """
                Method_name: acquire_data_from_src
                Description: It is used to get data from its source
                Output: Pandas Dataframe
                On_failure: Raises Exception
                Written by: Harsh KM
                Version: 1.0
                Revisions: 0

                """

        try:
           self.df = pd.read_csv(self.data_src)
           self.df_train = pd.read_csv(self.data_src)
           self.df_test = pd.read_csv(self.data_src)
           return(self.df)
        except Exception as e:
           print(e)

g = get_data()
print(g.acquire_data_from_src())



