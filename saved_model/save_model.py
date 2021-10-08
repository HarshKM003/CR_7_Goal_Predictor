import pickle
from model_building.model_formation import get_train_test_data
from Hyperparameter_tuning.h_tuning import p_tuning

a = get_train_test_data()
a.retrieve_data()
a.rfc_model()
#p = p_tuning

#p.model_access()
#p.set_parameters()
#p.model_tuning()

pickle.dump(a.rfc,open('model.pkl','wb'))
print("Success")