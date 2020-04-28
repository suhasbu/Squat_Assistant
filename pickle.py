import pickle
from sklearn.externals import joblib 

# filename = 'classifier.ipynb'
# pickle.dump(model_k, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
def final_classify(X_train):
    
    loaded_model = joblib.load(open('./data/model.pkl'))
    result = loaded_model.predict(X_train)
    return result



