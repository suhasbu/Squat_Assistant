import pickle
# filename = 'classifier.ipynb'
# pickle.dump(model_k, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
def final_classify():
    
    loaded_model = pickle.load(open('./data/model.sav', 'rb'))
    result = loaded_model.predict(X_train)
    return result



