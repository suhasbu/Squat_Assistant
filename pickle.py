import pickle
filename = 'classifier.ipynb'
pickle.dump(model_k, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.evaluate(X_train, y_train)
result