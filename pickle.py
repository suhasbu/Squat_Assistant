import pickle
filename = 'classifier.ipynb'
joblib.dump(model_k, filename)
 
# some time later...
 
# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.evaluate(X_test, y_test)
print(result)