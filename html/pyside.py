'''
Installation of pip, tensorflow, keras:
run this in terminal to install pip:  curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
run this in terminal to install tensorflow: pip install --upgrade tensorflow
run this in terminal to install keras: pip install keras
'''
# dataProcessing
# Load Dataset
import pandas as pd
import numpy as np
import json


def genList(dataDictList, requiredParts, startIndex, stopIndex):
    '''
    @args
    dataDictList - json load
    requiredParts - ["nose", etc.]
    startIndex - relative timestamp to start. 0
    stopIndex - relative timestamp to stop. len(dataDictList)
    @returns
    [#, x], [#, y]
    '''
    dataListX = None
    dataListY = None
    for _frame in range(startIndex, stopIndex):
        if dataListX is None:
            dataListX = [[dataDictList['frame-'+str(_frame)]['keypoints'][PARTS[X]]['position'][0] for X in requiredParts]]
            dataListY = [[dataDictList['frame-'+str(_frame)]['keypoints'][PARTS[X]]['position'][1] for X in requiredParts]]
        else:
            dataListX = dataListX + [[dataDictList['frame-'+str(_frame)]['keypoints'][PARTS[_part]]['position'][0] for _part in requiredParts]]
            dataListY = dataListY + [[dataDictList['frame-'+str(_frame)]['keypoints'][PARTS[_part]]['position'][1] for _part in requiredParts]]

    dataListX = np.array(dataListX) # [[x]]
    dataListY = np.array(dataListY) # [[y]]
    newDataListX = np.reshape(dataListX, (-1,len(requiredParts)))
    newDataListY = np.reshape(dataListY, (-1,len(requiredParts))) #[#instance, (y) for each keypoint]
    print(str(len(newDataListX)) + " entries")
    return newDataListX, newDataListY

'''
 This has 2 parts-
 Resize, by identifying and scaling as bounding box, and then (Min-Max) Normalize
 Note: The broadcasting here is element-wise.
 Sources:
 [1] https://medium.com/tensorflow/move-mirror-an-ai-experiment-with-pose-estimation-in-the-browser-using-tensorflow-js-2f7b769f9b23
 [2] https://raw.githubusercontent.com/paulvollmer/posenet-keypoints-normalization/master/src/index.js


'''

def scale(newDataListX, newDataListY):
    '''
    Assumes dataList of form [#instance, [(x,y) for each keypoint]]
    Possible optim: bounding box wont change much across frames
    '''
    # Bounding Box
    maxX = np.max(newDataListX, axis=1)
    minX = np.min(newDataListX, axis=1)
    maxY = np.max(newDataListY, axis=1)
    minY = np.min(newDataListY, axis=1)
    assert(len(minY)==len(newDataListX))
    l2Data = np.concatenate((newDataListX, newDataListY), axis=1)
    # Reset to Origin and Scale
    for _data in range(0, len(newDataListX)):
        l2Data[_data] = l2Data[_data] / np.linalg.norm(l2Data[_data]) # L2 norm if the need be
    return l2Data[:,:newDataListX.shape[-1]], l2Data[:,newDataListY.shape[-1]:]

def findApex(xTbl, yTbl):
    minColumnIndex = yTbl['leftHip'].idxmin()
    print("Entry with apex = " + str(minColumnIndex))
    xTblApex = xTbl.iloc[minColumnIndex,:]
    yTblApex = yTbl.iloc[minColumnIndex,:]
    apex = []
    for i in range(len(xTblApex)):
        xTblVal = xTblApex[i]
        yTblVal = yTblApex[i]
        apex.append(xTblVal)
        apex.append(yTblVal)
    return apex

def findInputApex(path):
    print("Starting findInputApex")
    print("--------------------")
    # Creats index names for final table, each part and each coordinate having separate columns
    columnIndices = []
    for index in requiredParts:
        columnIndices.append(index + "_x")
        columnIndices.append(index + "_y")

    apexes = []
    with open(path, 'r') as f:
        dataDictList = json.load(f)
    newDataListX, newDataListY = genList(dataDictList, requiredParts, 0, len(dataDictList))
    scaledDataX, scaledDataY = scale(newDataListX, newDataListY)
    x_key=pd.DataFrame(scaledDataX)
    y_key=pd.DataFrame(scaledDataY)
    x_key.columns=requiredParts
    y_key.columns=requiredParts
    apex_point = findApex(x_key, y_key)
    apexes.append(apex_point)
    apex_df = pd.DataFrame(apexes, columns = columnIndices)
    apex_df.index.name = 'entry'
    return apex_df



PARTS =  {"nose" : 0,
  "leftEye" : 1,
  "rightEye" : 2,
  "leftEar" : 3,
  "rightEar" : 4,
  "leftShoulder" : 5,
  "rightShoulder" : 6,
  "leftElbow" : 7,
  "rightElbow" : 8,
  "leftWrist" : 9,
  "rightWrist" : 10,
  "leftHip" : 11,
  "rightHip" : 12,
  "leftKnee" : 13,
  "rightKnee" : 14,
  "leftAnkle" : 15,
  "rightAnkle" : 16}

requiredParts = ["nose",
#   "leftEye",
#   "rightEye",
#   "leftEar",
#   "rightEar",
  "leftShoulder",
  "rightShoulder",
#   "leftElbow",
#   "rightElbow",
#   "leftWrist",
#   "rightWrist",
  "leftHip",
  "rightHip",
  "leftKnee",
  "rightKnee",
  "leftAnkle",
  "rightAnkle"
]

IMAGE_X_SIZE = 600
IMAGE_Y_SIZE = 450
"""
def add_features():
	features = apex_df.copy()
	features['xcoord_lhip_ank'] = features['leftHip_x'] - features['leftAnkle_x']
	features['ycoord_lhip_knee'] = features['leftHip_y'] - features['leftKnee_y']
	features['left_hip_angle'] = np.arctan(features['ycoord_lhip_knee']/features['xcoord_lhip_ank'])
	features['xcoord_rhip_ank'] = features['rightHip_x'] - features['rightAnkle_x']
	features['ycoord_rhip_knee'] = features['rightHip_y'] - features['rightKnee_y']
	features['right_hip_angle'] = np.arctan(features['ycoord_rhip_knee']/features['xcoord_rhip_ank'])
	return features

"""
# classify.ipnyb
'''
Installation of pip, tensorflow, keras:
run this in terminal to install pip:  curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
run this in terminal to install tensorflow: pip install --upgrade tensorflow
run this in terminal to install keras: pip install keras
'''
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def classify(input):
    training = pd.read_csv('../data/apex.csv')
    input_copy = input.copy()
    #Feature engineering
    '''
    Important features to look at for 'good' squats:
    1) angle at hip
    2) difference in x-coordinates of hip and ankle
    3) difference in y-coordinates of hip and knee
    '''

    data = [training, input_copy]
    for tbl in data:
        tbl['xcoord_lhip_ank'] = tbl['leftHip_x'] - tbl['leftAnkle_x']
        tbl['ycoord_lhip_knee'] = tbl['leftHip_y'] - tbl['leftKnee_y']
        tbl['left_hip_angle'] = np.arctan(tbl['ycoord_lhip_knee']/tbl['xcoord_lhip_ank'])

        tbl['xcoord_rhip_ank'] = tbl['rightHip_x'] - tbl['rightAnkle_x']
        tbl['ycoord_rhip_knee'] = tbl['rightHip_y'] - tbl['rightKnee_y']
        tbl['right_hip_angle'] = np.arctan(tbl['ycoord_rhip_knee']/tbl['xcoord_rhip_ank'])

    y = training['good']
    X = training.drop(['good', 'entry'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    clf_log = LogisticRegression()
    clf_log = clf_log.fit(X_train,y_train)
    y_pred_log = clf_log.predict(input_copy)[0]

    # Evaluate the model Accuracy on test set
    return y_pred_log


def integrate():
    good = 1
    path = "../data/log.json"
    tbl = findInputApex(path)
    good = classify(tbl)
    if(good==1):
    	file1.write("Good Squat")
    if(good==0):
    	file1.write("Bad Squat")
    file1.close()
    return good
