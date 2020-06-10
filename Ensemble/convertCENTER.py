import json
import pickle as pk
import numpy as np
from ensemble_boxes import *

#     tempDict = next(item for item in out if item['image_id'] == 1 and item['score']>confidence)
numVal = 548
numTrain = 191961
confidence = 0.001
with open('results.json', 'r') as f:
    out = json.load(f)

# boxesYOLO  = pk.load(open('val_boxes.pkl','rb'))
# labelsYOLO = pk.load(open('val_labels.pkl','rb'))
# scoresYOLO =pk. load(open('val_scores.pkl','rb'))

boxesCSC = []
labelsCSC = []
scoresCSC = []
index = 0
for i in range(548):
    tempID = i + 1
    check = True
    tempBoxes = []
    tempLabels = []
    tempScores = []
    while(check and index<len(out)):
        item = out[index]
        if item['image_id'] == tempID:
            print('Sample ' + str(tempID))
            tempBBOX = item['bbox']
            tempBBOX[2] += tempBBOX[0]
            tempBBOX[3] += tempBBOX[1]
            tempBoxes.append(tempBBOX)
            tempScores.append(item['score'])
            tempLabels.append(item['category_id'] - 1)
            index +=1
        else:
            check = False

    boxesCSC.append(np.asarray(tempBoxes))
    labelsCSC.append(np.asarray(tempLabels))
    scoresCSC.append(np.asarray(tempScores))

with open('val_boxes_center.pkl', 'wb') as f:
    pk.dump(boxesCSC, f)
with open('val_labels_center.pkl', 'wb') as f:
    pk.dump(labelsCSC, f)
with open('val_scores_center.pkl', 'wb') as f:
    pk.dump(scoresCSC, f)

a = 5