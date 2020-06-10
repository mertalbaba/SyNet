import json
import pickle as pk
import numpy as np
from ensemble_boxes import *

#     tempDict = next(item for item in out if item['image_id'] == 1 and item['score']>confidence)
numVal = 5000
numTrain = 191961
confidence = 0.001
with open('COCOVAL.json', 'r') as f:
    out = json.load(f)
with open('centerval.json', 'r') as f:
    out2 = json.load(f)

# boxesYOLO  = pk.load(open('val_boxes.pkl','rb'))
# labelsYOLO = pk.load(open('val_labels.pkl','rb'))
# scoresYOLO =pk. load(open('val_scores.pkl','rb'))

boxesfast = []
labelsfast = []
scoresfast = []
idsfast = []
boxescent = []
labelscent = []
scorescent = []
idscent = []
index = 0
index2 = 0
tempID = out[index]['image_id']
temptemp = 0
allids = []
for i in range(5000):
    check = True
    tempBoxes = []
    tempLabels = []
    tempScores = []
    tempIds = []
    tempBoxes2 = []
    tempLabels2 = []
    tempScores2 = []
    tempIds2 = []
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
            tempIds.append(item['image_id'])
            temptemp = tempID
            index +=1
        else:
            allids.append(tempID)
            tempID = out[index]['image_id']
            check = False

    boxesfast.append(np.asarray(tempBoxes))
    labelsfast.append(np.asarray(tempLabels))
    scoresfast.append(np.asarray(tempScores))
    idsfast.append(np.asarray(tempIds))

    # tmp = 0
    # check = True
    # while (check and tmp < len(out)):
    #     item = out2[tmp]
    #     if item['image_id'] == temptemp:
    #         check = False
    #     else:
    #         tmp +=1
    # check = True
    # while (check and tmp < len(out)):
    #     item = out2[tmp]
    #     if item['image_id'] == temptemp:
    #         tempBBOX = item['bbox']
    #         tempBBOX[2] += tempBBOX[0]
    #         tempBBOX[3] += tempBBOX[1]
    #         tempBoxes2.append(tempBBOX)
    #         tempScores2.append(item['score'])
    #         tempLabels2.append(item['category_id'] - 1)
    #         tempIds2.append(item['image_id'])
    #         tmp += 1
    #     else:
    #         check = False
    #
    # boxescent.append(np.asarray(tempBoxes2))
    # labelscent.append(np.asarray(tempLabels2))
    # scorescent.append(np.asarray(tempScores2))
    # idscent.append(np.asarray(tempIds2))

with open('fastcocobox.pkl', 'wb') as f:
    pk.dump(boxesfast, f)
with open('fastcocolab.pkl', 'wb') as f:
    pk.dump(labelsfast, f)
with open('fastcocosco.pkl', 'wb') as f:
    pk.dump(scoresfast, f)
with open('fastcocoid.pkl', 'wb') as f:
    pk.dump(idsfast, f)

# with open('centcocobox.pkl', 'wb') as f:
#     pk.dump(boxescent, f)
# with open('centcocolab.pkl', 'wb') as f:
#     pk.dump(labelscent, f)
# with open('centcocosco.pkl', 'wb') as f:
#     pk.dump(scorescent, f)
# with open('centcocoid.pkl', 'wb') as f:
#     pk.dump(idscent, f)

with open('allids.pkl', 'wb') as f:
    pk.dump(allids, f)

a = 5