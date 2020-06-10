from ensemble_boxes import *
import pickle as pk
import numpy as np

numVal = 5000
numTrain = 191961

boxesCSC50  = pk.load(open('centcocobox.pkl','rb'))
labelsCSC50 = pk.load(open('centcocolab.pkl','rb'))
scoresCSC50 =pk.load(open('centcocosco.pkl','rb'))
idsc = pk. load(open('centcocoid.pkl','rb'))

boxesCSC101  = pk.load(open('fastcocobox.pkl','rb'))
labelsCSC101 = pk.load(open('fastcocolab.pkl','rb'))
scoresCSC101 =pk.load(open('fastcocosco.pkl','rb'))
idsf = pk. load(open('fastcocoid.pkl','rb'))

for i in range(len(boxesCSC50)):
    boxesCSC50[i] = np.ndarray.tolist(boxesCSC50[i])
    labelsCSC50[i] = np.ndarray.tolist(labelsCSC50[i])
    scoresCSC50[i] = np.ndarray.tolist(scoresCSC50[i])

for i in range(len(boxesCSC101)):
    boxesCSC101[i] = np.ndarray.tolist(boxesCSC101[i])
    labelsCSC101[i] = np.ndarray.tolist(labelsCSC101[i])
    scoresCSC101[i] = np.ndarray.tolist(scoresCSC101[i])


weights = [1, 1]
weights2 = [5, 1]
threshiou = 0.55
threshlow = 0.025

boxesYC = []
scoresYC = []
labelsYC = []

boxesCC = []
scoresCC = []
labelsCC = []

# for i in range(numVal):
#     print(i)
#     boxes1, scores1, labels1 = weighted_boxes_fusion([boxesCSC101[i], boxesYOLO[i]], [scoresCSC101[i], scoresYOLO[i]],
#                                                      [labelsCSC101[i], labelsYOLO[i]], weights=weights, iou_thr=threshiou,
#                                                      skip_box_thr=threshlow)
#     # boxes2, scores2, labels2 = getBBOX([boxesCSC101[i], boxesYOLO[i]], [scoresCSC101[i], scoresYOLO[i]],
#     #                                                  [labelsCSC101[i], labelsYOLO[i]], w=weights, threshiou=threshiou,
#     #                                                  threshlow=threshlow)
#
#     boxesYC.append(boxes1)
#     scoresYC.append(scores1)
#     labelsYC.append(labels1)
#     # boxesYC.append(boxes1)
#     # scoresYC.append(scores1)
#     # labelsYC.append(labels1)

for i in range(numVal):
    print(i)
    boxes2, scores2, labels2 = weighted_boxes_fusion([boxesCSC101[i], boxesCSC50[i]], [scoresCSC101[i], scoresCSC50[i]],
                                                     [labelsCSC101[i], labelsCSC50[i]], weights=weights,
                                                     iou_thr=threshiou,
                                                     skip_box_thr=threshlow)
    # boxes2, scores2, labels2 = getBBOX([boxesCSC101[i], boxesYOLO[i]], [boxesCSC50[i], scoresCSC50[i]],
    #                                                  [labelsCSC101[i], labelsCSC50[i]], w=weights, threshiou=threshiou,
    #                                                  threshlow=threshlow)
    # boxesCC.append(boxes2)
    # scoresCC.append(scores2)
    # labelsCC.append(labels2)
    boxesCC.append(boxes2)
    scoresCC.append(scores2)
    labelsCC.append(labels2)

# with open('val_boxes_YC.pkl', 'wb') as f:
#     pk.dump(boxesYC, f)
# with open('val_labels_YC.pkl', 'wb') as f:
#     pk.dump(labelsYC, f)
# with open('val_scores_YC.pkl', 'wb') as f:
#     pk.dump(scoresYC, f)

with open('val_boxes_CC.pkl', 'wb') as f:
    pk.dump(boxesCC, f)
with open('val_labels_CC.pkl', 'wb') as f:
    pk.dump(labelsCC, f)
with open('val_scores_CC.pkl', 'wb') as f:
    pk.dump(scoresCC, f)

deneme = 0

