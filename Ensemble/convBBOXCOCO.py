import pickle as pk
import numpy as np
import json

numVal = 5000
numTrain = 191961
COCO_id_to_category_id = {13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32, 37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40, 46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48, 54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56, 62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72, 82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}
class_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "trafficlight",
        "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
        "snowboard", "sportsball", "kite", "baseballbat", "baseballglove", "skateboard", "surfboard",
        "tennisracket", "bottle", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
        "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair", "couch", "pottedplant", "bed",
        "diningtable", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cellphone", "microwave", "oven",
        "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddybear", "hairdrier",
        "toothbrush"]

boxes  = pk.load(open('val_boxes_CC.pkl','rb'))
labels = pk.load(open('val_labels_CC.pkl','rb'))
scores = pk. load(open('val_scores_CC.pkl','rb'))
allids = pk. load(open('allids.pkl','rb'))

with open('cocovaltrue.json', 'r') as f:
    gt = json.load(f)

for i in range(1,numVal+1):
    fname = 'image_' + str(i)+'.txt'
    f = open('input/detection-results/'+fname, 'w')
    for p in range(boxes[i-1].shape[0]):
        aa = int(labels[i-1][p])
        if int(labels[i-1][p])+1 in COCO_id_to_category_id:
            uu = COCO_id_to_category_id[int(labels[i-1][p]) + 1] - 1
            f.write(class_names[uu] + " ")
        else:
            f.write(class_names[int(labels[i-1][p])] + " ")
        f.write(str(scores[i-1][p])+" ")
        f.write(str(boxes[i-1][p][0])+" "+str(boxes[i-1][p][1])+" "+str(boxes[i-1][p][2])+" "+str(boxes[i-1][p][3])+" ")
        f.write("\n")
    f.close()

allids.append(581781)
for i in range(1,numVal+1):
    print(i)
    tempID = allids[i-1]
    fname = 'image_' + str(i) + '.txt'
    gtBBOX = []
    gtCat = []

    for item in gt['annotations']:
        if tempID == item['image_id']:
            tempBBOX = item['bbox']
            tempBBOX[3] += tempBBOX[1]
            tempBBOX[2] += tempBBOX[0]
            gtBBOX.append(tempBBOX)
            gtCat.append(item['category_id'] - 1)

    f = open('input/ground-truth/' + fname, 'w')
    for p in range(len(gtBBOX)):
        if gtCat[p] + 1 in COCO_id_to_category_id:
            uu = COCO_id_to_category_id[gtCat[p] + 1] - 1
            f.write(class_names[uu] + " ")
        else:
            f.write(class_names[gtCat[p]] + " ")
        for k in gtBBOX[p]:
            f.write(str(k) + " ")
        f.write("\n")
    f.close()