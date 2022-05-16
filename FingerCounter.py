import cv2
import  time
import os
import HandTrackingModule as HTM

wcam = 640
hcam = 480
cap = cv2.VideoCapture(0)

cap.set(3,wcam)
cap.set(4,hcam)

FileFolderPath = "FingerImage"
myList = os.listdir(FileFolderPath)
#print(myList)
imageList = []
for imagPath in myList:
    image = cv2.imread(f'{FileFolderPath}/{imagPath}')
    #print(f'{FileFolderPath}/{imagPath}')
    imageList.append(image)
print(len(imageList))
pTime=0

detector = HTM.handDetector(detectionCon=0.7)
tipids=[4,8,12,16,20]
while True:
    success , img =cap.read()
    img= detector.findHands(img)
    lmsList = detector.findPosition(img,draw=False)
    #print(lmsList)
    if len(lmsList) !=0:
        fingers=[]
        #Thumb
        if lmsList[tipids[0]][1] > lmsList[tipids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 finger
        for id in range(1,5):
            if lmsList[tipids[id]][2]<lmsList[tipids[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalfingers= fingers.count(1)
        print(totalfingers)

        hightimage ,widthimage , chaneimaage = imageList[totalfingers-1].shape

        img[0:hightimage,0:widthimage] = imageList[totalfingers-1]

        cv2.rectangle(img,(20,255),(170,425),(0,255,0),cv2.FILLED)
        cv2.putText(img,str(totalfingers),(45,375),cv2.FONT_HERSHEY_PLAIN,10,(255,0,0),3)
    cTime =time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img,f'Fps : {int(fps)}',(400,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

    cv2.imshow("Image",img)

    if cv2.waitKey(1) &0xff ==ord('q'):
        break