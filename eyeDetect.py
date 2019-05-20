import cv2
import numpy as np
import ClassyVirtualReferencePoint as ClassyVirtualReferencePoint
import ransac
from win32api import GetSystemMetrics

doTraining = True



verbose=True


showMainImg=True;


BLOWUP_FACTOR = 1 # Resizes image before doing the algorithm. Changing to 2 makes things really slow. So nevermind on this.
RELEVANT_DIST_FOR_CORNER_GRADIENTS = 8*BLOWUP_FACTOR
dilationWidth = 1+2*BLOWUP_FACTOR #must be an odd number
dilationHeight = 1+2*BLOWUP_FACTOR #must be an odd number
dilationKernel = np.ones((dilationHeight,dilationWidth),'uint8')





writeEyeDebugImages = True #enable to export image files showing pupil center probability
eyeCounter = 0



# init the filters we'll use below
haarFaceCascade = cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_alt.xml")
haarEyeCascade = cv2.CascadeClassifier("./haarcascades/haarcascade_eye.xml")
#img.listHaarFeatures() displays these Haar options:
#['eye.xml', 'face.xml', 'face2.xml', 'face3.xml', 'face4.xml', 'fullbody.xml', 'glasses.xml', 'lefteye.xml', #'left_ear.xml', 'left_eye2.xml', 'lower_body.xml', 'mouth.xml', 'nose.xml', 'profile.xml',
#'right_ear.xml', 'right_eye.xml', 'right_eye2.xml', 'two_eyes_big.xml', 'two_eyes_small.xml', 'upper_body.xml', #'upper_body2.xml']
OffsetRunningAvg = None
PupilSpacingRunningAvg = None

# global stuff for Adam's virtual ref point
#initialize the SURF descriptor
hessianThreshold = 500
nOctaves = 4
nOctaveLayers = 2
extended = True
upright = True
detector = cv2.xfeatures2d.SURF_create(hessianThreshold, nOctaves, nOctaveLayers, extended, upright)
#figure out a way to nearest neighbor map to index
virtualpoint = None
warm=0


def get_screen_sizes():    
    GetSystemMetrics(0)
    GetSystemMetrics(1)

#*********  getOffset  **********
def getOffset2(frame):
    face_cascade = cv2.CascadeClassifier('C:/Users/70187044/AppData/Local/Continuum/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml')
    #Obtain the size if the screen
    cx = 0
    cy = 0
    get_screen_sizes()
    numerator=0
    denominator=0
    roi=frame
    frame=cv2.flip(frame,1)
    gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),1)
        #----------- vertical mid line ------------------#
        #cv2.line(frame,(x+w/2,y),(x+w/2,y+h/2),(255,0,0),1)
        # ---------- horizontal lower line ----------------# 
        cv2.line(frame,(int(x+w/4.2),int(y+h/2.2)),(int(x+w/2.5),int(y+h/2.2)),(0,255,0),1)
        #----------- horizontal upper line ------#
        cv2.line(frame,(int(x+w/4.2),int(y+h/3)),(int(x+w/2.5),int(y+h/3)),(0,255,0),1)
        # ---------- vertical left line ----------#
        cv2.line(frame,(int(x+w/4.2),int(y+h/3)),(int(x+w/4.2),int(y+h/2.2)),(0,255,0),1)
        # ---------- vertical right line---------------#
        cv2.line(frame,(int(x+w/2.5),int(y+h/3)),(int(x+w/2.5),int(y+h/2.2)),(0,255,0),1)
        
        #------------ estimation of distance of the human from camera--------------#
        d=10920.0/float(w)
        #-------- coordinates of interest --------------# 
        x1=int(x+w/4.2)+1 		#-- +1 is done to hide the green color
        x2=int(x+w/2.5)
        y1=int(y+h/3)+1
        y2=int(y+h/2.2)
        roi=frame[y1:y2,x1:x2]
        gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        equ = cv2.equalizeHist(gray)
        thres=cv2.inRange(equ,0,20)
        kernel = np.ones((3,3),np.uint8)
        #/------- removing small noise inside the white image ---------/#
        dilation = cv2.dilate(thres,kernel,iterations = 2)
        #/------- decreasing the size of the white region -------------/#
        erosion = cv2.erode(dilation,kernel,iterations = 3)
        #/-------- finding the contours -------------------------------/#
        img_ctr,contours,hierachy=cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        cv2.imshow("Contours", img_ctr)		
        
        #image, contours, hierarchy = cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        #--------- checking for 2 contours found or not ----------------#
        if len(contours)==2 :
            numerator+=1
            #img = cv2.drawContours(roi, contours, 1, (0,255,0), 3)
            #------ finding the centroid of the contour ----------------#
            M = cv2.moments(contours[1])
            #print M['m00']
            #print M['m10']
            #print M['m01']
            if M['m00']!=0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                cv2.line(roi,(cx,cy),(cx,cy),(0,0,255),3)
                #print cx,cy
        #-------- checking for one countor presence --------------------#
        elif len(contours)==1:
            numerator+=1
            #img = cv2.drawContours(roi, contours, 0, (0,255,0), 3)
            #------- finding centroid of the countor ----#
            M = cv2.moments(contours[0])
            if M['m00']!=0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                #print cx,cy
                cv2.line(roi,(cx,cy),(cx,cy),(0,0,255),3)
        else:
            denominator+=1
            #print "iris not detected"
        ran=x2-x1
        mid=ran/2
#        if cx<mid:
#            print ("looking left")
#        elif cx>mid:
#            print ("looking right")
    
    cv2.imshow("frame2",frame)
    #cv2.imshow("eye",image)
    return tuple((cx,cy))




RANSAC_MIN_INLIERS = 30
#RANSAC_MIN_INLIERS = 200
    


def mainForTraining2():
    import pygamestuff
    crosshair = pygamestuff.Crosshair([7, 2], quadratic = False)
    vc = cv2.VideoCapture(0) # Initialize the default camera
    if vc.isOpened(): # try to get the first frame
        (readSuccessful, frame) = vc.read()
    else:
        raise(Exception("failed to open camera."))
        return

    MAX_SAMPLES_TO_RECORD = 999999
    recordedEvents=0
    HT = None
    try:
        while readSuccessful and recordedEvents < MAX_SAMPLES_TO_RECORD and not crosshair.userWantsToQuit:            
            pupilOffsetXYList = getOffset2(frame)
            if pupilOffsetXYList is not None: #If we got eyes, check for a click. Else, wait until we do.
                if crosshair.pollForClick():
                    crosshair.clearEvents()
                    #print( (xOffset,yOffset) )
                    #do learning here, to relate xOffset and yOffset to screenX,screenY                    
                    crosshair.record(pupilOffsetXYList)
                    print ("recorded something")
                    crosshair.remove()
                    recordedEvents += 1
                    print("Recorded Events",recordedEvents)
                    if recordedEvents > RANSAC_MIN_INLIERS:
                        resultXYpxpy =np.array(crosshair.result)
                        minSeedSize = 5
                        iterations = 800
                        maxInlierError = 240 #**2
                        pointX, pointY = ransac.ransac2(resultXYpxpy, minSeedSize, iterations, maxInlierError, RANSAC_MIN_INLIERS,pupilOffsetXYList[0],pupilOffsetXYList[1])                              
                        crosshair.drawCrossAt( (pointX, pointY) )
            readSuccessful, frame = vc.read()
        print ("writing")
        crosshair.write() #writes data to a csv for MATLAB
        crosshair.close()

    finally:
        vc.release() #close the camera


if __name__ == '__main__':
    if doTraining:
        mainForTraining2()
    else:
        main()