import cv2
import numpy as np
from skimage import exposure


class StrecthImage(object):
    
    #function to order points to proper rectangle
    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
    
    
    #function to transform image to four points
    def four_point_transform(self, image, pts):
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
    
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped
    
    
    #function to find two largest countours which ones are may be
    #  full image and our rectangle edged object
    def findLargestCountours(self, cntList, cntWidths):
        newCntList = []
        newCntWidths = []
    
        #finding 1st largest rectangle
        first_largest_cnt_pos = cntWidths.index(max(cntWidths))
    
        # adding it in new
        newCntList.append(cntList[first_largest_cnt_pos])
        newCntWidths.append(cntWidths[first_largest_cnt_pos])
    
        #removing it from old
        cntList.pop(first_largest_cnt_pos)
        cntWidths.pop(first_largest_cnt_pos)
    
        #finding second largest rectangle
        seccond_largest_cnt_pos = cntWidths.index(max(cntWidths))
    
        # adding it in new
        newCntList.append(cntList[seccond_largest_cnt_pos])
        newCntWidths.append(cntWidths[seccond_largest_cnt_pos])
    
        #removing it from old
        cntList.pop(seccond_largest_cnt_pos)
        cntWidths.pop(seccond_largest_cnt_pos)
    
        print('Old Screen Dimentions filtered', cntWidths)
        print('Screen Dimentions filtered', newCntWidths)
        return newCntList, newCntWidths
    
    #driver function which identifieng 4 corners and doing four point transformation
    def convert_object(self, image, screen_size = None, isDebug = False):
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)  # 11  //TODO 11 FRO OFFLINE MAY NEED TO TUNE TO 5 FOR ONLINE
        gray = cv2.medianBlur(gray, 5)
        edged = cv2.Canny(gray, 30, 400)
        _, countours, hierarcy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
        if isDebug : print('length of countours ', len(countours))
    
        imageCopy = img.copy()
        cnts = sorted(countours, key=cv2.contourArea, reverse=True)
        screenCntList = []
        scrWidths = []
        for cnt in cnts:
            peri = cv2.arcLength(cnt, True)  # cnts[1] always rectangle O.o
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            screenCnt = approx
          
            if (len(screenCnt) == 4):
    
                (X, Y, W, H) = cv2.boundingRect(cnt)
                screenCntList.append(screenCnt)
                scrWidths.append(W)
        print('Screens found :', len(screenCntList))
        print('Screen Dimentions', scrWidths)
    
        screenCntList, scrWidths = self.findLargestCountours(screenCntList, scrWidths)
    
        if not len(screenCntList) >=2: 
            return None
        elif scrWidths[0] != scrWidths[1]: 
            return None
        pts = screenCntList[0].reshape(4, 2)
        print('Found bill rectagle at ', pts)
        rect = self.order_points(pts)
        print(rect)
        warped = self.four_point_transform(img, pts)
        warp = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        warp = exposure.rescale_intensity(warp, out_range=(0, 255))
        if(isDebug):
            # cv2.imshow("Original", image)
            # cv2.imshow("warp", warp)
            cv2.imwrite(image,warp)
            #cv2.waitKey(0)
        if(screen_size != None):
            return cv2.cvtColor(cv2.resize(warp, screen_size), cv2.COLOR_GRAY2RGB)
        else:
            return cv2.cvtColor(warp, cv2.COLOR_GRAY2RGB)
