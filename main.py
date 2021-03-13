# References: https://levelup.gitconnected.com/playing-chromes-dinosaur-game-using-opencv-19b3cf9c3636

import numpy as np
import cv2
import math
import pyautogui    # buat kontrol keyboard (press space to jump)

# nyalain webcam
capture = cv2.VideoCapture(0)

while capture.isOpened():
    
    # Capture frame dari kamera
    # Grabs, decodes, and return the next video frames
    ret, frame = capture.read()
    
    # Bikin kotak buat tempat naro tangan
    # Param:
    #   - frame: gambarnya di frame
    #   - (100,100): starting coordinate (X,Y) dari rectangle 
    #   - (300,300): ending coordinate (X,Y) dari rectangle
    #   - (0,255,0): warna border rectangle, disini pake warna hijau, kalo biru 255 taro kiri, merah 255 taro kanan
    #   - 0: ketebalan warna rectangle, 0 berarti di sisi doang warnanya
    # crop image: frame dicrop jadi diambil dari rectangle doang imagenya
    cv2.rectangle(frame,(100,100),(300,300),(0,255,0),0)
    crop_image = frame[100:300, 100:300]
    
    # Gaussian blur buat ngurangin noise dan buat image lebih detail
    #  Param:
    #   - crop_image: source image yg mau diblur
    #   - (3,3): lebar dan tinggi kernel, harus positif dan ganjil
    #   - 0: standar deviasi X dan Y diset 0 jd samain kayak kernel, bisa diset custom
    blur = cv2.GaussianBlur(crop_image, (3,3), 0)
    
    # Dari RGB warnanya diganti ke HSV
    # Why? lebih gampang ngeproses HSV dibanding RGB
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    
    # Masking image, jadi imagenya dibikin black and white (white for skin, black for others)
    # Param:
    #   - lower bound: (2,0,0)
    #   - upper bound: (20,255,255)
    # Batasan dipilih buat representasiin warna kulit, harusnya bisa dituning lagi
    # Disini sebenernya bisa dibuat trackbar cuman belom bisa dibikin
    # Why numpy array? size and performance reasons
    mask = cv2.inRange(hsv, np.array([2,0,0]), np.array([20,255,255]))
    
    # Matrix 5x5 buat kernel, seluruh elemen matrix isinya integer 1
    kernel = np.ones((5,5))
    
    # Dilation sama erosion diaplikasikan ke image yg udah ditransformasikan oleh kernel
    # Buat apa? filter background noise
    dilation = cv2.dilate(mask, kernel, iterations = 1)
    erosion = cv2.erode(dilation, kernel, iterations = 1)    
       
    # Image yg udah kena erosion dikasih gaussian blur lagi
    # Param:
    #   - filtered: source image yg udah di gaussian blur
    #   - 127: threshold value, setengah dari max value
    #   - 255: max value
    #   - 0: threshold type, refer to OpenCV docs for further explanation
    filtered = cv2.GaussianBlur(erosion, (3,3), 0)
    ret,thresh = cv2.threshold(filtered, 127, 255, 0)

    # Dapetin contour dari source image
    # Param:
    #   - thresh: source dari image yg udah di threshold
    #   - cv2.RETR_TREE: from docs --> retrieves all of the contours and reconstructs a full hierarchy of nested contours. 
    #   - cv2.CHAIN_APPROX_SIMPLE: ambil endpointnya aja, contoh contour rectangular berarti 4 points
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
    
    
    try:
        # Ambil contour dengan luas maksimum (buat dapet tangan)
        contour = max(contours, key = lambda x: cv2.contourArea(x))
        
        # Bikin kotak pembatas di contour
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(crop_image,(x,y),(x+w,y+h),(0,0,255),0)
        
        # Cari convex hull
        # Why? Biar lebih pasti capture tangannya
        hull = cv2.convexHull(contour)
        
        # Gambar contour dari convex hull tadi
        drawing = np.zeros(crop_image.shape,np.uint8)
        cv2.drawContours(drawing,[contour],-1,(0,255,0),0)
        cv2.drawContours(drawing,[hull],-1,(0,0,255),0)
        
        # Cari convexity defects
        # Misal di convex hullnya ngeliputin semua tangan termasuk ruang antar jari
        # Defectnya berarti ruang antar jari itu
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour,hull)
        
        # Deteksi jari pake aturan kosinus (sqrt(a^2 + b^2 - 2abcostheta))
        count_defects = 0   # ngitung berapa jarinya
        
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            angle = (math.acos((b**2 + c**2 - a**2)/(2*b*c))*180)/3.14
            
            # Kalo sudutnya <= 90 tambahin jari
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_image,far,1,[0,0,255],-1)

            cv2.line(crop_image,start,end,[0,255,0],2)

        # kalo tangan kedetect (kebuka) press SPACE
        if count_defects >= 4:
            pyautogui.press('space')
            # tampilin text JUMP
            cv2.putText(frame,"JUMP", (115,80), cv2.FONT_HERSHEY_SIMPLEX, 2, 2, 2)

    except:
        pass

    # Tampilin video
    cv2.imshow("Gesture", frame)
     
    # Quit pencet 'q'
    if cv2.waitKey(1) == ord('q'):
        break       

capture.release()
cv2.destroyAllWindows()