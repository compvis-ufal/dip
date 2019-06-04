# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 21:29:46 2018

@author: tvieira
"""

#%%
import cv2
from dip import *

#%%
rgb = cv2.imread(os.path.join(folder,'jenny.jpg'), cv2.IMREAD_COLOR)
gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
# Load cascade
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
# Detect face
faces = face_cascade.detectMultiScale(gray, 1.2, 5)
for (x, y, w, h) in faces:
    cv2.rectangle(rgb, (x, y), (x+w, y+h), (255, 0, 0), 2)
# Show the result
cv2.imshow('rgb', rgb)
while True:
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()
