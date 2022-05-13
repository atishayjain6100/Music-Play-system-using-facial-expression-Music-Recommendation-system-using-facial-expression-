import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pandastable import Table, TableModel
from tensorflow.keras.preprocessing import image
import datetime
from threading import Thread
import random
import os
from pygame import mixer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from Spotipy import *  
import time
import pandas as pd
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
ds_factor=0.6

emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('model.h5')

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0:"Angry",1:"Disgusted",2:"Fearful",3:"Happy",4:"Neutral",5:"Sad",6:"Surprised"}
music_dist={0:"songs/angry.csv",1:"songs/disgusted.csv ",2:"songs/fearful.csv",3:"songs/happy.csv",4:"songs/neutral.csv",5:"songs/sad.csv",6:"songs/surprised.csv"}
global last_frame1                                    
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1 
show_text=[0]


''' Class for calculating FPS while streaming. Used this to check performance of using another thread for video streaming '''
class FPS:
	def __init__(self):
		# store the start time, end time, and total number of frames
		# that were examined between the start and end intervals
		self._start = None
		self._end = None
		self._numFrames = 0
	def start(self):
		# start the timer
		self._start = datetime.datetime.now()
		return self
	def stop(self):
		# stop the timer
		self._end = datetime.datetime.now()
	def update(self):
		# increment the total number of frames examined during the
		# start and end intervals
		self._numFrames += 1
	def elapsed(self):
		# return the total number of seconds between the start and
		# end interval
		return (self._end - self._start).total_seconds()
	def fps(self):
		# compute the (approximate) frames per second
		return self._numFrames / self.elapsed()


''' Class for using another thread for video streaming to boost performance '''
class WebcamVideoStream:
    	
		def __init__(self, src=0):
			self.stream = cv2.VideoCapture(src,cv2.CAP_DSHOW)
			(self.grabbed, self.frame) = self.stream.read()
			self.stopped = False

		def start(self):
				# start the thread to read frames from the video stream
			Thread(target=self.update, args=()).start()
			return self
			
		def update(self):
			# keep looping infinitely until the thread is stopped
			while True:
				# if the thread indicator variable is set, stop the thread
				if self.stopped:
					return
				# otherwise, read the next frame from the stream
				(self.grabbed, self.frame) = self.stream.read()

		def read(self):
			# return the frame most recently read
			return self.frame
		def stop(self):
			# indicate that the thread should be stopped
			self.stopped = True

''' Class for reading video stream, generating prediction and recommendations '''
class VideoCamera(object):

	def capture_frame():
# 		TIMER = int(5)
  
# # Open the camera
# 		cap = cv2.VideoCapture(0)
  
 
# 		while True:
# 			# Read and display each frame
# 			ret, img = cap.read()
# 			cv2.imshow('a', img)
# 		    # check for the key pressed
# 			k = cv2.waitKey(125)
 
# 			# set the key for the countdown
# 			# to begin. Here we set q
# 			# if key pressed is q
# 			if k == ord('q'):
# 				prev = time.time()
 
# 				while TIMER >= 0:
# 					ret, img = cap.read()
 
#             # Display countdown on each frame
#             # specify the font and draw the
#             # countdown using puttext
# 					font = cv2.FONT_HERSHEY_SIMPLEX
# 					cv2.putText(img, str(TIMER),
# 								(200, 250), font,
# 								7, (0, 255, 255),
# 								4, cv2.LINE_AA)
# 					cv2.imshow('a', img)
# 					cv2.waitKey(125)
 
#             # current time
# 					cur = time.time()
		
# 					# Update and keep track of Countdown
# 					# if time elapsed is one second
# 					# than decrease the counter
# 					if cur-prev >= 1:
# 						prev = cur
# 						TIMER = TIMER-1
 
# 					else:
# 						ret, img = cap.read()
			
# 					# Display the clicked frame for 2
# 					# sec.You can increase time in
# 					# waitKey also
# 						cv2.imshow('a', img)
			
# 						# time for which image displayed
# 						cv2.waitKey(2000)
 
#             # HERE we can reset the Countdown timer
#             # if we want more Capture without closing
#             # the camera
 
#     # Press Esc to exit
# 			elif k == 27:
# 				break
 
# # close the camera
# 		cap.release()
  
# # close all the opened windows
# 		cv2.destroyAllWindows()
		vid = cv2.VideoCapture(0)

		while (True):

			# Capture the video frame
			# by frame
			ret, frame = vid.read()

			# Display the resulting frame
			cv2.imshow('frame', frame)

			# the 'q' button is set as the
			# quitting button you may use any
			# desired button of your choice
			if cv2.waitKey(1) or 0xFF == ord('q'):
				break

		# After the loop release the cap object
		vid.release()
		# Destroy all the windows
		cv2.destroyAllWindows()     

	def get_frame(self):
		global cap1
		global df1
		# cap1 = WebcamVideoStream(src=0).start()
		# image = cap1.read()
		# print('Image PATH', image)
		vid = cv2.VideoCapture(0)
		ret, image = vid.read()
		vid.release()
		# image = cv2.imread('46.jpg')
		image=cv2.resize(image,(600,500))
		gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		face_rects=face_cascade.detectMultiScale(gray,1.3,5)
		df1 = pd.read_csv(music_dist[show_text[0]])
		df1 = df1[['Name','Album','Artist']]
		df1 = df1.head(15)
		for (x,y,w,h) in face_rects:
			cv2.rectangle(image,(x,y-50),(x+w,y+h+10),(0,255,0),2)
			roi_gray_frame = gray[y:y + h, x:x + w]
			cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
			prediction = emotion_model.predict(cropped_img)

			maxindex = int(np.argmax(prediction))
			show_text[0] = maxindex 
			#print("===========================================",music_dist[show_text[0]],"===========================================")
			#print(df1)
			cv2.putText(image, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
			df1 = music_rec()
			print('Expression detected ==> ', emotion_dict[maxindex])
			
		global last_frame1
		last_frame1 = image.copy()
		pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)     
		img = Image.fromarray(last_frame1)
		img = np.array(img)
		ret, jpeg = cv2.imencode('.jpg', img)
		print('||||||||||||||||||||| Classification completed |||||||||||||||||||')
		print("==============================================="+emotion_dict[maxindex])

		if(emotion_dict[maxindex]=="Happy"):
			mixer.init()
			path = '/home/atishay/gs_project/project1/songs/Happy/'
			files = os.listdir(path)
			d = random.choice(files)
			print(d)
			mixer.music.load("/home/atishay/gs_project/project1/songs/Happy/"+d)
			mixer.music.set_volume(0.7)
			mixer.music.play()

		elif(emotion_dict[maxindex]=="Sad"):
			mixer.init()
			path = '/home/atishay/gs_project/project1/songs/Sad/'
			files = os.listdir(path)
			d = random.choice(files)
			print(d)
			mixer.music.load("/home/atishay/gs_project/project1/songs/Sad/"+d)
			mixer.music.set_volume(0.7)
			mixer.music.play()

		elif(emotion_dict[maxindex]=="Angry"):
			mixer.init()
			path = '/home/atishay/gs_project/project1/songs/Angry/'
			files = os.listdir(path)
			d = random.choice(files)
			print(d)
			mixer.music.load("/home/atishay/gs_project/project1/songs/Angry/"+d)
			mixer.music.set_volume(0.7)
			mixer.music.play()

		elif(emotion_dict[maxindex]=="Disgusted"):
			mixer.init()
			path = '/home/atishay/gs_project/project1/songs/Disgusted/'
			files = os.listdir(path)
			d = random.choice(files)
			print(d)
			mixer.music.load("/home/atishay/gs_project/project1/songs/Disgusted/"+d)
			mixer.music.set_volume(0.7)
			mixer.music.play()

		elif(emotion_dict[maxindex]=="Fearful"):
			mixer.init()
			path = '/home/atishay/gs_project/project1/songs/Fearful/'
			files = os.listdir(path)
			d = random.choice(files)
			print(d)
			mixer.music.load("/home/atishay/gs_project/project1/songs/Fearful/"+d)
			mixer.music.set_volume(0.7)
			mixer.music.play()

		elif(emotion_dict[maxindex]=="Neutral"):
			mixer.init()
			path = '/home/atishay/gs_project/project1/songs/Neutral/'
			files = os.listdir(path)
			d = random.choice(files)
			print(d)
			mixer.music.load("/home/atishay/gs_project/project1/songs/Neutral/"+d)
			mixer.music.set_volume(0.7)
			mixer.music.play()

		else:
			mixer.init()
			path = '/home/atishay/gs_project/project1/songs/Surprised/'
			files = os.listdir(path)
			d = random.choice(files)
			print(d)
			mixer.music.load("/home/atishay/gs_project/project1/songs/Surprised/"+d)
			mixer.music.set_volume(0.7)
			mixer.music.play()
		return jpeg.tobytes(), df1

def music_rec():
	# print('---------------- Value ------------', music_dist[show_text[0]])
	df = pd.read_csv(music_dist[show_text[0]])
	df = df[['Name','Album','Artist']]
	df = df.head(15)
	return df
