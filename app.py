import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
from helper import *
from solver import *
import cv2
from tensorflow.keras.models import load_model

model = load_model('trained_model.h5')

st.title("Live sudoku solver")
st.write("place the sudoku puzzle in center of the frame.")


class VideoProcessor:
	def recv(self, frame):
		heightImg = 450
		widthImg = 450
		img = frame.to_ndarray(format="bgr24")
		img = cv2.resize(img, (widthImg, heightImg))  # RESIZE IMAGE TO MAKE IT A SQUARE IMAGE
		imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  
		imgThreshold = preProcess(img)
		imgContours = img.copy() 
		imgBigContour = img.copy() 
		contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
		cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3) # DRAW ALL DETECTED CONTOURS
		
		biggest, maxArea = biggestContour(contours) # FIND THE BIGGEST CONTOUR
		if biggest.size != 0:
			biggest = reorder(biggest)
			cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 25) # DRAW THE BIGGEST CONTOUR
			pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
			pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
			matrix = cv2.getPerspectiveTransform(pts1, pts2) # GER
			imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
			imgDetectedDigits = imgBlank.copy()
			imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)

			### SPLIT THE IMAGE AND FIND EACH DIGIT AVAILABLE

			imgSolvedDigits = imgBlank.copy()
			boxes = splitBoxes(imgWarpColored)
			refine_boxes = clean_squares(boxes)
			# print(len(boxes))
			
			### PREDICTION
			numbers = getPredection(refine_boxes, model)

			# print(numbers)
			imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255, 0, 255))
			numbers = np.asarray(numbers)
			posArray = np.where(numbers > 0, 0, 1)
			# print(posArray)

			## SOLVING THE BOARD
			board = np.array_split(numbers,9)
			if isValid(board):
				solve(0, 0, board)
			flatList = []
			for sublist in board:
				for item in sublist:
					flatList.append(item)
			solvedNumbers =flatList*posArray
			imgSolvedDigits= displayNumbers(imgSolvedDigits,solvedNumbers)

			## OVERLAY SOLUTION
			pts2 = np.float32(biggest) # PREPARE POINTS FOR WARP
			pts1 =  np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
			matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GER
			imgInvWarpColored = img.copy()
			imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
			inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
			imgDetectedDigits = drawGrid(imgDetectedDigits)
			imgSolvedDigits = drawGrid(imgSolvedDigits)

			final_image = inv_perspective
			# imageArray = ([img, imgBigContour],
						# [imgDetectedDigits,inv_perspective])
			# imageArray = ([imgContours, imgBigContour, imgWarpColored],
			# [imgInvWarpColored, inv_perspective, imgSolvedDigits])
			# stackedImage = stackImages(imageArray, 1)
			# cv2.imshow('Stacked Images', stackedImage)
			# ret, jpeg = cv2.imencode('.jpg', stackedImage)
			# return jpeg.tobytes()

		else:
			final_image = img
			# ret, jpeg = cv2.imencode('.jpg', img)
			# imageArray = ([imgContours, imgBigContour])
			# stackedImage = stackImages(imageArray, 1)
		# 	return jpeg.tobytes()
		# 	print("No Sudoku Found")

		return av.VideoFrame.from_ndarray(final_image, format="bgr24")


webrtc_streamer(key="example", video_processor_factory=VideoProcessor, rtc_configuration={  # Add this line
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

# ctx = webrtc_streamer(
#     key="example",
#     video_processor_factory=VideoProcessor,
#     rtc_configuration={  # Add this line
#         "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
#     }
# )