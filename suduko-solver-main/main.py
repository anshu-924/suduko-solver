print('Setting UP')
import os
import cv2
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utilities.utils import *
import matplotlib.pyplot as plt
import sudukoSolver
# Capture image using the function from capture_image.py
from capture_image import capture_images
# capture_images()

# Specify the path to the captured image
pathImage = "photos\hand-sudukoo .jpg"

# Load the image from the specified path
# image = cv2.imread(pathImage)

# Check if the image is loaded properly
# if image is None:
#     print("Failed to load image. Please check the file path.")
# else:
#     # Display the loaded image
#     cv2.imshow("Captured Image", image)
#     cv2.waitKey(0)  # Wait until a key is pressed
#     cv2.destroyAllWindows()


# - Image preprocessing

heightImg = 450
widthImg = 450

# - Load the model
model = intializePredectionModel()


img = cv2.imread(pathImage)
img = cv2.resize(img, (widthImg, heightImg))  # RESIZE IMAGE TO MAKE IT A SQUARE IMAGE
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
imgThreshold = preProcess(img)


# ##. FIND ALL COUNTOURS
imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
imgBigContour = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3) # DRAW ALL DETECTED CONTOURS

biggest,maxArea=biggestContour(contours)
print(biggest)

if(biggest.size!=0):
    biggest=reorder(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 25) 
    pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
    pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2) # GER
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgDetectedDigits = imgBlank.copy()
    imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
    
    #### 4. SPLIT THE IMAGE AND FIND EACH DIGIT AVAILABLE
    imgSolvedDigits = imgBlank.copy()
    boxes = splitBoxes(imgWarpColored)
    print(len(boxes))
     ##### - Predict the numbers
    numbers = getPredection(boxes, model)
    print(numbers)
    imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255, 0, 255))
    numbers = np.asarray(numbers)
    posArray = np.where(numbers > 0, 0, 1)
    # print(posArray)

    # print(boxes[0].shape)
    # cv2.imshow("sample",boxes[2])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # img = boxes[2]  # Image is already in grayscale format

    # img_resized = cv2.resize(img, (28, 28))

    # # Normalize the pixel values to be between 0 and 1
    # img_normalized = img_resized / 255.0

    # # Reshape the image to (28, 28, 1)
    # img_reshaped = img_normalized.reshape((28, 28, 1))

    # # Add a batch dimension if required by the model
    # img_ready = np.expand_dims(img_reshaped, axis=0)

    # image = (img_ready).reshape(1,28,28,1)
    # model_pred = model.predict(image, verbose=0)
    # plt.imshow(image.reshape(28,28))
    # print('Prediction of model: {}'.format(model_pred[0]))
    # print(np.argmax(model_pred))

    # for i, box in enumerate(boxes):
    #     box_resized = cv2.resize(box, (28, 28))  # Assuming model expects 28x28 input
    #     box_normalized = box_resized / 255.0
    #     input_data = np.expand_dims(box_normalized, axis=0)
    #     input_data = np.expand_dims(input_data, axis=-1)  # If model expects channels last format
        
    #     prediction = model.predict(input_data)
    #     predicted_digit = np.argmax(prediction, axis=1)[0]
    #     probabilityValue = np.amax(prediction)
    #     print(f"Box {i}: Predicted Digit: {predicted_digit} :Probability {probabilityValue}")
        

# - Rearrange the digits and solve the Sudoku
    board=np.array_split(numbers,9)
    try:
        sudukoSolver.solve(board)
        print("solved")
    except:
        pass
    print(board)
    flatList=[]
    for sublist in board:
        for item in sublist:
            flatList.append(item)
    solvedNumbers =flatList*posArray
    imgSolvedDigits= displayNumbers(imgSolvedDigits,solvedNumbers)
# - Display the output
    pts2 = np.float32(biggest) # PREPARE POINTS FOR WARP
    pts1 =  np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GER
    imgInvWarpColored = img.copy()
    imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
    inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
    imgDetectedDigits = drawGrid(imgDetectedDigits)
    imgSolvedDigits = drawGrid(imgSolvedDigits)

    imageArray = ([img,imgThreshold, imgBigContour],
                  [imgDetectedDigits, imgSolvedDigits,inv_perspective])
    stackedImage = stackImages(imageArray, 1)
    cv2.imshow('Stacked Images', stackedImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No suduko found")

