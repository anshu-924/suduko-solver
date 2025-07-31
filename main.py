
print('Setting UP')
import os
from pathlib import Path  
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from utilities.utils import *
import matplotlib.pyplot as plt
import sudukoSolver  # algorithm to solve the sudoku
from capture_image import capture_images

# FIXED MODEL LOADING FUNCTIONS
def combine_branches(inputs):
    """Custom function for combining branches - must match the original exactly"""
    b1, b2, b3, weights = inputs
    return (b1 * tf.expand_dims(weights[:, 0], 1) + 
           b2 * tf.expand_dims(weights[:, 1], 1) + 
           b3 * tf.expand_dims(weights[:, 2], 1))

def load_model_with_fallbacks(model_path):

    try:
        print("🔍 Trying: Enhanced custom objects")
        custom_objects = {
            'combine_branches': combine_branches,
            'mse': tf.keras.losses.MeanSquaredError(),
            'categorical_crossentropy': tf.keras.losses.CategoricalCrossentropy(),
            'accuracy': tf.keras.metrics.CategoricalAccuracy(),
            'Adam': tf.keras.optimizers.Adam,
            'mean_squared_error': tf.keras.losses.MeanSquaredError(),
            'MeanSquaredError': tf.keras.losses.MeanSquaredError(),
        }
        
        model = load_model(model_path, custom_objects=custom_objects)
        print(f"✅ Model loaded successfully with enhanced custom objects")
        return model
    except Exception as e:
        print(f"❌ Enhanced loading failed: {e}")
    
    # Method 2: No compile (load weights only, recompile manually)
    try:
        print("🔍 Trying: No compile method")
        custom_objects = {'combine_branches': combine_branches}
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
        print(f"✅ Model loaded successfully (without compilation)")
        
        # Recompile manually for inference
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        print(f"❌ No-compile loading failed: {e}")
    
    # Method 3: Global custom objects
    try:
        print("🔍 Trying: Global custom objects")
        tf.keras.utils.get_custom_objects()['combine_branches'] = combine_branches
        model = load_model(model_path)
        print(f"✅ Model loaded successfully with global custom objects")
        return model
    except Exception as e:
        print(f"❌ Global loading failed: {e}")
    
    # Method 4: Minimal custom objects with string references
    try:
        print("🔍 Trying: String reference method")
        custom_objects = {
            'combine_branches': combine_branches,
            'mse': 'mse',
            'categorical_crossentropy': 'categorical_crossentropy',
            'accuracy': 'accuracy',
        }
        model = load_model(model_path, custom_objects=custom_objects)
        print(f"✅ Model loaded successfully with string references")
        return model
    except Exception as e:
        print(f"❌ String reference loading failed: {e}")
    
    print("❌ All loading methods failed!")
    return None

def getPredectionWithConfidence_robust(boxes, model, confidence_threshold=0.6):
    """
    Robust prediction function that works with different model loading methods
    Returns predictions with confidence gating
    """
    result = []
    print(f"\n🔍 Processing {len(boxes)} boxes with confidence threshold {confidence_threshold}")
    
    for i, image in enumerate(boxes):
        # Prepare Image
        img=preprocess_cell(image)  # Ensure image is preprocessed correctly
        
        try:
            # Get predictions from the model
            outputs = model.predict(img, verbose=0)
            
            # Handle different output formats
            main_probs = None
            confidence_scores = None
            
            if isinstance(outputs, dict):
                # Multi-output model with named outputs
                main_probs = outputs.get('main_logits', None)
                confidence_scores = outputs.get('confidence', None)
                
                # If main_logits not found, try alternative names
                if main_probs is None:
                    for key in ['output_1', 'dense', 'predictions']:
                        if key in outputs:
                            main_probs = outputs[key]
                            break
                
                # If still not found, use first available output
                if main_probs is None:
                    main_probs = list(outputs.values())[0]
                    
            elif isinstance(outputs, (list, tuple)):
                # List of outputs
                main_probs = outputs[0]  # Assume first output is main prediction
                confidence_scores = outputs[1] if len(outputs) > 1 else None
            else:
                # Single output
                main_probs = outputs
                confidence_scores = None
            
            # Ensure main_probs is valid
            if main_probs is None:
                raise ValueError("Could not extract main predictions from model output")
            
            # Get predicted class
            predicted_class = np.argmax(main_probs, axis=1)[0]
            max_prob = np.max(main_probs[0])
            
            # Handle confidence
            if confidence_scores is not None:
                if hasattr(confidence_scores, 'shape') and len(confidence_scores.shape) > 1:
                    confidence = float(confidence_scores[0][0])
                else:
                    confidence = float(confidence_scores[0])
            else:
                # Fallback: use max probability as confidence
                confidence = float(max_prob)
            
            # Convert to 1-9 range (assuming model outputs 0-8)
            digit_1_to_9 = predicted_class + 1
            
            # Apply confidence threshold
            if confidence > confidence_threshold:
                result.append(digit_1_to_9)
                print(f"Box {i:2d}: Predicted={digit_1_to_9}, Confidence={confidence:.3f}, MaxProb={max_prob:.3f} ✓")
            else:
                result.append(0)  # Empty cell
                print(f"Box {i:2d}: Rejected (low confidence: {confidence:.3f}) ✗")
                
        except Exception as e:
            print(f"❌ Error predicting box {i}: {e}")
            # Fallback to simple prediction
            try:
                simple_pred = model.predict(img, verbose=0)
                if isinstance(simple_pred, (list, tuple)):
                    simple_pred = simple_pred[0]
                elif isinstance(simple_pred, dict):
                    simple_pred = list(simple_pred.values())[0]
                
                predicted_class = np.argmax(simple_pred, axis=1)[0]
                digit_1_to_9 = predicted_class + 1
                confidence = np.max(simple_pred[0])
                
                if confidence > confidence_threshold:
                    result.append(digit_1_to_9)
                    print(f"Box {i:2d}: Fallback Predicted={digit_1_to_9}, Confidence={confidence:.3f} ✓")
                else:
                    result.append(0)
                    print(f"Box {i:2d}: Fallback Rejected (low confidence: {confidence:.3f}) ✗")
            except Exception as fallback_error:
                print(f"❌ Fallback prediction also failed for box {i}: {fallback_error}")
                result.append(0)
    
    return result
def preprocess_cell(cell: np.ndarray) -> np.ndarray:
    """
    Takes a single 56×56 (or similar) Sudoku cell, returns a (1,28,28) float32 tensor.
    """
    # 1. binary threshold to isolate black digit
    _, thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # 2. find tight bounding box around the white blobs (the digit strokes)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x,y,w,h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        digit_roi = thresh[y:y+h, x:x+w]
    else:                               # empty cell → all zeros
        digit_roi = np.zeros((28,28), np.uint8)

    # 3. resize to 20 × 20 keeping aspect ratio
    digit_roi = cv2.resize(digit_roi, (20, 20), interpolation=cv2.INTER_NEAREST)

    # 4. place the 20×20 digit in a 28×28 canvas, centred
    canvas = np.zeros((28,28), np.uint8)
    x_off = (28-20)//2
    y_off = (28-20)//2
    canvas[y_off:y_off+20, x_off:x_off+20] = digit_roi

    # 5. final normalisation → float32 0-1 range
    canvas = canvas.astype("float32")/255.0
    return np.expand_dims(canvas, 0).reshape(1, 28, 28, 1)     # shape (1,28,28)

# MAIN EXECUTION STARTS HERE
if __name__ == "__main__":
    # Uncomment to capture new image
    # capture_images()

    # Specify the path to the captured image
    pathImage = Path("photos") / "opencv_frame_1.png"  # Change to your image path]"

    # Load the image from the specified path
    image = cv2.imread(str(pathImage)) 

    # Check if the image is loaded properly
    if image is None:
        print("❌ Failed to load image. Please check the file path.")
        print(f"Looking for image at: {pathImage}")
        exit()

    print("✅ Image loaded successfully.")

    # Image preprocessing
    heightImg = 504
    widthImg = 504

    image = cv2.resize(image, (widthImg, heightImg))  # RESIZE IMAGE TO MAKE IT A SQUARE IMAGE
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
    imgThreshold = preProcess(image)

    # FIND ALL CONTOURS
    imgContours = image.copy() # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = image.copy() # COPY IMAGE FOR DISPLAY PURPOSES
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3) # DRAW ALL DETECTED CONTOURS

    try:
        biggest, maxArea = biggestContour(contours)
        print(f"✅ Biggest contour found with area: {maxArea}")
    except Exception as e:
        print(f"❌ No suitable sudoku contour found: {e}")
        biggest = None

    # Load model with comprehensive fallback methods
    model = load_model_with_fallbacks('telkhatamhai.keras')
    
    if model is None:
        print("❌ Failed to load model with all methods. Please check:")
        print("   1. Model file exists: are_yrr.h5")
        print("   2. TensorFlow version compatibility")
        print("   3. File permissions")
        exit()

    # Check model architecture
    try:
        print(f"\n📊 Model summary:")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output names: {list(model.output_names) if hasattr(model, 'output_names') else 'N/A'}")
        print(f"   Number of outputs: {len(model.outputs) if hasattr(model, 'outputs') else 'N/A'}")
    except Exception as e:
        print(f"⚠️ Could not inspect model architecture: {e}")

    if biggest is not None and biggest.size != 0:
        biggest = reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 25) 
        pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
        pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2) # GET TRANSFORMATION MATRIX
        imgWarpColored = cv2.warpPerspective(image, matrix, (widthImg, heightImg))
        imgDetectedDigits = imgBlank.copy()
        imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        
        # SPLIT THE IMAGE AND FIND EACH DIGIT AVAILABLE
        imgSolvedDigits = imgBlank.copy()
        boxes = splitBoxes(imgWarpColored)
        print(f"\n📦 Total boxes extracted: {len(boxes)}")
        
        # Uncomment to see sample boxes for debugging
        # cv2.imshow("Sample box 0", boxes[0])
        
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        box = boxes[4]
        cv2.imshow("Sample box 2", boxes[4])
        #save boxes[4]
        cv2.imwrite("box_4_image.png", box)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if box.ndim == 3:                       # in case the warp stayed BGR
            box = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)

        input_tensor = preprocess_cell(box)     # → (1,28,28)
        pred = model.predict(input_tensor)
        digit = int(np.argmax(pred))
        print("Predicted digit:", digit)
        # cv2.imshow("Sample box 2", boxes[4])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # box2 = boxes[4]
        # box2 = cv2.resize(box2, (28, 28))   
        # print("shape of box2:", box2.shape)
        # box2 = box2.astype('float32') / 255.0  # Normalize pixel values
        # box2 = box2.reshape(1, 28, 28)
        # # cv2.imshow("Box 2 Resized", )  # Convert back to displayable format
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # pred0 = model.predict(box2)
        # digit0 = np.argmax(pred0)
        # print(f"🔢 Prediction for box[0]: {pred0},{digit0}") 
                
        # Predict the numbers using the robust confidence model
        print("\n🔍 Predicting digits with confidence scoring...")
        numbers = getPredectionWithConfidence_robust(boxes, model, confidence_threshold=0.998)
        
        print(f"\n📊 Detected numbers: {numbers}")
        print("📋 Grid layout:")
        grid = np.array(numbers).reshape(9, 9)
        for row_idx, row in enumerate(grid):
            row_str = ' '.join([str(x) if x != 0 else '.' for x in row])
            print(f"Row {row_idx + 1}: {row_str}")
        
        # Display detected digits
        imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255, 0, 255))
        numbers = np.asarray(numbers)
        posArray = np.where(numbers > 0, 0, 1)
        
        if len(numbers) != 81:
            raise ValueError(f"❌ Grid split failed: expected 81 cells, got {len(numbers)}")

        # Count detected digits
        detected_count = np.sum(numbers > 0)
        empty_count = np.sum(numbers == 0)
        print(f"\n📈 Detection summary:")
        print(f"   Detected digits: {detected_count}/81")
        print(f"   Empty cells: {empty_count}/81")
        print(f"   Detection rate: {(detected_count/81)*100:.1f}%")

        # Solve the Sudoku
        board = numbers.reshape(9, 9)
        print("\n🧩 Sudoku board:",board)
        print("\n🧩 Solving Sudoku...")
        try:
            # if detected_count < 17:  # Minimum clues needed for unique solution
            #     print(f"⚠️ Warning: Only {detected_count} digits detected. Need at least 17 for unique solution.")
            sudukoSolver.solve(board)
            print("✅ Sudoku solved successfully!")
        except Exception as e:
            print(f"❌ Failed to solve Sudoku: {e}")
            print("💡 This might be due to:")
            print("   - Insufficient detected digits")
            print("   - Incorrect digit predictions")
            print("   - Invalid sudoku puzzle")
            
        print("\n🎯 Solved board:")
        for row_idx, row in enumerate(board):
            row_str = ' '.join([str(x) for x in row])
            print(f"Row {row_idx + 1}: {row_str}")
        
        # Create solution overlay (only new digits)
        flatList = board.flatten()         
        solvedNumbers = (flatList * posArray).astype(int)  # Only show newly solved digits
        
        imgSolvedDigits = displayNumbers(imgSolvedDigits, solvedNumbers)
        
        # Display the output with inverse perspective
        pts2 = np.float32(biggest) # PREPARE POINTS FOR WARP
        pts1 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GET INVERSE TRANSFORMATION MATRIX
        imgInvWarpColored = image.copy()
        imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
        inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, image, 0.5, 1)
        
        # Add grids for better visualization
        imgDetectedDigits = drawGrid(imgDetectedDigits)
        imgSolvedDigits = drawGrid(imgSolvedDigits)

        # Stack all images for display
        imageArray = ([image, imgThreshold, imgBigContour],
                      [imgDetectedDigits, imgSolvedDigits, inv_perspective])
        stackedImage = stackImages(imageArray, 1)
        
        print("\n👀 Displaying results...")
        print("Press any key to close the display window")
        cv2.imshow('Sudoku Solver Results', stackedImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print("🎉 Process completed successfully!")
        
        # Optional: Save results
        try:
            cv2.imwrite('sudoku_result.png', stackedImage)
            print("💾 Results saved to: sudoku_result.png")
        except Exception as e:
            print(f"⚠️ Could not save results: {e}")
        
    else:
        print("❌ No sudoku puzzle found in the image")
        print("💡 Troubleshooting tips:")
        print("   - Ensure the sudoku puzzle is clearly visible")
        print("   - Check lighting conditions (avoid shadows)")
        print("   - Make sure the puzzle edges are clearly defined")
        print("   - Try adjusting the camera angle for better perspective")
        print("   - Ensure the puzzle is roughly square-shaped in the image")
        
        # Display the processed images for debugging
        print("🔍 Displaying diagnostic images...")
        cv2.imshow('Original Image', image)
        cv2.imshow('Threshold Image', imgThreshold)
        cv2.imshow('Detected Contours', imgContours)
        print("Press any key to close diagnostic windows")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print("📝 Debug info:")
        print(f"   Total contours found: {len(contours) if 'contours' in locals() else 'N/A'}")
        print(f"   Image dimensions: {image.shape}")
        print(f"   Threshold image type: {imgThreshold.dtype}")