import HandTrackingModule as htm
import cv2 as cv
import numpy as np
import autopy

###########################
# Set camera resolution and screen size
wCam, hCam = 640, 480
wScr, hScr = autopy.screen.size()
smootheing = 3  # Lower smoothing value for smoother movement
plocX, plocY = 0, 0  # Previous location
clocX, clocY = 0, 0  # Current location
###########################

cap = cv.VideoCapture(0)
cap.set(3, wCam)  # Set the width
cap.set(4, hCam)  # Set the height
detector = htm.HandDetector()

def Mouse(img):
    global smootheing, plocX, plocY, clocX, clocY, wScr, wCam, hScr, hCam

    # Find hands
    detector.findhands(img)
    lmlist, bbox = detector.findPosition(img)

    # Check if landmarks list is not empty
    if lmlist and len(lmlist) >= 12:
        try:
            # Get the tip of the index and middle fingers
            Xindex, Yindex = lmlist[8][1], lmlist[8][2]
            Xmidel, Ymidel = lmlist[12][1], lmlist[12][2]

            # Check which fingers are up
            fingers = detector.fingersUp()

            # Debug information
            print(f"Landmarks: {lmlist}")
            print(f"Fingers up: {fingers}")

            # If only index finger is up -> moving mode
            if fingers[1] == 1 and fingers[2] == 0:
                # Convert coordinates from camera to screen size (no boundary restriction)
                xMOUSE = np.interp(Xindex, (0, wCam), (0, wScr))
                yMOUSE = np.interp(Yindex, (0, hCam), (0, hScr))

                # Smoothen the cursor movement
                clocX = plocX + (xMOUSE - plocX) / smootheing
                clocY = plocY + (yMOUSE - plocY) / smootheing

                # Move the mouse
                autopy.mouse.move(clocX, clocY)
                cv.circle(img, (Xindex, Yindex), 15, (20, 180, 90), cv.FILLED)

                # Update previous location
                plocX, plocY = clocX, clocY

            # If both index and middle fingers are up -> clicking mode
            if fingers[1] == 1 and fingers[2] == 1:
                # Find the distance between the index and middle fingers
                length, _ = detector.findDistance(8, 12, img)

                # If distance is short, perform click
                if length < 40:
                    autopy.mouse.click()

        except Exception as e:
            print(f"Error while processing hand landmarks: {e}")

    return img

def add_text(img):
    # Add text to the top-left corner
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1  # Font size
    font_thickness = 2
    color = (255, 255, 255)  # White color

    text = "Lin's Infotechs"
    position = (10, 40)  # Top-left corner

    # Add text to the frame
    cv.putText(img, text, position, font, font_scale, color, font_thickness, cv.LINE_AA)

    return img

def main():
    while True:
        # Read frame from the camera
        success, img = cap.read()
        if not success:
            print("Failed to capture image. Exiting...")
            break

        # Flip the image horizontally for natural interaction
        img = cv.flip(img, 1)

        # Perform mouse control
        img = Mouse(img)

        # Add text to the frame
        img = add_text(img)

        # Display the output
        cv.imshow("Virtual Mouse", img)

        # Break the loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release camera and close all OpenCV windows
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
