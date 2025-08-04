# ------------------------ IMPORTING NECESSARY LIBRARIES ------------------------
from tkinter import *          # Tkinter is used for creating the GUI window and widgets
from tkinter import ttk        # ttk provides themed widgets with a modern look
import cv2                     # OpenCV is used for capturing video feed from the webcam
from PIL import Image, ImageTk # PIL is used for handling and converting images to display in Tkinter
import numpy as np             # NumPy for numerical computations (not used here but kept for future use)
import threading               # Threading is used to prevent GUI freezing during video updates
import depthai as dai             #import the depthai library
from deepface import DeepFace     #import the deepface library: <pip install tf-keras> needed
import matplotlib.pyplot as plt   #import the plotting library
from IPython.display import clear_output #to clear the print() output in JupyterNotebook; may not work in VScode
import os
from mtcnn import MTCNN           #import the mtcnn library: <pip install mtcnn> needed
import RPi.GPIO as GPIO
import time
import tensorflow as tf

# ------------------------ GLOBAL VARIABLES ------------------------
# Controls visibility of the "Scanning in progress..." label
is_scan_label_visible = False

# Controls visibility of the progress bar
is_progress_bar_visible = False

is_locked = True

# Variable to store the progress bar value
progress_var = None

# Global reference to the Lock Screen frame
lock_screen = None

# Global reference to the Home Screen frame
home_screen = None

# Default font settings for all buttons and labels
default_font = ("Times New Roman", 22, "bold")

# Default blue color for general buttons
default_button_color = "#0000FF"

# Default red color for exit buttons
default_exit_button_color = "#FF0000"

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="facial_recognition_model_tflite/facial_recognition_tflite_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

GPIO.setwarnings(False)

GPIO.setmode(GPIO.BOARD)

GPIO.setup(16, GPIO.OUT)

servo_motor = GPIO.PWM(16,50)

servo_motor.start(0)

confidence_score = 0.8

# ------------------------ FUNCTION DEFINITIONS ------------------------
def exit_program(*args):
    """
    Displays a pop-up window to confirm if the user wants to exit.
    """
    # Create a pop-up window for exit confirmation
    confirm_exit = Toplevel()
    confirm_exit.title("Exit Confirmation")
    confirm_exit.geometry("400x400")  # Adjusted window size
    confirm_exit.resizable(False, False)

    # Center the confirmation window on the screen
    window_width = 600
    window_height = 250
    position_x = (root.winfo_screenwidth() - window_width) // 2
    position_y = (root.winfo_screenheight() - window_height) // 2
    confirm_exit.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")
    confirm_exit.configure(bg="white")  # Dark background for the window

    # Add a label to display the confirmation message
    Label(
        confirm_exit,
        text="Are you sure you want to exit?",
        font=default_font,
        bg="white",
        fg="black",
        anchor="center"
    ).pack(pady=30)

    # Create a frame for the Yes and No buttons
    button_frame = Frame(confirm_exit, bg="white")
    button_frame.pack(pady=20)

    # Add a "Yes" button to quit the application
    yes_button = Button(
        button_frame,
        text="Yes",
        command=root.quit,  # Closes the entire application
        height=2,
        width=12,
        bg="#00FF00",  # Green color for the Yes button
        fg="white",
        font=default_font
    )
    yes_button.grid(row=0, column=0, padx=20)

    # Add a "No" button to close the confirmation pop-up
    no_button = Button(
        button_frame,
        text="No",
        command=confirm_exit.destroy,  # Closes the confirmation window only
        height=2,
        width=12,
        bg="#FF0000",  # Red color for the No button
        fg="white",
        font=default_font
    )
    no_button.grid(row=0, column=1, padx=20)

def unlock_door():
    """
    Unlock the door using a servo motor.
    """
    is_locked = False
    servo_motor.start(0)
    print('Waiting 1 Second!')
    time.sleep(1)

    print("Turning Back 90 Degrees")
    servo_motor.ChangeDutyCycle(7.5)
    time.sleep(0.3)
    servo_motor.ChangeDutyCycle(0)
    print("Unlock button clicked.")

def lock_door():
    """
    Lock the door using a servo motor.
    """
    is_locked = True
    print("Turning back to 90 degrees")
    servo_motor.ChangeDutyCycle(12)
    print("Waiting 0.5 seconds!")
    time.sleep(0.3)
    servo_motor.ChangeDutyCycle(0)
    time.sleep(0.7)

def update_progress(value=0):
    """
    Incrementally fills the progress bar and updates the GUI.
    After completion, opens the Home Screen.
    """
    global is_scan_label_visible, is_progress_bar_visible

    # Show the scanning label and progress bar
    is_scan_label_visible = True
    is_progress_bar_visible = True
    scan_label.grid()  # Display the "Scanning in progress..." label
    progress_bar.grid()  # Display the progress bar

    # Update the progress bar value
    progress_var.set(value)

    # Continue updating progress until it reaches 100%
    if value < 100:
        root.after(20, update_progress, value + 1)  # Call this function again after 20ms
    else:
        open_home_screen()  # Once complete, switch to the Home Screen

def open_home_screen():
    """
    Opens the Home Screen window after scanning is complete.
    """
    global lock_screen, home_screen, person
    # Hide the Lock Screen
    lock_screen.grid_remove()

    # Create the Home Screen as a new frame
    home_screen = ttk.Frame(root, padding="20 20 20 20")
    home_screen.grid(column=0, row=0, sticky=(N, W, E, S))
    root.title("Home Screen")  # Update the title for the Home Screen
    root.configure(bg="white")  # Set the background color

    # Configure grid weights for layout
    home_screen.columnconfigure([0, 1, 2], weight=1)  # Three columns for button placement
    home_screen.rowconfigure([0, 1, 2, 3], weight=1)  # Rows for layout adjustments

    # Add a label to display the logged-in user (positioned slightly lower)
    Label(
        home_screen,
        text="Logged in as " + person,  # Placeholder text
        font=("Times New Roman", 20, "bold"),  # Larger font size for the label
        fg="black",
        bg="white"
    ).grid(row=1, column=1, pady=20) 

    # Add an "Unlock" button to the left
    Button(
        home_screen,
        text="Unlock",
        command=unlock_door,  # Placeholder for functionality
        height=4,  # Increased height
        width=25,  # Increased width
        bg="#0000FF",  # Fully blue color (RGB = 0000FF)
        fg="white",
        font=("Times New Roman", 22, "bold")  # Larger font size
    ).grid(row=2, column=0, pady=20)

    # Add a "Lock" button in the center
    Button(
        home_screen,
        text="Lock",
        command=lock_door,  # Placeholder for functionality
        height=4,  # Increased height
        width=25,  # Increased width
        bg="#FFA500",  # More orange color (RGB = FFA500)
        fg="white",
        font=("Times New Roman", 22, "bold")  # Larger font size
    ).grid(row=2, column=1, pady=20)

    # Add an "Exit to Lock Screen" button to the right
    Button(
        home_screen,
        text="Exit to Lock Screen",
        command=return_to_lock_screen,
        height=4,  # Increased height
        width=25,  # Increased width
        bg="#FF0000",  # Fully red color (RGB = FF0000)
        fg="white",
        font=("Times New Roman", 22, "bold"),  # Larger font size
        activebackground="#FF0000",  # Fully red when active
        activeforeground="white"
    ).grid(row=2, column=2, pady=20)


def return_to_lock_screen():
    """
    Resets the Home Screen and brings back the Lock Screen.
    """
    global lock_screen, home_screen, is_scan_label_visible, is_progress_bar_visible

    # Remove the Home Screen and show the Lock Screen
    home_screen.grid_remove()
    lock_screen.grid()
    root.title("Biometric Scanner: Lock Screen")
    root.configure(bg="#2B2B2B")

    # Reset flags
    is_scan_label_visible = False
    is_progress_bar_visible = False

def preprocess_image(frame):
    image = cv2.resize(frame,(300,300))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    image = image.astype(np.float32)  # Ensure dtype is float32
    return image

def cnn_predict(image):
    """ Use the loaded CNN model to make a prediction on the input image """
    
    input_shape = input_details[0]['shape']
    input_data = np.array(image, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    #print("Input Details (shape): ", input_shape)

    #print("Input Details (index): ", input_details[0]['index'])

    #model = tf.saved_model.load("facial_recognition_model")  # Adjust with your actual model path
    #prediction = model(image)
    return output_data

def display_result(result):
    """ Display the result of CNN prediction in the GUI """
    # result_label.config(text=f"Prediction: {result}")

def Scan(*args):
    """
    Starts the scanning process by updating the progress bar.
    """
    global is3D, current_frame, person, confidence_score

    #so we need to freeze the frame. rishi does this in his program.
    if current_frame is not None:

        frame_to_process = current_frame.copy()

        # Get the Face ROI
        FaceROI = DetectFace(frame_to_process);
        
        if FaceROI is not None:
            x, y, w, h = FaceROI
            DepthCurrentFrame = DepthFrames.get()
            if DepthCurrentFrame is not None:
                DepthFrameData = DepthCurrentFrame.getCvFrame()
                FaceDepthROI = DepthFrameData[y:y + h, x:x + w]
                DepthStdDev = np.std(FaceDepthROI)

                is3D = DepthStdDev >= threshold
                print(f"Face Detected: {'3D (real)' if is3D else '2D (fake)'}")
                print("Depth Frame Standard Deviation: ", DepthStdDev)
        else:
            print("No Face Detected.")
        
        #freeze and determine if it is 2D or 3D
        #send it thru if it's 3D
        if is3D:
            preprocessed_RGBFrame = preprocess_image(current_frame)
            
            result = cnn_predict(preprocessed_RGBFrame)

            print(np.argmax(result[0])) 
            
            person = None

            #TO-DO: add a confidence score, and if it passes, send the user through

            if np.argmax(result[0]) == 0 and max(result[0]) >= confidence_score:
                update_progress()  # Begin the progress bar update
                print("Welcome Brendan")
                person = "Brendan"    
            elif np.argmax(result[0]) == 1 and max(result[0]) >= confidence_score:
                update_progress()  # Begin the progress bar update
                print("Welcome David")
                person = "David"    
            elif np.argmax(result[0]) == 2 and max(result[0]) >= confidence_score:
                update_progress() 
                print("Welcome Josh")
                person = "Josh"
            elif np.argmax(result[0]) == 3 and max(result[0]) >= confidence_score:
                update_progress() 
                print("Welcome Rishi")
                person = "Rishi" 
            print(result) #display result

        elif not is3D:
            print("Not a real person...")

def update_frame():
    """
    Updates the webcam feed on the Lock Screen.
    """
    global current_frame, video_label

    inRgb = qRgb.tryGet()
    DepthCurrentFrame = DepthFrames.get();

    if inRgb is not None and DepthCurrentFrame is not None:
        # Convert the Frame into a CV image we can handle/want
        RGBFrameData = inRgb.getCvFrame()
        DepthFrameData = DepthCurrentFrame.getCvFrame();

        current_frame = RGBFrameData

        # Resize the frames for display purposes
        resized_RGBFrame = cv2.resize(RGBFrameData, (300, 300))  # Resize to 640x480 pixels for display
        resized_DepthFrame = cv2.resize(DepthFrameData, (300, 300))  # Resize to 640x480 pixels for display

        # Convert the frame to an ImageTk object for Tkinter
        rgb_img = Image.fromarray(cv2.cvtColor(resized_RGBFrame, cv2.COLOR_BGR2RGB))
        rgb_imgtk = ImageTk.PhotoImage(image=rgb_img)
        video_label.imgtk = rgb_imgtk
        video_label.configure(image=rgb_imgtk)

    # Show or hide the scanning label and progress bar
    if is_scan_label_visible:
        scan_label.grid()
    else:
        scan_label.grid_remove()

    if is_progress_bar_visible:
        progress_bar.grid()
    else:
        progress_bar.grid_remove()

    # Call this function again after 10ms
    root.after(10, update_frame)

# ------------------------ LOCK SCREEN INITIALIZATION ------------------------
def initialize_lock_screen():
    """
    Initializes the Lock Screen with video feed, buttons, and progress bar.
    """
    global lock_screen, progress_var, scan_label, progress_bar, video_label

    # Create the Lock Screen frame
    lock_screen = ttk.Frame(root, padding="20 20 20 20")
    lock_screen.grid(column=0, row=0, sticky=(N, W, E, S))

    # Configure grid weights for responsiveness
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    lock_screen.columnconfigure(0, weight=1)
    lock_screen.rowconfigure(0, weight=1)

    # Add a video label to display the webcam feed
    video_label = Label(lock_screen, bg="black")
    video_label.grid(column=0, row=0, columnspan=4, rowspan=2, sticky='nsew', padx=20, pady=20)

    # Add a label to indicate scanning progress
    scan_label = Label(lock_screen, text="Scanning in progress...", font=default_font, fg="black", bg="white")
    scan_label.grid(column=0, row=2, columnspan=4, sticky='ew')
    scan_label.grid_remove()  # Initially hidden

    # Add a progress bar for scanning progress
    progress_var = IntVar()
    progress_bar = ttk.Progressbar(lock_screen, orient='horizontal', length=200, mode='determinate', variable=progress_var)
    progress_bar.grid(column=0, row=3, sticky='w', padx=30, pady=10)
    progress_bar.grid_remove()  # Initially hidden

    # Add a "Scan" button to start the scanning process
    Button(lock_screen, text="Scan", command=Scan, bg=default_button_color, fg="white", font=default_font).grid(column=2, row=5, sticky='nsew', ipadx=60, ipady=30)

    # Add an "Exit" button to open the exit confirmation dialog
    Button(lock_screen, text="Exit", command=exit_program, bg=default_exit_button_color, fg="white", font=default_font).grid(column=3, row=5, sticky='nsew', ipadx=60, ipady=30)

def DetectFace(frame):
    # The OpenCV format image we send into this function is BGR format, detect_faces requires RGB format
    RGBFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB);

    # Try to detect faces
    detectedFaces = detector.detect_faces(RGBFrame);

    # If no faces are detected, return None
    if not detectedFaces:
        return None
        
    # If faces are detected, return the bounding box
    return detectedFaces[0]['box']

# --- Start the pipeline and connect to depth camera ---
# Initialize a new device and call it CameraDevice; everything below is the instructions for it
def process_video():
    global current_frame

    inRgb = qRgb.tryGet()
    if inRgb is not None:
        frame = inRgb.getCvFrame()
        current_frame = frame

# ------------------------ MAIN PROGRAM ------------------------
# Initialize the detector from the MTCNN library
detector = MTCNN();

#--------------------------DEPTH AI CAMERA STUFF---------------------------------

# Boolean variable: If True, then the face is 3D; if False, then the face/frame is 2D
is3D = False;

# Threshold for seeing 3D vs 2D with depth map
threshold = 1000;

# We need to create a pipeline which we will call 'pipeline'
pipeline = dai.Pipeline();

# --- Input node for RGB camera which we will call RGBcam; this node captures the RGB camera frames ---
RGBcam = pipeline.createColorCamera(); #initialize the node
RGBcam.setBoardSocket(dai.CameraBoardSocket.RGB) #configures the port on the OAK-D camera
RGBcam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P) #set the camera resolution
RGBcam.setFps(30) #set the number of frames per second that the camera should be capturing at

# --- Output queue node (RGB_OutQueue) for RGB frames (buffer stage to hold onto the frames temporarily; like a register w/E in digital logic) ---
RGB_OutQueue = pipeline.createXLinkOut();
RGB_OutQueue.setStreamName("RGBStream");

# Link the output of the RGB camera node (which gets the raw RGB frames) to the input of the RGB output queue node
RGBcam.video.link(RGB_OutQueue.input);

# --- Input nodes for Left and Right Depth (mono) cameras; this node captures the depth camera frames (grayscale) but not depth data ---
LeftMonoCam = pipeline.createMonoCamera();                                         #initialize left monochrome camera input node
LeftMonoCam.setBoardSocket(dai.CameraBoardSocket.LEFT);                            #configure node to take from left camera 
LeftMonoCam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P);    #set the resolution to 720p

RightMonoCam = pipeline.createMonoCamera();                                        #initialize right monochrome camera input node
RightMonoCam.setBoardSocket(dai.CameraBoardSocket.RIGHT);                          #configure node to take from right camera 
RightMonoCam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P);   #set the resolution to 720p

# --- Special Node for Depth (Stereo Vision) ---
StereoNode = pipeline.createStereoDepth();                                         #initialize the stereo vision processing node
StereoNode.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY); #set mode for calculating depth map

# --- Output queue node (Depth_OutQueue) for Depth frames (buffer stage for holding depth data from camera) ---
Depth_OutQueue = pipeline.createXLinkOut();                   #initialize the output queue node for the Depth Frames
Depth_OutQueue.setStreamName("DepthStream");                  #set the stream name we are getting the data from
LeftMonoCam.out.link(StereoNode.left);                        #link the left monochrome camera to the stereo vision node's left piece
RightMonoCam.out.link(StereoNode.right);                      #link the right monochrome camera to the stereo vision node's right piece
StereoNode.depth.link(Depth_OutQueue.input);                  #connect the input of the output queue node for the depth frames to the stereo node output

# Start OAK-D device
device = dai.Device(pipeline)
qRgb = device.getOutputQueue(name="RGBStream", maxSize=4, blocking=False)
DepthFrames = device.getOutputQueue(name="DepthStream", maxSize=4, blocking=False)

#--------------------------DEPTH AI CAMERA STUFF END---------------------------------
#current_frame = np.zeros((720, 1280, 3), dtype=np.uint8) #init blank screen

# Create the main Tkinter window
root = Tk()
root.title("Biometric Scanner: Lock Screen")
#root.geometry("400x400")
root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight())) #full screen
root.configure(bg="#2B2B2B")

# Initialize the Lock Screen
initialize_lock_screen()

# Start a thread to continuously update the webcam feed
video_thread = threading.Thread(target=process_video, daemon=True)
video_thread.start()

update_frame()

# Start the Tkinter event loop
root.mainloop()

# Release webcam and clean up OpenCV resources when the program ends
#cap.release()
cv2.destroyAllWindows()

servo_motor.stop()
GPIO.cleanup()

#How to run the program:
#source env/bin/activate
# python Final_GUI.py
