# for realsense frame number info - https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.frame.html#pyrealsense2.frame.get_frame_number
# for datetime info - https://stackoverflow.com/questions/7999935/python-datetime-to-string-without-microsecond-component
# for how to add aheader to a csv file - https://stackoverflow.com/questions/20347766/pythonically-add-header-to-a-csv-file
# for using random integers - https://www.w3schools.com/python/ref_random_randint.asp
# for finding an element in a list of dictionaries - https://www.geeksforgeeks.org/check-if-element-exists-in-list-in-python/
# for converting to black and white image - https://techtutorialsx.com/2019/04/13/python-opencv-converting-image-to-black-and-white/
# for cropping images - https://learnopencv.com/cropping-an-image-using-opencv/


import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2
from dronekit import connect, VehicleMode, LocationGlobalRelative
import datetime
import rosbag
import csv
import random
from imutils.video import FPS
from cv_bridge import CvBridge

#initialize camera settings and turn on camera
def initialize_pipeline(run):
    pipeline = rs.pipeline()
    config = rs.config()
    #record video to bag file
    config.enable_record_to_file(f'/home/usafa/usafa_472/rover_lab_01/data{run}.bag')
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline

#use to connect to rover in future but not used right now
def connect_device(s_connection):
    print("Connecting to device...")
    device = connect(ip=s_connection, wait_ready=True)
    print("Device connected.")
    print(f"Device version: {device.version}")
    return device

#gets color image frame from stream, displays it, and returns the frame number for use in data organization
def get_video_data(pipeline,fps,bag):
    bridge = CvBridge()
    frame = pipeline.wait_for_frames() #get frame
    color_frame = frame.get_color_frame()
    depth_frame = frame.get_depth_frame()
    #bag.write("color frame", color_frame)
    fps.update()
    if not color_frame: #if no frame is available signal to restart loop
        return None

    color_image = np.asanyarray(color_frame.get_data()) #get image from frame data
    resized_color_image = cv2.resize(color_image, (320,240))
    color_image_fn = color_frame.get_frame_number()
    cv2.imshow('color', color_image) #display image
    cv2.imshow('resized', resized_color_image) #display image

    #convert to hsv and threshold for white values
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    lower = np.array([0,0,150])
    upper = np.array([179,50,255])
    mask = cv2.inRange(hsv_image, lower, upper)
    mask_cropped = mask[160:,:]
    #cv2.imshow("white objects", mask)
    #cv2.imshow("white objects", mask)
    cv2.imshow("white objects cropped", mask_cropped)
    #print(type(mask_cropped))
    #write data to bag file
    
    bag.write("color image", bridge.cv2_to_imgmsg(color_image,'bgr8'))
    bag.write("white object cropped", bridge.cv2_to_imgmsg(mask_cropped,'mono8'))
    bag.write("color image resized", bridge.cv2_to_imgmsg(resized_color_image,'bgr8'))
    return color_image_fn, fps

#in future collect relavent data form the rover, right now return random ints for each field
def get_rover_data():
    throttle = random.randint(0, 2000)
    steering = random.randint(0, 2000)
    heading = random.randint(0, 359)

    return [throttle, steering, heading]

#add data points to the csv file
def append_data(data, index, data_file):
    field_names = ['index', 'throttle', 'steering', 'heading']
    data_dict = {'index': index, 'throttle': data[0], 'steering': data[1], 'heading': data[2]}
    csv.DictWriter(data_file,  fieldnames=field_names).writerow(data_dict)

def main():
    #get starting timestamp for file naming
    run = datetime.datetime.now()
    #create bag file
    bag = rosbag.Bag(f'/home/usafa/usafa_472/rover_lab_01/data{run}.bag', 'w')
    #create csv and add headers
    header = ['index', 'throttle', 'steering', 'heading']
    with open(f'/home/usafa/usafa_472/rover_lab_01/data{run}.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    #reopen csv for the remainder of the run
    data_file = open(f'/home/usafa/usafa_472/rover_lab_01/data{run}.csv', 'a')

    pipeline = initialize_pipeline(run)
    firstImage = True
    while True:
        #start FPS timer
        if firstImage:
            firstImage = False
            fps = FPS().start()

        #get index of current frame
        index, fps = get_video_data(pipeline,fps,bag)
        #restart loop if no frame was available
        if index == None:
            continue

        #get data from rover: throttle, steering, heading
        data = get_rover_data()
        #add data to csv with current frame index
        append_data(data, index, data_file)

#exit condition
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            fps.stop()
            break
#close csv file
    data_file.close()
    cv2.destroyAllWindows()
    print('fps elapsed time: {}\napproximate fps: {}\ndone'.format(fps.elapsed(), fps.fps()))


main()