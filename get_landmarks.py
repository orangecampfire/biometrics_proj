import dlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import os

def distances(points):
    dist = []
    for i in range(points.shape[0]):
        for j in range(points.shape[0]):
            p1 = points[i,:]
            p2 = points[j,:]      
            dist.append( math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) )
    return dist

def get_bounding_box(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) 
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y 
	return x, y, w, h

def shape_to_np(shape, num_coords, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((num_coords, 2), dtype=dtype)
 	# loop over the facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, num_coords):
		coords[i] = (shape.part(i).x, shape.part(i).y) 
	# return the list of (x, y)-coordinates
	return coords

def get_landmarks(images, labels, save_directory="", num_coords=5, to_save=False):
    
    print("Getting %d facial landmarks" % num_coords)
    landmarks = []
    new_labels = []
    img_ct = 0
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if num_coords == 5:
      predictor_path = os.path.join(script_dir, 'shape_predictor_5_face_landmarks.dat')
    else:
      predictor_path = os.path.join(script_dir, 'shape_predictor_68_face_landmarks.dat')


    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    for img, label in zip(images, labels):
        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        img_ct += 1
        detected_faces = detector(img, 1)
        for d in detected_faces:
            new_labels.append(label)
            x, y, w, h  = get_bounding_box(d) 
            # Get the landmarks/parts for the face in box d.
            points = shape_to_np(
                    predictor(img, d), 
                    num_coords) 
                        
            dist = distances(points)                   
            landmarks.append(dist)    
            
            if to_save:
                for (x_, y_) in points:
                    cv2.circle(img, 
                           (x_, y_), 
                           1, 
                           (0, 255, 0), 
                           -1)
                plt.figure()
                plt.imshow(img)
                if not os.path.isdir(save_directory):
                    os.mkdir(save_directory)
                plt.savefig(save_directory + label + '%d.png' % img_ct)
                plt.close()
                
            if img_ct % 50 == 0:
                print(f"{img_ct} images with facial landmarks completed.")
                
    return np.array(landmarks), np.array(new_labels)