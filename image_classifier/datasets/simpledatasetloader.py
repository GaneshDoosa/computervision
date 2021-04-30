import numpy as np
import cv2
import os

class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors
        
        # if the processors are None, initialize them as an empty list
        if self.preprocessors is None:
            self.preprocessors = []
    
    def load(self, image_paths, verbose=-1):
        # initialize list of features and labels
        data = []
        labels = []
        
        for (i, image_path) in enumerate(image_paths):
            # load the image and extract class label assuming that our path has the following format
            # /path/to/dataset/{class}/{image}.png
            image= cv2.imread(image_path)
            label = image_path.split(os.path.sep)[-2]
            
            # check to see if preprocessors are not None
            if self.preprocessors is not None:
                # loop over the preprocessors and apply each to the image
                for p in self.preprocessors:
                    image = p.preprocess(image)
                
            # treat preprocessed image as a feature vector
            data.append(image)
            labels.append(label)
            
            # show an update every 'verbose' images
            if verbose>0 and i>0 and (i+1) % verbose==0:
                print("[INFO] processed {}/{}".format(i+1, len(image_paths)))
                    
        # return a tuple of data and labels
        return (np.array(data), np.array(labels))