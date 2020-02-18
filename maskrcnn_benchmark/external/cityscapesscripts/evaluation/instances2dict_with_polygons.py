#!/usr/bin/python
#
# Convert instances from png files to a dictionary
#

from __future__ import print_function, absolute_import, division

# Cityscapes imports
from maskrcnn_benchmark.external.cityscapesscripts.evaluation.instance import *
from maskrcnn_benchmark.external.cityscapesscripts.helpers.csHelpers import *
from maskrcnn_benchmark.utils.visualization.cv2_util import findContours

import cv2


def instances2dict_with_polygons(imageFileList, verbose=False):
    imgCount = 0
    instanceDict = {}

    if not isinstance(imageFileList, list):
        imageFileList = [imageFileList]

    if verbose:
        print("Processing {} images...".format(len(imageFileList)))

    for imageFileName in imageFileList:
        # Load image
        img = Image.open(imageFileName)

        # Image as numpy array
        imgNp = np.array(img)

        # Initialize label categories
        instances = {}
        for label in labels:
            instances[label.name] = []

        # Loop through all instance ids in instance image
        for instanceId in np.unique(imgNp):
            if instanceId < 1000:
                continue

            instanceObj = Instance(imgNp, instanceId)
            instanceObj_dict = instanceObj.toDict()

            if id2label[instanceObj.labelID].hasInstances:
                mask = (imgNp == instanceId).astype(np.uint8)

                # contours = cv2.findContours(image=mask.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
                # _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours, hierarchy = findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                segmentation = []
                for contour in contours:
                    # Valid polygons have >= 6 coordinates (3 points)
                    if contour.size >= 6:
                        segmentation.append(contour.flatten().tolist())

                instanceObj_dict['contours'] = segmentation

            instances[id2label[instanceObj.labelID].name].append(instanceObj_dict)

        imgKey = os.path.abspath(imageFileName)
        instanceDict[imgKey] = instances
        imgCount += 1

        if verbose:
            print("\rImages Processed: {}".format(imgCount), end=' ')
            sys.stdout.flush()

    if verbose:
        print("")

    #print(instanceDict)
    return instanceDict


def main(argv):
    fileList = []
    if (len(argv) > 2):
        for arg in argv:
            if ("png" in arg):
                fileList.append(arg)
    instances2dict(fileList, True)


if __name__ == "__main__":
    main(sys.argv[1:])
