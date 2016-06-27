# DetectPlates.py

import cv2
import numpy as np
import math
import Definitions
import random

import Preprocess
import DetectChars
import PossiblePlate
import PossibleChar

def detectPlatesInScene(imgOriginalScene):
    possiblePlates = []

    height, width, numChannels = imgOriginalScene.shape

    imgGrayscaleScene = np.zeros((height, width, 1), np.uint8)
    imgThreshScene = np.zeros((height, width, 1), np.uint8)
    imgContours = np.zeros((height, width, 3), np.uint8)

    cv2.destroyAllWindows()

    imgGrayscaleScene, imgThreshScene = Preprocess.preprocess(imgOriginalScene)

    possibleCharsInScene = findPossibleCharsInScene(imgThreshScene)

    listOfListsOfMatchingCharsInScene = DetectChars.findListOfListsOfMatchingChars(possibleCharsInScene)


    for matchingChars in listOfListsOfMatchingCharsInScene:
        possiblePlate = extractPlate(imgOriginalScene, matchingChars)

        if possiblePlate.imgPlate is not None:
            possiblePlates.append(possiblePlate)

    print "\n" + str(len(possiblePlates)) + " possible plates found"

    return possiblePlates

def findPossibleCharsInScene(imgThresh):
    possibleChars = []

    possibleCharsCount = 0

    imgThreshCopy = imgThresh.copy()

    imgContours, contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    height, width = imgThresh.shape
    imgContours = np.zeros((height, width, 3), np.uint8)

    for i in range(0, len(contours)):

        possibleChar = PossibleChar.PossibleChar(contours[i])

        if DetectChars.checkIfPossibleChar(possibleChar):
            possibleCharsCount = possibleCharsCount + 1
            possibleChars.append(possibleChar)

    return possibleChars

def extractPlate(imgOriginal, matchingChars):
    possiblePlate = PossiblePlate.PossiblePlate()

    matchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)

    fltPlateCenterX = (matchingChars[0].intCenterX + matchingChars[len(matchingChars) - 1].intCenterX) / 2.0
    fltPlateCenterY = (matchingChars[0].intCenterY + matchingChars[len(matchingChars) - 1].intCenterY) / 2.0

    ptPlateCenter = fltPlateCenterX, fltPlateCenterY

    intPlateWidth = int((matchingChars[len(matchingChars) - 1].intBoundingRectX + matchingChars[len(matchingChars) - 1].intBoundingRectWidth - matchingChars[0].intBoundingRectX) * Definitions.PLATE_WIDTH_PADDING_FACTOR)

    intTotalOfCharHeights = 0

    for matchingChar in matchingChars:
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight

    fltAverageCharHeight = intTotalOfCharHeights / len(matchingChars)

    intPlateHeight = int(fltAverageCharHeight * Definitions.PLATE_HEIGHT_PADDING_FACTOR)

    fltOpposite = matchingChars[len(matchingChars) - 1].intCenterY - matchingChars[0].intCenterY
    fltHypotenuse = DetectChars.distanceBetweenChars(matchingChars[0], matchingChars[len(matchingChars) - 1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)

    possiblePlate.rrLocationOfPlateInScene = ( tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg )

    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)

    height, width, numChannels = imgOriginal.shape

    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))

    imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter))

    possiblePlate.imgPlate = imgCropped

    return possiblePlate
