# Main.py

import cv2
import numpy as np
import os

import DetectChars
import DetectPlates
import Draw
import PossiblePlate

def main():

    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()         # treniranje KNN-a

    if blnKNNTrainingSuccessful == False:                               # Ukoliko nije uspesno izvrseno treniranje vraca se poruka o tome
        print ("\nerror: KNN traning was not successful\n")
        return


    imgOriginalScene  = cv2.imread("6.png")               # otvaranje slike

    if imgOriginalScene is None:                            # ukoliko slika nije uspesno otvorena vraca se poruka o tome
        print("\nerror: image not read from file \n\n")
        os.system("pause")
        return


    possiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)           # detektovanje samih tablica

    possiblePlates = DetectChars.detectCharsInPlates(possiblePlates)        # detektovanje karaktera na tablicama

    cv2.imshow("Original image", imgOriginalScene)

    if len(possiblePlates) == 0:
        print("\nno license plates were detected\n")
    else:
        possiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)


        licPlate = possiblePlates[0]

        if len(licPlate.strChars) == 0:
            print("\nno characters were detected\n\n")
            return
        # end if

        Draw.drawRedRectangleAroundPlate(imgOriginalScene, licPlate)

        print("\nlicense plate read from image = " + licPlate.strChars + "\n")
        print("----------------------------------------")

        Draw.writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)

        cv2.imshow("Result", imgOriginalScene)

        cv2.imwrite("result.png", imgOriginalScene)

    cv2.waitKey(0)

    return

if __name__ == "__main__":
    main()
