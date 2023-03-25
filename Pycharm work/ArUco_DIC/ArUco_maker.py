import numpy as np
import cv2, PIL
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

def aruco_dict_name(name='DICT_4X4_50'):
    ARUCO_DICT = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
        "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
        "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
        "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
        "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
    }
    return ARUCO_DICT[name],name
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

def aruco_sheet(aruco_dict,nx = 4,ny = 3,px=700):
    if type(aruco_dict) == str:
        aruco_dict, aruco_name = aruco_dict_name(aruco_dict)
        aruco_dict = aruco.Dictionary_get(aruco_dict)
    else:
        aruco_name = str(aruco_dict)
    fig = plt.figure()
    for i in range(1, nx*ny+1):
        ax = fig.add_subplot(ny,nx, i)
        img = aruco.drawMarker(aruco_dict,i, px)
        plt.imshow(img, cmap = mpl.cm.gray, interpolation = "nearest")
        ax.axis("off")
    os.makedirs(rf"C:\Users\Daniel\PycharmProjects\Daniel_Thesis\ArUco_DIC\ArUcos\{aruco_name}",exist_ok=True)
    plt.savefig(rf"C:\Users\Daniel\PycharmProjects\Daniel_Thesis\ArUco_DIC\ArUcos\{aruco_name}\{ny}x{nx}_markers.jpeg")
    plt.show()

def aruco_sheets(aruco_dict,ids=range(1,11),nx = 4,ny = 3,px=700):
    if type(aruco_dict) == str:
        aruco_dict, aruco_name = aruco_dict_name(aruco_dict)
        aruco_dict = aruco.Dictionary_get(aruco_dict)
    else:
        aruco_name = str(aruco_dict)
    for id in ids:
        fig = plt.figure()
        for i in range(1, nx*ny+1):
            ax = fig.add_subplot(ny,nx, i)
            img = aruco.drawMarker(aruco_dict,id, px)
            plt.imshow(img, cmap = mpl.cm.gray, interpolation = "nearest")
            ax.axis("off")
        os.makedirs(rf"C:\Users\Daniel\PycharmProjects\Daniel_Thesis\ArUco_DIC\ArUcos\{aruco_name}",exist_ok=True)
        plt.savefig(rf"C:\Users\Daniel\PycharmProjects\Daniel_Thesis\ArUco_DIC\ArUcos\{aruco_name}\id_{id}_({ny}x{nx}).jpeg")

def aruco_ids(aruco_dict,ids=range(1,21),px=700):
    if type(aruco_dict) == str:
        aruco_dict, aruco_name = aruco_dict_name(aruco_dict)
        aruco_dict = aruco.Dictionary_get(aruco_dict)
    else:
        aruco_name = str(aruco_dict)
    for id in ids:
        img = aruco.drawMarker(aruco_dict, id, px)
        os.makedirs(rf"C:\Users\Daniel\PycharmProjects\Daniel_Thesis\ArUco_DIC\ArUcos\{aruco_name}\ids",exist_ok=True)
        cv2.imwrite(rf"C:\Users\Daniel\PycharmProjects\Daniel_Thesis\ArUco_DIC\ArUcos\{aruco_name}\ids\id_{id}_{px}px.jpeg",img)

# aruco_sheets('DICT_5X5_50',nx=5,ny=4)
# aruco_sheets('DICT_6X6_50',nx=5,ny=4)
# aruco_ids('DICT_5X5_50',px=300)
aruco_ids('DICT_6X6_50',px=700)
