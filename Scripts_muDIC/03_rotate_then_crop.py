import cv2
import os
import glob
import numpy as np
from tqdm import tqdm

# folder = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_8\1-14442C107152C0D200\right\2_8_DIC'
# folder = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_8\1-14442C107152C0D200\left\2_8_DIC'
# folder = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_1\4-14442C107152C0D200\left\2_1_DIC'
# folder = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_1\4-14442C107152C0D200\right\2_1_DIC'
# folder = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_2\1-14442C107152C0D200\right\2_2_DIC'
# folder = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_2\1-14442C107152C0D200\left\2_2_DIC'
# folder = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_4\1-14442C107152C0D200\left\2_4_DIC'
# folder = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_4\1-14442C107152C0D200\right\2_4_DIC'
# folder = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_5\1-14442C107152C0D200\right\2_5_DIC'
# folder = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_5\1-14442C107152C0D200\left\2_5_DIC'
# folder = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_11\1-14442C10E17CC0D200\left\2_11_DIC'
# folder = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_11\1-14442C10E17CC0D200\right\2_11_DIC'
# folder = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_14\5-14442C10E17CC0D200\left\2_14_DIC'
# folder = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_14\5-14442C10E17CC0D200\right\2_14_DIC'
# folder = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_16\5-14442C10E17CC0D200\left\2_16_DIC'
# folder = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_16\5-14442C10E17CC0D200\right\2_16_DIC'
# folder = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_16\5-14442C107152C0D200\left\2_16_DIC'
# folder = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_16\5-14442C107152C0D200\right\2_16_DIC'
# folder = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_17\1-14442C107152C0D200\left\2_17_DIC'
# folder = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_17\1-14442C107152C0D200\right\2_17_DIC'
# folder = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_19\1-14442C107152C0D200\left\2_19_DIC'
# folder = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_19\1-14442C107152C0D200\right\2_19_DIC'
# folder = r'C:\Users\dmchacon\Documents\OAK-D Videos\3_2\1-14442C107152C0D200\left\3_2_DIC'
# folder = r'C:\Users\dmchacon\Documents\OAK-D Videos\3_2\1-14442C107152C0D200\right\3_2_DIC'
# folder = r'C:\Users\dmchacon\Documents\OAK-D Videos\3_4\1-14442C107152C0D200\left\3_4_DIC'
# folder = r'C:\Users\dmchacon\Documents\OAK-D Videos\3_4\1-14442C107152C0D200\right\3_4_DIC'
# folder = r'C:\Users\dmchacon\Documents\OAK-D Videos\3_6\2-14442C107152C0D200\left\3_6_DIC'
# folder = r'C:\Users\dmchacon\Documents\OAK-D Videos\3_6\2-14442C107152C0D200\right\3_6_DIC'
# folder = r'C:\Users\dmchacon\Documents\OAK-D Videos\3_9\4-14442C107152C0D200\left\3_9_DIC'
# folder = r'C:\Users\dmchacon\Documents\OAK-D Videos\3_9\4-14442C107152C0D200\right\3_9_DIC'
# folder = r'C:\Users\dmchacon\Documents\OAK-D Videos\3_11\1-14442C10E17CC0D200\left\3_11_DIC'
# folder = r'C:\Users\dmchacon\Documents\OAK-D Videos\3_11\1-14442C10E17CC0D200\right\3_11_DIC'
# folder = r'C:\Users\dmchacon\Documents\OAK-D Videos\3_12\1-14442C10E17CC0D200\left\3_12_DIC'
# folder = r'C:\Users\dmchacon\Documents\OAK-D Videos\3_12\1-14442C10E17CC0D200\right\3_12_DIC'
# folder = r'C:\Users\dmchacon\Documents\OAK-D Videos\3_14\1-14442C10E17CC0D200\color\3_14_DIC'
folder = r'C:\Users\dmchacon\Documents\OAK-D Videos\2_1\right\2_1_DIC'
# folder = r''
# folder = r''

cropped = os.path.join(os.path.split(folder)[0],'cropped')
os.makedirs(cropped,exist_ok=True)

images = glob.glob(os.path.join(folder,'*.png'))

cropping = False
start = False
x_start, y_start, x_end, y_end, angle = 0, 0, 0, 0, 0
image = cv2.imread(images[0])
oriImage = image.copy()

def get_angle(event, x, y, flags, param):
    global x_start, y_start, x_end, y_end, angle, start
    if event == cv2.EVENT_LBUTTONDOWN and not start:
        x_start, y_start, x_end, y_end = x, y, x, y
        start = True
    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if start == True:
            x_end, y_end = x, y
    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONDOWN:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        angle = np.rad2deg(np.arctan2((y_end-y_start),(x_end-x_start)))
        print(f'Angle:\t{angle} deg.')
        start = False

def rotate_image(image, angle, cX = None, cY = None):
    """
    Sprout (2019)
    Function to rotate the input image by a specified angle
    :param image: input image to rotate
    :param angle: number of degrees the image will be rotated by
    :return: rotated input image
    """
    # determine center of the image
    (h, w) = image.shape[:2]
    if cX == None and cY == None:
        (cX, cY) = (w // 2, h // 2)

    # get the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then get the sine and cosine
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # Calculate bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH))

def mouse_crop(event, x, y, flags, param):
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping
    # if the left mouse button was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is being
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True
    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y
    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False # cropping is finished
        refPoint = [(x_start, y_start), (x_end, y_end)]
        if len(refPoint) == 2: #when two points were found
            print('x_start, y_start, x_end, y_end:',x_start, y_start, x_end, y_end)
            roi = rotated[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            cv2.imshow("Cropped", roi)

cv2.namedWindow("angle")
cv2.setMouseCallback("angle", get_angle)

while True:
    i = image.copy()
    if not start:
        cv2.imshow("angle", image)
    elif start:
        cv2.line(i, (x_start, y_start), (x_end, y_end), (0, 255, 0), 1)
        cv2.imshow("angle", i)
    if cv2.waitKey(1) == ord('q'):
        break

rotated = rotate_image(image.copy(), angle)

cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_crop)

while True:
    i = rotated.copy()
    if not cropping:
        cv2.imshow("Image", rotated)
    elif cropping:
        cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        cv2.imshow("Image", i)
    if cv2.waitKey(1) == ord('q'):
        break

for path in tqdm(images):
    name = os.path.basename(path)
    img = rotate_image(cv2.imread(path), angle)
    roi = img[y_start:y_end, x_start:x_end]
    cv2.imwrite(os.path.join(cropped,name),roi)