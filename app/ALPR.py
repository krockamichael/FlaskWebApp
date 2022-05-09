from imutils.perspective import order_points
from imutils.contours import sort_contours
from imutils import resize
from typing import Union
import numpy as np
import pytesseract
import cv2
import re


class PlateDetector:
    def __init__(self):
        # TODO
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))


    def dirty_ratio_check(self, width:int, height:int) -> bool:
        if width == 0 or height == 0:
            return False

        ratio = float(width / height)
        if ratio < 1:
            return False

        if 2.3 < ratio < 8.5: # rectangle ratio
            return True
        elif 1 < ratio < 2.1: # square ratio
            return True

        return False


    def clean_ratio_check(self, image_shape:tuple, width:int, height:int) -> bool:
        ratio = float(width / height)
        if ratio < 1:
            ratio = 1 / ratio

        if 3 < ratio <= 6.4 and width < 0.7 * image_shape[0]: # rectangle ratio - <6 ORIGINALLY
            return True
        elif 1 < ratio < 1.8: # square ratio
            return True

        return False


    def perspective_transformation(self, image:np.ndarray, contour:np.ndarray) -> np.ndarray:
        hull = cv2.convexHull(contour, returnPoints=True)
        box = cv2.minAreaRect(hull)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        src_pts = order_points(box)

        # correct contour coords as they are tied to original image, not contour cutout
        width_offset = min(src_pts[:,0])
        height_offset = min(src_pts[:,1])
        src_pts_corrected = np.array([[x - width_offset, y - height_offset] for x, y in src_pts])

        # use Euclidean distance to get width & height
        width = int(np.linalg.norm(src_pts[0] - src_pts[1]))
        height = int(np.linalg.norm(src_pts[0] - src_pts[3]))
        dst_pts = np.array([[0,0], [width,0], [width,height], [0,height]], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_pts_corrected, dst_pts)
        warped = cv2.warpPerspective(image, M, (width, height))

        return warped


    def get_contours(self, original_image:np.ndarray) -> list:
        plate_area = original_image.shape[0] * original_image.shape[1] * 0.15

        image = cv2.UMat(original_image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equalized = self.clahe.apply(gray)
        blurred = cv2.GaussianBlur(equalized, (7,7), 0)
        sobelx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=-1)

        # do not miss edges with negative slopes (white-to-black)
        sobelx = cv2.UMat.get(sobelx)
        sobelx = np.absolute(sobelx)
        minVal, maxVal = np.min(sobelx), np.max(sobelx)
        sobelx = 255 * ((sobelx - minVal) / (maxVal - minVal))
        sobelx = sobelx.astype(np.uint8)
        sobelx = cv2.UMat(sobelx)

        thresh = cv2.adaptiveThreshold(sobelx, 255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY_INV, blockSize=15, C=10)
        close = cv2.morphologyEx(thresh, op=cv2.MORPH_CLOSE, kernel=np.ones((2,10), np.uint8)) # 3,20

        # get 7 biggest contours for classic pipeline
        final_1 = cv2.morphologyEx(close, op=cv2.MORPH_OPEN, kernel=np.ones((3,20), np.uint8))
        contours_1 = cv2.findContours(final_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours_1 = sorted(contours_1, key=cv2.contourArea, reverse=True)[:7]
        contours_1 = [x for x in contours_1 if cv2.contourArea(x) < plate_area]

        # get 7 biggest contours for light mask pipeline
        light = cv2.morphologyEx(gray, op=cv2.MORPH_CLOSE, kernel=np.ones((5,5), np.uint8))
        light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        bit_and = cv2.bitwise_and(close, close, mask=light)
        dilate = cv2.dilate(bit_and, kernel=np.ones((3,3), np.uint8), iterations=2)
        close = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel=np.ones((3,3), np.uint8))
        final_2 = cv2.morphologyEx(close, op=cv2.MORPH_OPEN, kernel=np.ones((3,20), np.uint8))
        contours_2 = cv2.findContours(final_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours_2 = sorted(contours_2, key=cv2.contourArea, reverse=True)[:7]
        contours_2 = [x for x in contours_2 if cv2.contourArea(x) < plate_area]

        return sorted(contours_1 + contours_2, key=cv2.contourArea, reverse=True)[:8]


    def clean2_plate(self, original_image_shape:tuple, original_plate:np.ndarray) -> Union[np.ndarray, bool]:
        plate = cv2.UMat(original_plate)
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        equalized = self.clahe.apply(gray)
        blurred = cv2.bilateralFilter(equalized, 11, 75, 75)
        #------------------------------ Adaptive Binary ------------------------------
        thresh_binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 7)
        contours_1 = cv2.findContours(thresh_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours_1 = sorted(contours_1, key=cv2.contourArea, reverse=True)[:5]
        #------------------------------ Adaptive Inverse -----------------------------
        thresh_inverse = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
        dilate = cv2.dilate(thresh_inverse, kernel=np.ones((2,2), np.uint8))
        contours_2 = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours_2 = sorted(contours_2, key=cv2.contourArea, reverse=True)[:5]
        #----------------------------------- OTSU ------------------------------------
        thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        close = cv2.morphologyEx(thresh_otsu, op=cv2.MORPH_CLOSE, kernel=np.ones((5,13), np.uint8))
        open = cv2.morphologyEx(close, op=cv2.MORPH_OPEN, kernel=np.ones((1,20), np.uint8))
        contours_3 = cv2.findContours(open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours_3 = sorted(contours_3, key=cv2.contourArea, reverse=True)[:5]
        #-----------------------------------------------------------------------------
        plate_area = original_plate.shape[0] * original_plate.shape[1]
        contours = contours_1 + contours_2 + contours_3
        contours = [x for x in contours if plate_area * 0.4 < cv2.contourArea(x) < plate_area * 0.95]
        contours = sorted(contours, key=cv2.contourArea)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            final_img = original_plate[y:y+h, x:x+w]

            if self.clean_ratio_check(original_image_shape, w, h):
                return final_img, True

        return original_plate, False


    def get_license_plate_candidates(self, image):
        candidates = []
        for cnt in self.get_contours(image):
            x,y,w,h = cv2.boundingRect(cnt)

            if self.dirty_ratio_check(w, h):
                plate = image[abs(y - 2):y + int(h * 1.1), x:x + int(w * 1.1)]
                clean_plate, passed_clean = self.clean2_plate(image.shape, plate)

                if not passed_clean:
                    rot_plate = self.perspective_transformation(plate, cnt)
                    clean_plate, _ = self.clean2_plate(image.shape, rot_plate)

                candidates.append(clean_plate)

        return candidates



class PlateRecognizer:
    def __init__(self):
        self.temp = ''


    def is_text_valid(self, char_text:str) -> bool:
        letter_flag = number_flag = False
        for i in char_text:
            if i.isdigit():
                number_flag = True
            elif i.isalpha():
                letter_flag = True

        return letter_flag and number_flag


    def postprocess_text(self, char_text:str) -> str:
        char_list = list(char_text)
        #########################################################################
        # IF WE COULD BE CERTAIN THAT LICENSE PLATE ARE FROM A SINGULAR COUNTRY #
        #########################################################################

        # text_length = len(char_list)
        # half_len = int(text_length / 2)
        # num_to_letter = {'1':'I', '4':'A', '5':'S', '7':'Z', '8':'B'}

        # for key in num_to_letter.keys():
        #   if key in char_list[0:2]:
        #     char_list[0:2] = char_text[0:2].replace(key, num_to_letter[key])
        #   if text_length % 2 == 0:
        #     if num_to_letter[key] in char_list[half_len - 1:half_len + 1]:
        #       print(char_text[half_len - 1:half_len + 1])
        #       char_list[half_len - 1:half_len + 1] = char_text[half_len - 1:half_len + 1].replace(num_to_letter[key], key)
        #   else:
        #     if num_to_letter[key] in char_list[half_len - 1:half_len + 2]:
        #       print(num_to_letter[key], char_text[half_len - 1:half_len + 2])
        #       char_list[half_len-1:half_len + 2] = char_text[half_len-1:half_len + 2].replace(num_to_letter[key], key)

        if '7' in char_list[0:2]:
          char_list[0:2] = char_text[0:2].replace('7', 'Z')

        return ''.join(char_list)


    def pytesseract_image_to_string(self, image:np.ndarray, oem:int=3, psm:int=7) -> str:
        '''
        oem - OCR Engine Mode
            0 = Original Tesseract only.
            1 = Neural nets LSTM only.
            2 = Tesseract + LSTM.
            3 = Default, based on what is available.
        psm - Page Segmentation Mode
            0 = Orientation and script detection (OSD) only.
            1 = Automatic page segmentation with OSD.
            2 = Automatic page segmentation, but no OSD, or OCR. (not implemented)
            3 = Fully automatic page segmentation, but no OSD. (Default)
            4 = Assume a single column of text of variable sizes.
            5 = Assume a single uniform block of vertically aligned text.
            6 = Assume a single uniform block of text.
            7 = Treat the image as a single text line.
            8 = Treat the image as a single word.
            9 = Treat the image as a single word in a circle.
            10 = Treat the image as a single character.
            11 = Sparse text. Find as much text as possible in no particular order.
            12 = Sparse text with OSD.
            13 = Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.
        '''
        tess_string = pytesseract.image_to_string(image, config=f'--oem {oem} --psm {psm}')
        regex_result = re.findall(r'[A-Z0-9]', tess_string) # filter only uppercase alphanumeric symbols
        return ''.join(regex_result)


    def handle_character_contour(self, contour_list:list, mask:np.ndarray, contour:np.ndarray, colour:str, char_contour_count:int) -> int:
        contour_list.append(contour)
        cv2.drawContours(mask, [contour], -1, (255,255,255) if colour == 'white' else (0,0,0), -1)
        char_contour_count += 1 if colour == 'white' else 0
        return char_contour_count


    def find_characters(self, image_shape:tuple, thresh:cv2.UMat) -> Union[np.ndarray, list, int]:
        mask = cv2.UMat(np.zeros(image_shape, dtype=np.uint8))
        cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        cnts = sort_contours(cnts, method="left-to-right")[0]

        image_area = image_shape[0] * image_shape[1]
        cnt_list = list()
        char_contour_count = 0

        for c in cnts:
            area = cv2.contourArea(c)
            x,y,w,h = cv2.boundingRect(c)
            aspect_ratio = w / h

            if image_area * 0.005 < area < image_area * 0.2 and aspect_ratio < 1 and w < h and h < 0.92 * image_shape[0]:
                if cnt_list:
                    M = cv2.moments(c) # get center of contour
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    if cv2.pointPolygonTest(cnt_list[-1], (cX,cY), False) > 0 or (len(cnt_list) > 1 and cv2.pointPolygonTest(cnt_list[-2], (cX,cY), False) > 0): # A,4,O,... inside cutout should be black
                        char_contour_count = self.handle_character_contour(cnt_list, mask, c, 'black', char_contour_count)
                    else: # add white contour
                        char_contour_count = self.handle_character_contour(cnt_list, mask, c, 'white', char_contour_count)
                else: # add first white contour
                    char_contour_count = self.handle_character_contour(cnt_list, mask, c, 'white', char_contour_count)

        mask = cv2.bitwise_not(mask)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        return mask, cnt_list, char_contour_count


    def handle_square_plate(self, contour_list:list, mask:np.ndarray, image_height:int) -> str:
        # get two top letters
        top_letters = np.array([[0,image_height,0,0], [0,image_height,0,0]])
        for i, c in enumerate(contour_list):
            x,y,w,h = cv2.boundingRect(c)
            if y < top_letters[1,1] and y < top_letters[0,1]:
                top_letters[1] = top_letters[0]
                top_letters[0] = [x,y,w,h]
            elif y < top_letters[1,1]:
                top_letters[1] = [x,y,w,h]
        if top_letters[1,1] + top_letters[1,3] > top_letters[0,1] + top_letters[0,3]:
            top_letters[[0, 1]] = top_letters[[1, 0]]

        ADDITIONAL_PIXELS = 5
        top_half = mask[0:int(top_letters[0,1] + top_letters[0,3] + ADDITIONAL_PIXELS)]
        if len(top_half) < 1:
            return ''
        top_text = self.pytesseract_image_to_string(top_half)

        bot_half = mask[int(top_letters[0,1] + top_letters[0,3] + ADDITIONAL_PIXELS):int(mask.shape[0])]
        if len(bot_half) < 1:
            return ''
        bot_text = self.pytesseract_image_to_string(bot_half)

        return top_text + bot_text


    def get_plate_shape(self, plate_shape:tuple) -> str:
        ratio = float(plate_shape[0] / plate_shape[1])
        ratio = 1 / ratio if ratio < 1 else ratio

        if 3 < ratio <= 6.4: # rectangle ratio
            return 'Rectangle'
        elif 1 < ratio < 1.8: # square ratio
            return 'Square'

        return None


    def get_characters(self, original_image:np.ndarray, plate_shape:str) -> Union[np.ndarray, list, str]:
        original_image = resize(original_image, width=300)
        image = cv2.UMat(original_image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 6)
        mask, cnt_list, char_contour_count = self.find_characters(original_image.shape, thresh_adapt)

        if char_contour_count < 5:
        # if char_contour_count < 5 or char_contour_count > 9:
            return None, None, ''
        if plate_shape == 'Square':
            char_text = self.handle_square_plate(cnt_list, cv2.UMat.get(mask), original_image.shape[0])
        else:
            char_text = self.pytesseract_image_to_string(cv2.UMat.get(mask))

        return mask, cnt_list, char_text


    def get_text(self, candidates:list):
        for cnd in candidates:
            plate_shape = self.get_plate_shape(cnd.shape)
            char_img, char_contour_list, char_text = self.get_characters(cnd, plate_shape)
            if len(char_text) > 4 and self.is_text_valid(char_text):
                char_text = self.postprocess_text(char_text)
                return char_text
