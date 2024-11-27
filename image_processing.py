import cv2
import imutils
import pytesseract
import re
from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"


class image_processing():
    def __init__(self, imgloc):
        self.imgloc = imgloc
        self.screenCnt = None

    def process_image(self):
        self.image = cv2.imread(self.imgloc)
        self.ratio = self.image.shape[0] / 500.0
        self.orig = self.image.copy()
        self.image = imutils.resize(self.image, height=500)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.gray = cv2.GaussianBlur(self.gray, (5, 5), 0)
        self.edged = cv2.Canny(self.gray, 75, 200)
        self.cnts = cv2.findContours(self.edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        self.cnts = imutils.grab_contours(self.cnts)
        self.cnts = sorted(self.cnts, key=cv2.contourArea, reverse=True)[:5]
        for c in self.cnts:
            self.peri = cv2.arcLength(c, True)
            self.approx = cv2.approxPolyDP(c, 0.02 * self.peri, True)
            if len(self.approx) == 4:
                self.screenCnt = self.approx
                break
        print("STEP 1: Edge Detection")
        print("STEP 2: Find contours of paper")
        self.warped = four_point_transform(self.orig, self.screenCnt.reshape(4, 2) * self.ratio)
        self.warped = cv2.cvtColor(self.warped, cv2.COLOR_BGR2GRAY)
        self.T = threshold_local(self.warped, 11, offset=10, method="gaussian")
        self.warped = (self.warped > self.T).astype("uint8") * 255
        print("STEP 3: Apply perspective transform")
        return self.extract_data(self.warped)

    def extract_data(self, imgscanned):
        self.text = pytesseract.image_to_string(imgscanned).strip().split("\n")
        print(self.text)
        self.items_list = []
        for i in range(len(self.text)):
            p = re.search("EAN: ", self.text[i])
            if p:
                dct = {}
                q = self.text[i].strip().split(" ")
                dct["price"] = float(q[3])
                if q[4] in ["I", "i", "L", "1", "tL"]:
                    dct["quantity"] = 1.0
                else:
                    dct["quantity"] = float(q[4])
                dct["item_name"] = self.text[i+1]
                self.items_list.append(dct)
        return self.items_list
