from os import remove, getpid
from PIL import Image
import cv2
import pytesseract
import codecs
import pyocr
import pyocr.builders


import pillowfight
import PIL.Image as Image
import cv2
import numpy as np
from imutils.perspective import four_point_transform
from imutils import  grab_contours


class Output:
    BYTES = 'bytes'
    DATAFRAME = 'data.frame'
    DICT = 'dict'
    STRING = 'string'





def compare_roi(value, roi):
    """
    @brief compares ROIs with the regions of scanned data (ROSD) to result in indexed data which will be placed in the
    database
    @note this works for form. so if the class.form == true, that's  when it will work.
    :param value:
    :param roi:
    """


def ocr(img, ind, language="eng"):  # inputs the image, language(eng or swa or both) as well as type of output needed
    global result
    filename = "{}.png".format(getpid())
    cv2.imwrite(filename, img)
    # load the image as a PIL/Pillow image, apply OCR, and then delete
    # the temporary file
    if language == "both":
        language = "eng+swa"

    if ind == 1:
        result = pytesseract.image_to_string(Image.open(filename), lang=language, output_type=Output.STRING)
        print(result)
    elif ind == 2:
        result = pytesseract.image_to_data(Image.open(filename), lang=language, output_type=Output.DICT)
    elif ind == 3:
        result = pytesseract.image_to_pdf_or_hocr('test.png', extension='pdf')
    remove(filename)
    return result


# call the stroke width transform first then pass it through pyocr.. if its possible you can set it as an option when
# natural photos are used. or when handwriting is used. it is off by default
# added to config.py as SWT | type = boolean | config.SWT  
# will later be added to a class as an entity in the class


def ocrapi(img_path, lang="eng"):
    """

    :type lang: str
    :param lang: 
    :type img_path: str
    """
    tools = pyocr.get_available_tools()
    tool = tools[0]

    if lang == "both":
        lang = "eng+swa"

    word_boxes = tool.image_to_string(
        # Image.open(img_path),
        img_path,
        lang=lang,
        builder=pyocr.builders.TextBuilder())
    return word_boxes

#
# def io(path_to_file):
#     with codecs.open("toto.txt", 'w', encoding='utf-8') as file_descriptor:
#         builder.write_file(file_descriptor, txt)
#
#     print(word_boxes[0].__dict__)

# todo make a def library that will call the boxes, []checkmark
#  set up the image,                     [^]checkmark
#  enable the zone OCR,                  []checkmark
#  as well as the full page ocr..            []checkmark
# todo: def to add the data to a database(sql db)   []checkmark
# todo:                      []checkmark

def stroke_width_transform(img):
    """
    stroke width transform
    for text in natural images. useful for phone images that have not been scanned and converted.
    This algorithm extracts text from natural scenes images.
    To find text, it looks for strokes. Note that it doesn't appear to work well on
    scanned documents because strokes are too small.
    This implementation can provide the output in 3 different ways:

    Black & White : Detected text is black. Background is white.
    Grayscale : Detected text is gray. Its exact color is proportional to the stroke width detected.
    Original boxes : The rectangle around the detected is copied as is in the output image. Rest of the image is white.
    :param img:
    :return:
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = pillowfight.swt(img, output_type=pillowfight.SWT_OUTPUT_ORIGINAL_BOXES)
    """
    the returned image is in form of PIL 
    """
    return img
def get_contours(img):
    """
    @brief: gets the contours of the image created
    :param img:
    :return:
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 2, 255,
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    for c in contours:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCont = approx
            print(screenCont)
            break
    warped = four_point_transform(img, screenCont.reshape(4, 2))
    return warped
def ocrwork(path):
    img = cv2.imread(path)
    img = cv2.resize(img,dsize=(800,1000))
    contured = get_contours(img)
    contured = stroke_width_transform(contured)
    a = ocrapi(contured,"both")
    return a