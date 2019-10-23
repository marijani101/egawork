from distutils.core import setup

setup(console=['src.upload.py'], requires=['pytesseract', 'cv2', 'PIL', 'tesserocr', 'pandas'
    , 'numpy', 'pyocr', 'imutils', 'pillowfight', 'tensorflow'])
