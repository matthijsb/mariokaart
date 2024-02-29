First install required dependencies:
    pip install -r requirements.txt
(Or use something like anaconda)

Initially make sure you update the input.xlsx with all player names

Next determine the AABB coordinates for the frame and names and determine the confidence threshold:
    for the demo video:
    python ocr.py --test_file test.jpg --aabb 1325,400,1750,800 --aabb_names 1400,410,1566,765 --movie none --sheet none

    for the videos from erik:
    python ocr.py --test_file test.jpg --aabb 896,288,1340,804 --aabb_names 984,288,1160,787 --movie none --sheet none

Then run the OCR to xlsx extraction:
    for the demo video:
    OPENCV_FFMPEG_READ_ATTEMPTS=100000 python ocr.py --movie movie.mp4 --sheet input.xlsx --output ./output --aabb 1325,400,1750,800 --aabb_names 1400,410,1566,765 --east_detection_threshold=100

    For the video from erik:
    OPENCV_FFMPEG_READ_ATTEMPTS=100000 python ocr.py --movie movie.mp4 --sheet input.xlsx --output ./output --aabb 896,288,1340,804 --aabb_names 984,288,1160,787 --east_detection_threshold=350

Result should be an excel sheet

* Note: OPENCV_FFMPEG_READ_ATTEMPTS=100000 might note be required, or might need a different value depending on your OS / hardware capabilities

The result will be written to the OUTPUT_DIRECTORY/output.xlsx
This file can serve as new input for the next round of OCR on a new video file.