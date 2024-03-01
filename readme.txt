First install required dependencies:
    pip install -r requirements.txt
(Or use something like anaconda)

Initially make sure you update the input.xlsx with all player names

Next determine the AABB coordinates for the frame and names and determine the confidence threshold:
    For the demo video:
    python ocr.py --test_file input-demo/none1.jpg --aabb 1325,400,1750,800 --aabb_names 1400,410,1566,765 --east_detection_threshold=150 --movie none --sheet none

    For the videos from erik:
    python ocr.py --test_file input-erik/none1.jpg --aabb 896,288,1340,804 --aabb_names 984,288,1160,787 --east_detection_threshold=150 --movie none --sheet none

Then run the OCR to xlsx extraction:
    For the demo video:
    OPENCV_FFMPEG_READ_ATTEMPTS=100000 python ocr.py --movie ./input-demo/demo.mp4 --aabb 1325,400,1750,800 --aabb_names 1400,410,1566,765 --east_detection_threshold=150 --sheet ./input-demo/input.xlsx --output ./output-demo

    For the video from erik:
    OPENCV_FFMPEG_READ_ATTEMPTS=100000 python ocr.py --movie ./input-erik/movie2.mp4 --sheet ./input-erik/input.xlsx --output ./output-erik --aabb 896,288,1340,804 --aabb_names 984,288,1160,787 --east_detection_threshold=150

Result will be written to the OUTPUT_DIRECTORY/output.xlsx
This file can serve as new input for the next round of OCR on a new video fragment to stich them together.

* Note: OPENCV_FFMPEG_READ_ATTEMPTS=100000 might note be required, or might need a different value depending on your OS / hardware capabilities

