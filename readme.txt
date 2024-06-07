First install required dependencies:
    pip install -r requirements.txt
(Or use something like anaconda)

sudo apt install nvidia-cuda-toolkit

Initially make sure you update the input.xlsx with all player names

Next determine the AABB coordinates for the frame and names and determine the confidence threshold, the AABBs for the names, scores and level name should be determined and provided:
    python run.py --action test --openai_key ??? --input_movie ./movie.mkv --input_sheet ./input.xlsx --output_sheet ./finals.xlsx --output_frames_dir ./test-track --aabb_names 435,70,570,650 --aabb_scores 820,70,860,650 --aabb_track 400,625,900,675 --threshold_names=0 --threshold_scores=0 --threshold_track=0

Then run the OCR to xlsx extraction:
    python run.py --action run --openai_key ??? --input_movie ./movie.mkv --input_sheet ./input.xlsx --output_sheet ./finals.xlsx --output_frames_dir ./winnaars --aabb_names 435,70,570,650 --aabb_scores 820,70,860,650 --aabb_track 400,625,900,675 --threshold_names=6 --threshold_scores=4 --threshold_track=0.6

Result will be written to the OUTPUT_DIRECTORY/output.xlsx
This file can serve as new input for the next round of OCR on a new video fragment to stich them together.

Cross check the results/tracks if there arent any invalid tracknames, delete them if so, when false positives are found check the thresholds and/or sample_rate

* Note: OPENCV_FFMPEG_READ_ATTEMPTS=100000 might note be required, or might need a different value depending on your OS / hardware capabilities



TODO:
    timestamp opslaan op sheets -> beter nog, sla op in een sqllite db
    bij de levelnaam check -> check de timestamp per sheet om te plaatsen
