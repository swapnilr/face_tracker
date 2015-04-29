# face_tracker
Face Tracker

Sample Usage:
python face_detector.py Intro.mp4 --output_filename='Intro'
rm output/Intro-patch.png # Removing the patch file
python cleanup.py output
ffmpeg -i output2/Intro-%04d.png -c:v libx264 -pix_fmt yuv420p out.mp4
