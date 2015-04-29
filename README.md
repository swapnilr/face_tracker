# Face Tracker

Face detector and tracker:
1. Finds faces using a cascade filter
2. Tracks faces using a particle filter

##Sample Usage:
```
python face_detector.py Intro.mp4 --output_filename='Intro'
rm output/Intro-patch.png # Removing the patch file
python cleanup.py output
ffmpeg -i output2/Intro-%04d.png -c:v libx264 -pix_fmt yuv420p out.mp4
```

##Outputs
###Intro.mp4
* (Original)[https://youtu.be/zCPZ4DYTnk8]
* (Tracking Only)[https://youtu.be/VSTTC0IQHnE]
* (Detection & Tracking)[https://youtu.be/MfTe8e61GKI]
###08-01.avi
* (Original)[https://youtu.be/n6piIxmyNk0]
* (Tracking Only)[https://youtu.be/lzdnHCHfhH0]
* (Detection & Tracking)[https://youtu.be/5JR1CSldIyA]

##Credits
08-01.avi downloaded from - http://www.videorecognition.com/db/video/faces/cvglab/

Dmitry O. Gorodnichy  Video-based framework for face recognition in video. 
Second Workshop on Face Processing in Video (FPiV'05) in Proceedings of Second Canadian Conference on Computer and Robot Vision (CRV'05), pp. 330-338, Victoria, BC, Canada, 9-11 May, 2005. ISBN 0-7695-2319-6. NRC 48216.


