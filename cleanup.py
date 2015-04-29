# This script will take a folder containing video frames and merge them together
# into a video.

import argparse
import sys, os
import shutil

parser = argparse.ArgumentParser(description='Merge frames into video.')
parser.add_argument('img_dir', type=str, help='folder containing image frames')
args = parser.parse_args()

if not os.path.exists(args.img_dir):
  print "Error - the given path is not valid: {}".format(args.img_dir)

filenames = os.listdir(args.img_dir)


for filename in filenames:
    name, ext = os.path.splitext(filename)
    if '.png' in filename:
        new_name = '%s/%s-%04d.png' %(args.img_dir,name.split('-')[0], int(name.split('-')[1]))
        shutil.move('%s/%s' % (args.img_dir, filename), new_name)

