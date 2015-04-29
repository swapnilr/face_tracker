import cv2
import numpy as np
from collections import namedtuple
import math
import functools
import time
import argparse

Model = namedtuple('Model', ['x', 'y', 'w', 'h'])

NUM_PARTICLES = 1000
SIGMA_MSE = 10
ALPHA = 0.95


def createModel(filename):
    with open(filename) as f:
        x, y = map(float, f.readline().split())
        w, h = map(float, f.readline().split())
        model = Model(x=x, y=y, w=w, h=h)
    return model

def getTextFile(filename):
    return '%s.txt' % filename[:-4]

def getFilename(index=0):
    return 'input/%s' % FILES[index]

def upscale(image):
    return (image * 255).astype(np.uint8)

def downscale(image):
    return image.astype(np.float) / 255

class Filter():

    def __init__(self, image, template):
        self.image = image
        self.template = template

    # Model should take in 2 templates, the current template and the best guess
    def setAppearanceModel(self, model):
        self.appearance_model = model

    # Model is a function that takes in an (y,x) tuple and returns another
    # (y,x) tuple.
    def setDynamicsModel(self, model):
        self.dynamics_model = model

    # Model is a function that takes in an image, a template and an (y,x) tuple
    # and returns a float.
    def setSensorModel(self, model):
        self.sensor_model = model

    # Particles is an array of (y,x) locations
    def compute(self, particles):
        probs = np.zeros(len(particles))
        new_particles = [0.] * len(particles)
        #print particles
        for index, particle in enumerate(particles):
            probs[index] = self.sensor_model(self.image, self.template, particle)
            new_particles[index] = self.dynamics_model(particle)
        # Normalize probs
        probs = probs/np.sum(probs)
        # Resample
        instances = np.random.multinomial(len(particles), probs)
        final_particles = [0.] * len(particles)
        loc = 0
        for index, val in enumerate(instances):
            for j in range(val):
                final_particles[loc] = new_particles[index]
                loc += 1

        height, width = self.template.shape
        pindex = np.argmax(probs)
        particle = new_particles[pindex]
        v = particle[0] - height/2
        u = particle[1] - width/2
        patch = self.image[v:v+height,u:u+width]
        self.template = self.appearance_model(self.template, patch)
        return final_particles, particle, self.template

def simpleAppearanceModel(template, best, alpha=ALPHA):
    return template

def alphaAppearanceModel(template, best, alpha=ALPHA):
    return alpha * best + (1 - alpha) * template

def gaussNoiseDynamicsModel(particle, sigma=1.0):
    return np.around(np.random.normal(scale=sigma,size=2) + particle)


def MSESensorModel(image, template, particle, sigma_mse=SIGMA_MSE):
    height, width = template.shape
    MSE = 0.
    v = max(0, particle[0] - height/2)
    u = max(0, particle[1] - width/2)
    if v + height > image.shape[0]:
        v = image.shape[0] - (v + height)
    if u + width > image.shape[1]:
        u = image.shape[1] - (u + width)

    patch = image[v:v+height,u:u+width]
    th, tw = patch.shape
    if th != height or tw != width:
        print "Size is wrong"
        print particle, height, width, v, u
        patch = np.zeros((template.shape))
    MSE = np.mean(((template - patch)**2))
    return math.exp(-MSE/(2 * (sigma_mse ** 2)))

def test(filename, appearanceModel, outputfileprefix, sigma=0.0,
         dynamicsModel=gaussNoiseDynamicsModel,  alpha=ALPHA, savePatchFile=False, 
         sigma_mse=SIGMA_MSE):
    vid = cv2.VideoCapture(filename)
    index = 1
    ret, orig = vid.read()
    frame = (cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)).astype(np.float)
    print frame.shape
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
    rects = cascade.detectMultiScale(orig, 1.3, 5)
    print rects

    for x, y, w, h in rects:
            cv2.rectangle(orig, (x, y), (x+w, y+h), (127, 255, 0), 2)
            model = Model(x=x, y=y, w=w, h=h)
    cv2.imwrite("%s-%d.png" % (outputfileprefix, index), orig)
    template = frame[int(math.floor(model.y)):int(math.ceil(model.y+model.h)), int(math.floor(model.x)):int(math.ceil(model.x+model.w)) ]
    index += 1

    count = 1
    if savePatchFile:
        cv2.imwrite("%s-patch.png" % outputfileprefix, upscale(template))
        count += 1
    particles = []
    size = int(math.ceil(math.sqrt(NUM_PARTICLES)))
    rows, columns = frame.shape
    ys = np.around(np.linspace(int(model.h/2) + 1, rows-int(model.h/2) - 1, size, endpoint=False))
    xs = np.around(np.linspace(int(model.w/2) + 1, columns-int(model.w/2) - 1, size, endpoint=False))
    for y in ys:
        for x in xs:
            particles.append(np.array((y,x)))

    while vid.isOpened():
        ret, frame = vid.read()
        orig = frame
        if not ret:
            print "ret is False at index %d" % index
            break
        frame = (cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)).astype(np.float)
        f = Filter(frame, template)
        f.setAppearanceModel(functools.partial(appearanceModel, alpha=alpha))
        f.setDynamicsModel(dynamicsModel)
        f.setSensorModel(functools.partial(MSESensorModel, sigma_mse=sigma_mse))
        particles, best, template = f.compute(particles)
        if True:
            frame = cv2.cvtColor(upscale(frame), cv2.COLOR_GRAY2RGB)
            for particle in particles:
                pt = (int(particle[1]), int(particle[0]))
                cv2.circle(orig, pt, 1, (0, 0, 255))
            pt = (int(best[1]), int(best[0]))
            cv2.circle(orig, pt, 5, (255, 0, 0))
            height, width = template.shape
            top = (pt[0] - width/2, pt[1] - height/2)
            bottom = (pt[0]+width/2, pt[1] + height/2)
            cv2.rectangle(orig, top, bottom, (255, 0, 0))
            cv2.imwrite("%s-%d.png" % (outputfileprefix, index), orig)
            count += 1
            #time.sleep(5)
        index += 1
        #if index > 150:
        #    break

def main():
    suffix = time.strftime('%y%m%d%H%M') 
    parser = argparse.ArgumentParser(description='Track objects in file')
    parser.add_argument('filename', type=str, help='name of the video file to track in')
    parser.add_argument('--frame', type=int, default=1, help='Optional. Frame from which to track. 1-indexed')
    parser.add_argument('--output_folder', type=str, default='output', help='name of the output folder')
    parser.add_argument('--output_filename', type=str, default='output_%s' % suffix, help='prefix for the output files')
    args = parser.parse_args()
    output_prefix = '%s/%s' % (args.output_folder, args.output_filename)
    test(args.filename, simpleAppearanceModel, output_prefix, dynamicsModel=functools.partial(gaussNoiseDynamicsModel, sigma=4.4), savePatchFile=True)

if __name__ == '__main__':
    main()
