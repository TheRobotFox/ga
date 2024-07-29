#!/usr/bin/env python3

import re
import numpy
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import os
import scipy


def transform(img, func):
    def fft_edit(channel2d, func):
        fft = func(numpy.fft.rfft2(channel2d))
        return numpy.fft.irfft2(fft)


def img_fft(img):
    return numpy.array([numpy.fft.rfft2(numpy.asarray(c)) for c in img.split()])
def fft_img(fft):
    data = numpy.fft.irfft2(fft).swapaxes(0,2).swapaxes(0,1).squeeze()
    return Image.fromarray(data.astype(numpy.uint8))


crisp = lambda p: p*numpy.log(p)/2
dots = lambda p: numpy.log(p)*2000

input_img = Image.open("test1.png").convert("RGB")

scale = 1
new_size = (int(input_img.size[0]*scale), int(input_img.size[1]*scale))
input_img = input_img.resize(new_size)

print(numpy.asarray(input_img).shape)

fft = img_fft(input_img)

def video(path, func, frames):

    print(f"Frames: {frames}")

    res = []

    print("Clear old frames")
    [os.remove("frames/"+old) for old in os.listdir("frames")]
    with ThreadPoolExecutor(max_workers=15) as p:
        print("Building thread pool!")
        futures = [p.submit(func, i) for i in range(frames)]

        print("Rendering Frames")
        p = -1
        count = 0
        for _ in as_completed(futures):
            count+=1
            np = int(count/frames*100)
            if np>p:
                p=np
                print(f"{p}%", flush=True)
    print("Starting FFMPEG!")
    cmd_out = ['ffmpeg',
           '-i', 'frames/%d.bmp',  # Indicated input comes from pipe
               '-y',
               path]
    pipe = subprocess.Popen(cmd_out)

count = 800

def infWorld(i):
    frame = []
    for s, c in enumerate(fft):
        scale = 1+ 0.5*i/count
        tmp = scipy.ndimage.zoom(c,scale, order=3)
        frame.append(scipy.ndimage.zoom(tmp,1/scale, order=1))

    img = fft_img(frame).resize(new_size)
    img.save(f"frames/{i}.bmp")

def vhs(i):
    frame = numpy.copy(fft)
    p = int(min(fft.shape[1:])*i/count)
    frame[:,p, p] *= frame[:,p+1, p]

    img = fft_img(frame).resize(new_size)
    img.save(f"frames/{i}.bmp")

def crop(i):
    img = fft_img(fft[:, :-i-1]).resize(new_size)
    img.save(f"frames/{i}.bmp")
def jpeg(i):
    x = int(fft.shape[1]*i/count+1)
    y = int(fft.shape[2]*i/count+1)
    frame = numpy.copy(fft)
    tmp = frame[:, :-x, :-y]
    frame = numpy.pad(tmp, ((0,0), (0, x), (0, y)), constant_values=100)

    # tmp = frame[2, :-x, :-y]
    # hx=int(x/10)
    # frame[2] = numpy.pad(tmp, ((hx, x-hx), (y, 0)), constant_values=0)


    # tmp = frame[0, :-x, :-y]
    # frame[0] = numpy.pad(tmp, ((0, x), (y, 0)), constant_values=0)

    img = fft_img(frame).resize(new_size)
    img.save(f"frames/{i}.bmp")

def dmt(i):
    frame = numpy.copy(fft)
    frame *= 1-numpy.log(frame)*(i/count)

    img = fft_img(frame).resize(new_size)
    img.save(f"frames/{i}.bmp")

video("video.mp4", vhs, count)
