from __future__ import unicode_literals

import subprocess
import os

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from pydub import AudioSegment


def modify(origin, destination):
    sound = AudioSegment.from_file(origin, format='m4a')
    sound.set_frame_rate(44100)
    file_handle = sound.export(destination, format='wav')


def main():
    current_folder = os.getcwd()
    for file in os.listdir(current_folder):
        if os.path.isdir(file):
            continue
        name, ext = os.path.splitext(file)
        if ext == ".wav" or ext == ".py":
            continue
        print(file)
        modify(os.path.join(current_folder, file), os.path.join(current_folder, name + ".wav"))

if __name__ == "__main__":
    main()
