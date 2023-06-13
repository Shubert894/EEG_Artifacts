import os
import json
import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt


def extract_files_data():
    folder_name = os.path.join(os.path.abspath(""), "Recordings")
    folder = [f for f in os.listdir(folder_name)]

    data = []

    for file_name in folder:
        path = os.path.join(folder_name, file_name)
        with open(path) as f:
            d = json.load(f)
            d["id"] = file_name
            data.append(d)

    return data


def compute_overlap(a, b, c):
    diff = np.abs(a - b * c)
    ov = np.sum(diff)
    return ov


def find_scaling_coeff(x1, x2, range):
    ran = np.arange(range[0], range[1], 0.01)
    coeff = []

    for c in ran:
        coeff.append(compute_overlap(x1, x2, c))

    return ran[np.argmin(coeff)], np.min(coeff)


class EEGSignal:
    def __init__(
        self, sig, sf, filter=True, filterFreq=(0.5, 45), standardize=True
    ) -> None:
        self.sf = sf
        self.n = len(sig)
        self.t = self.n / sf

        self.sig = self.initial_thresholding(sig)

        if filter == True:
            self.sig = self.filter(self.sig, filterFreq[0], filterFreq[1], sf, 5)
        if standardize == True:
            self.sig = self.standardize(self.sig)

    def initial_thresholding(self, a):
        a = np.array(a)
        a[np.abs(a) > 4000] = 0
        return a

    def standardize(self, a):
        mean = np.mean(a)
        std = np.std(a)
        return (a - mean) / std

    def normalize(self, a):
        diff = a.max() - a.min()
        return (a - a.min()) / diff

    def filter(self, data, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = ss.butter(order, [low, high], btype="band")
        y = ss.filtfilt(b, a, data)
        return y

    def preview(self):
        timeScale = np.arange(0, self.t, 1 / self.sf)

        plt.xlim((0, self.t))
        plt.plot(timeScale, self.standardize(self.sig))
        plt.show()
