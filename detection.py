import pywt
import scipy
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
import sklearn.metrics as skm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from helpers import *


class ArtifactSVC:
    def __init__(self, windowSize, stepSize) -> None:
        self.model = None
        self.wS = windowSize
        self.sS = stepSize

    def load_model(self, filename):
        self.model = pickle.load(open(filename, "rb"))

    def save_model(self, filename):
        pickle.dump(self.model, open(filename, "wb"))

    def train(self, data, labels, filename, showMetrics=True):
        trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.3)

        param_grid = {"kernel": ["rbf", "linear"], "C": [1.0, 10.0, 15.0, 20.0]}

        model = SVC()
        grid_search = GridSearchCV(model, param_grid, cv=4, refit=True)

        grid_search.fit(trainX, trainY)

        self.model = grid_search.best_estimator_
        self.save_model(filename)

        if showMetrics == True:
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            print(f"Best Parameters : {best_params}")
            print(f"Best Score : {best_score}")
            print()

            predictions = self.model.predict(testX)
            acc = skm.accuracy_score(testY, predictions)
            prec = skm.precision_score(testY, predictions, average="weighted")
            rec = skm.recall_score(testY, predictions, average="weighted")
            print("Accuracy:", acc)
            print("Precision:", prec)
            print("Recall:", rec)

            cM = skm.confusion_matrix(testY, predictions)
            disp = skm.ConfusionMatrixDisplay(
                confusion_matrix=cM, display_labels=np.unique(labels)
            )
            disp.plot()
            plt.show()

    def predict(self, samp):
        if self.model is not None:
            feat = Misc.compute_features(samp)
            valSigF = np.array(feat).reshape(1, -1)
            valSigPred = self.model.predict(valSigF)[0]

            return valSigPred

        return None

    def get_arifacted_intervals(self, rec, sf):
        if isinstance(rec, EEGSignal):
            recording = rec
        else:
            recording = EEGSignal(rec, sf, standardize=False)
        artifacts = []
        for i in np.arange(0, len(recording.sig) - self.wS * sf, self.sS * sf):
            x1 = int(i)
            x2 = int(i + self.wS * sf)
            samp = recording.sig[x1:x2]

            sampPred = self.predict(samp)

            if sampPred != 0:
                if len(artifacts) == 0:
                    artifacts.append((x1, x2))
                else:
                    if artifacts[len(artifacts) - 1][1] >= x1:
                        artifacts[len(artifacts) - 1] = (
                            artifacts[len(artifacts) - 1][0],
                            x2,
                        )
                    else:
                        artifacts.append((x1, x2))
        return artifacts


class Misc:
    @staticmethod
    def pad(data, level=5):
        min_div = 2**level
        remainder = len(data) % min_div
        pad_len = (min_div - remainder) % min_div

        return np.pad(data, (0, pad_len))

    @staticmethod
    def compute_entropy(data, alpha=1):
        probabilities = np.abs(data) / np.sum(np.abs(data))
        non_zero_probabilities = probabilities[probabilities != 0]
        entropy = np.sum(non_zero_probabilities**alpha)
        entropy = 1 / (1 - alpha) * np.log2(entropy)
        return entropy

    @staticmethod
    def clipVal(val):
        if val < -10:
            return -10
        if val > 10:
            return 10
        return val

    @staticmethod
    def compute_features(samp, mother="sym4", level=5):
        samp = Misc.pad(samp)

        samp_swt = pywt.swt(samp, mother, level, norm=True, trim_approx=True)

        feat = []

        for lev in samp_swt:
            k = scipy.stats.kurtosis(lev)
            eR = Misc.compute_entropy(lev, 2) - 7
            skew = scipy.stats.skew(lev)

            feat.append(Misc.clipVal(k))
            feat.append(Misc.clipVal(eR))
            feat.append(Misc.clipVal(skew))

        return feat


class Example:
    @staticmethod
    def organize_for_training(interval=0.7):
        file_names = ["clean_28", "ocular_28", "musc_28", "cardio_28"]
        cleanSig, ocularSig, muscSig, cardioSig, validationSig = [], [], [], [], []
        cleanSigF, ocularSigF, muscSigF, cardioSigF, validationSigF = [], [], [], [], []

        data = extract_files_data()

        for d in data:
            name = d["id"]

            if "validation_28" in name:
                validationSig = d["data"]

            s = EEGSignal(d["data"], d["sf"], standardize=False)
            epochs = int(s.sig.shape[0] // (s.sf * interval))

            for i in range(epochs):
                samp = s.sig[int(s.sf * interval * i) : int(s.sf * interval * (i + 1))]

                feat = Misc.compute_features(samp)

                if file_names[0] in name:
                    cleanSigF.append(feat)
                elif file_names[1] in name:
                    ocularSigF.append(feat)
                elif file_names[2] in name:
                    muscSigF.append(feat)
                elif file_names[3] in name:
                    cardioSigF.append(feat)
                elif "validation_28" in name:
                    validationSigF.append(feat)

        cleanSigF = np.array(cleanSigF)
        ocularSigF = np.array(ocularSigF)
        muscSigF = np.array(muscSigF)
        cardioSigF = np.array(cardioSigF)

        featureData = np.concatenate(
            [cleanSigF, ocularSigF, muscSigF, cardioSigF], axis=0
        )
        labels = np.array(
            [0] * len(cleanSigF)
            + [1] * len(ocularSigF)
            + [2] * len(muscSigF)
            + [3] * len(cardioSigF)
        )
        return featureData, labels, validationSig


if __name__ == "__main__":
    windowSize = 0.8
    stepSize = 0.6
    filename = "models/my_model.pickle"

    # ----Data Preparation For Training----
    # Required format: data = [[sample1],[sample2], ...], label = [label1, label2, ...]
    data, labels, valS = Example.organize_for_training(windowSize)

    # ----Model Instantiation----
    # It is possible to load a pretrained model or train it on custom data
    m = ArtifactSVC(windowSize, stepSize)
    # m.train(data, labels, filename, showMetrics=True)
    m.load_model(filename)

    # ----Get the artifacted intervals----
    # Returns an array of tuples with all the detected artifacts
    art = m.get_arifacted_intervals(valS, 512)
    print(art)

    # ----Example----
    colors = ["white", "red", "green", "blue"]
    predictions = []
    valSig = EEGSignal(valS, 512, standardize=False)
    plt.plot(valSig.sig)
    for i in np.arange(
        0, len(valSig.sig) - windowSize * valSig.sf, stepSize * valSig.sf
    ):
        x1 = int(i)
        x2 = int(i + windowSize * valSig.sf)
        samp = valSig.sig[x1:x2]
        pred = m.predict(samp)
        plt.axvspan(x1, x2, color=colors[pred], alpha=0.3, label=pred)
    plt.show()
