import pywt
import numpy as np


class WaveletThresholding:
    def __init__(self, wavelet="sym4", level=5, mode="hard"):
        self.wavelet = wavelet
        self.level = level
        self.mode = mode

    def pad(self, data):
        min_div = 2**self.level
        remainder = len(data) % min_div
        pad_len = (min_div - remainder) % min_div

        return np.pad(data, (0, pad_len))

    def intervals_to_mask(self, intervals, size=None):
        mask = np.zeros(size, dtype=bool)
        for i, j in intervals:
            mask[i:j] = True
        return mask

    def run(self, signal, artifacts, fs=None, reference=None):
        sig_ = self.pad(signal)
        coeffs = pywt.swt(sig_, self.wavelet, self.level, norm=True, trim_approx=True)
        coeffs = np.array(coeffs)

        artifact_mask = self.intervals_to_mask(artifacts, coeffs.shape[1])

        k = np.median(np.abs(coeffs), axis=1) / 0.6745
        thresholds = np.sqrt(2 * np.log(coeffs.shape[1])) * k

        for ws, th in zip(coeffs, thresholds):
            ws[artifact_mask] = self.threshold(ws[artifact_mask], th)

        rec = pywt.iswt(coeffs, wavelet=self.wavelet, norm=True)

        return rec[: len(signal)]

    def threshold(self, coeffs, threshold):
        if self.mode == "hard":
            return np.where(np.abs(coeffs) <= threshold, coeffs, 0.0)
        elif self.mode == "soft":
            return np.clip(coeffs, -threshold, threshold)
        return None


class WaveletQuantileNormalization:
    # Adaptive Single-Channel EEG Artifact Removal With Applications to Clinical Monitoring. Dora, et al.
    # Might require a license for industrial use

    def __init__(self, wavelet="sym4", mode="periodization", alpha=1, n=30):
        self.wavelet = wavelet
        self.alpha = alpha
        self.mode = mode
        self.n = n

    def run(self, signal, artifacts):
        restored = signal.copy()

        # Iterate over the artifacted intervals
        for n, (i, j) in enumerate(artifacts):
            # We consider the signal between indices `a` and `b`. The artifact
            # correspond to the interval `i` to `j` and we will keep two portions
            # of the signal, before and after the artifact, that will be used as
            # a reference. The references correspond to the intervals `a` to `i`,
            # b` to `j`.
            min_a = 0
            max_b = signal.size

            if n > 0:
                # `a` must be bigger than the end of the previous artifact
                min_a = artifacts[n - 1][1]
            if n + 1 < len(artifacts):
                # `b` must be smaller than the start of the next artifact
                max_b = artifacts[n + 1][0]

            size = j - i  # the artifacted interval size
            level = int(np.log2(size / self.n))  # max decomposition level for DWT

            if level < 1:
                continue

            # We define `a` and `b` based on the desired size of the reference
            # signal intervals.
            ref_size = max(self.n * 2**level, size)
            a = max(min_a, i - ref_size)
            b = min(max_b, j + ref_size)

            # Calculate DWT
            coeffs = pywt.wavedec(
                signal[a:b], self.wavelet, mode=self.mode, level=level
            )

            # Iterate over wavelet coefficient by level
            for cs in coeffs:
                # Define the inteval indices `ik`, `jk` we have to use
                # since the signal is downsampled based on the level
                # of decomposition in the DWT.
                k = int(np.round(np.log2(b - a) - np.log2(cs.size)))
                ik, jk = np.array([i - a, j - a]) // 2**k
                refs = [cs[:ik], cs[jk:]]
                if len(refs[0]) == 0 and len(refs[1]) == 0:
                    continue

                # Transport the CDFs of the absolute value
                order = np.argsort(np.abs(cs[ik:jk]))
                inv_order = np.empty_like(order)
                inv_order[order] = np.arange(len(order))

                vals_ref = np.abs(np.concatenate(refs))
                ref_order = np.argsort(vals_ref)
                ref_sp = np.linspace(0, len(inv_order), len(ref_order))
                vals_norm = np.interp(inv_order, ref_sp, vals_ref[ref_order])

                # Attenuate the coefficients
                r = vals_norm / np.abs(cs[ik:jk])
                cs[ik:jk] *= np.minimum(1, r) ** self.alpha

            # Reconstruct the signal
            rec = pywt.waverec(coeffs, self.wavelet, mode=self.mode)
            restored[i:j] = rec[i - a : j - a][: restored[i:j].shape[0]]

        return restored


if __name__ == "__main__":
    pass
