import cv2
import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy


class Filters:
    @classmethod
    def butter_lowpass_filter(
        cls, data, cutoff, nyq, order, sampling_frequency, time, plot=False
    ):
        """ Lowpass filter for the input signal.
        :param data:
        :type data: librosa.Audio
        :param cutoff: 
        :type cutoff: int
        :param nyq: 
        :type nyq: float
        :param order: 
        :type order: int
        :param sampling_frequency: 
        :type sampling_frequency: float
        :param time: 
        :type time: ndarray
        :param plot: defaults to False
        :type plot: bool, optional
        :return: 
        :rtype: librosa.Audio
        """
        normalized_cutoff = cutoff / nyq
        numerator_coeffs, denominator_coeffs = scipy.signal.butter(
            order, normalized_cutoff, btype="low", analog=False, fs=sampling_frequency
        )
        filtered_signal = scipy.signal.lfilter(
            numerator_coeffs, denominator_coeffs, data
        )
        if plot:
            plt.plot(time, data, "b-", label="signal")
            plt.plot(time, filtered_signal, "g-", linewidth=2, label="filtered signal")
            plt.legend()
            plt.show()
        return filtered_signal

    @classmethod
    def butter_highpass_filter(
        cls, data, cutoff, nyq, order, sampling_frequency, time, plot=False
    ):
        """ High pass filter for the input signal.
        :param data: 
        :type data: librosa.Audio
        :param cutoff: 
        :type cutoff: int
        :param nyq: 
        :type nyq: float
        :param order: 
        :type order: int
        :param sampling_frequency: 
        :type sampling_frequency: float
        :param time: 
        :type time: ndarray
        :param plot: defaults to False
        :type plot: bool, optional
        :return: 
        :rtype: librosa.Audio
        """
        normalized_cutoff = cutoff / nyq
        numerator_coeffs, denominator_coeffs = scipy.signal.butter(
            order, normalized_cutoff, btype="high", analog=False, fs=sampling_frequency
        )
        filtered_signal = scipy.signal.lfilter(
            numerator_coeffs, denominator_coeffs, data
        )
        if plot:
            plt.plot(time, data, "b-", label="signal")
            plt.plot(time, filtered_signal, "g-", linewidth=2, label="filtered signal")
            plt.legend()
            plt.show()
        return filtered_signal


def plot_audio(
    y,
    sr,
    cutoff=500,
    n_mels=512,
    n_fft=int(22050 / 0.98 * 4),
    hop_length=int(0.0255 * 22050 / 0.98),
    window_length=None,
    extra_power=1,
    f_max=1000,
):
    htk = False

    window_length = window_length or n_fft
    hop_length = hop_length or window_length // 4
    input_data, sampling_frequency = y, sr

    # Adapted from https://github.com/AI-Cloud-and-Edge-Implementations/Project15-G4/blob/8f16003ce1e6aa0658bb71e91c7180a4729348fb/elephantcallscounter/data_analysis/analyse_sound_data.py
    duration = len(input_data) / sampling_frequency
    # plots upto sampling rate/2(Nyquist theorem)
    # Filter requirements.
    fs = sampling_frequency  # sample rate, Hz
    nyq = 0.5  # Nyquist Frequency
    order = 4  # sin wave can be approx represented as quadratic
    time = np.linspace(0, duration, len(input_data), endpoint=False)

    lowpass_signal = Filters.butter_lowpass_filter(
        input_data, cutoff, nyq, order, sampling_frequency, time, plot=False
    )

    cutoff_high = 10
    highpass_signal = Filters.butter_highpass_filter(
        lowpass_signal, cutoff_high, nyq, order, sampling_frequency, time, plot=False
    )

    spectrogram = librosa.feature.melspectrogram(
        y=highpass_signal,
        sr=fs,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        htk=htk,
        win_length=window_length,
        fmax=f_max,
    )
    spectrogram = spectrogram ** extra_power

    fig, ax = plt.subplots()
    img = librosa.display.specshow(
        spectrogram,
        ax=ax,
        sr=sampling_frequency,
        fmax=f_max,
        hop_length=hop_length,
        x_axis="s",
        y_axis="mel",
        htk=htk,
    )
    # plt.savefig("spec.png")
    return plt.gcf()
