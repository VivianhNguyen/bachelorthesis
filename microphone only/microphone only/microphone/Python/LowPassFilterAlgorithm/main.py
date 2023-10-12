import librosa
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

thresholds = [ 0.31102143484167755,0.39422079222276807,0.15763295302167535,0.2232690748060122, 0.2764345044561196]

"""
Returns the item at the specified index if it exists, otherwise returns None

@param my_list: the list to check
@param index_to_check: the index to check
@return: the item at the index or None
"""
def get_item_or_none(my_list, index_to_check):
    if index_to_check < len(my_list):
        return my_list[index_to_check]
    else:
        return None
    

"""
Returns the minimum and maximum values of the filtered signal

@param path: the path to the audio file
@return: the minimum and maximum values of the filtered signal
"""
def getMinMax(path):
    # get audio file
    audio, sr = librosa.load(path, sr=11025)

    # rectify audio file
    rectified_audio = np.abs(audio)

    # Define the filter specifications
    order = 5
    cutoff_freq = 2  # Hz

    # Normalize the cutoff frequency
    normalized_cutoff = cutoff_freq / (sr / 2)

    # Compute the digital Butterworth filter coefficients
    b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False, output='ba')

    # Apply the filter to the audio signal
    filtered_signal = signal.lfilter(b, a, rectified_audio)

    return np.min(filtered_signal), np.max(filtered_signal)

"""
Processes an audio file, applies a Butterworth filter, and detects chew events based on amplitude thresholds.
"""
def main():
    # Get command line arguments
    path = get_item_or_none(sys.argv, 1)
    min_tresh = get_item_or_none(sys.argv, 2)

    # simple checks for quality of life improvements
    if path == None:
        print("Specify a path please")
        sys.exit(0)
    if not os.path.exists(path):
        print("Path does not exist")
        sys.exit(0)

    # simple checks for quality of life improvements
    if min_tresh == None:
        print("Specify a minimum treshold please")
        sys.exit(0)


    # get audio file
    audio, sr = librosa.load(path, sr=11025)

    #rectify audio file
    rectified_audio = np.abs(audio)

    # Define the filter specifications
    order = 5
    cutoff_freq = 2  # Hz

    # Normalize the cutoff frequency
    normalized_cutoff = cutoff_freq / (sr / 2)


    # Compute the digital Butterworth filter coefficients
    b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False, output='ba')

    # Apply the filter to the audio signal
    filtered_signal = signal.lfilter(b, a, rectified_audio)

    # set minimum treshold
    min_treshold = float(min_tresh)
    min_treshold = np.sum(thresholds)/ len(thresholds)

    # set treshold
    treshold = 0

    chew_count = 0

    from_last_max = 3072

    # frame counter for time stamps
    count_frame = 0
    frames = []

    # Check for maximum points based on the condition
    for i in range(len(filtered_signal) - 3072):
        from_last_max += 1
        count_frame += 1
        if from_last_max > 3072:
            if filtered_signal[i] > np.max(filtered_signal[i+1: i+3073]):
                if filtered_signal[i] > treshold:
                    # time in ms of chew event detection
                    frames.append(count_frame * (1/11_025))

                    treshold = max(min(filtered_signal[i+3072], min_treshold * 1.25), min_treshold)
                    chew_count += 1
                    from_last_max = 0


    print(chew_count)
    # # Plot the audio signal waveform
    # plt.figure(figsize=(12, 4))
    # librosa.display.waveshow(filtered_signal, sr=sr)
    #
    # # plot lines where we detect chewing
    # for x in frames:
    #     plt.axvline(x=x, color='red', linestyle='--')
    #
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.title('Audio Signal Waveform')
    # plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


