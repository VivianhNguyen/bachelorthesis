import librosa
import numpy as np
import os
import soundfile as sf
import sys
import matplotlib.pyplot as plt

"""
Returns an item from the list if it exists, otherwise returns None

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
Plots the audio signal waveform
"""
def main():
    # Get command line arguments
    path = get_item_or_none(sys.argv, 1)


    #simple checks for quality of life improvements
    if path == None:
        print("Specify a path please")
        sys.exit(0)
    if not os.path.exists(path):
        print("Path does not exist")
        sys.exit(0)

    audio, sr = librosa.load(path, sr= 11025)


      

    # Get the mean RMS value (single value for the entire audio)
    mean_rms = rms.mean()
    print(mean_rms)

    # Plot the audio signal waveform
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(audio, sr=sr)

    # plot line of rms
    plt.axhline(mean_rms, color='red', linestyle='--')

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Audio Signal Waveform')
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()




