# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import librosa
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import soundfile as sf

"""
Returns the item at the index if it exists, otherwise returns None

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
Returns the file name without the extension

@param path: the path to the file
@return: the file name without the extension
"""
def get_file_name(path):
    file_name = os.path.basename(path)
    file_name_without_extension = os.path.splitext(file_name)[0]
    return file_name_without_extension

"""
 Processes a directory specified by the command line argument path.
 It retrieves the file names within the directory and iterates over each file.
 For each audio file, it loads the audio data using librosa, normalizes the audio by dividing it by
 the maximum absolute amplitude, and writes the normalized audio to a new file in the "NormalizedAudioFiles"
 directory with the same file name as the original file but appended with "-normalized.wav".
"""
def main():
    # Get command line arguments
    path = get_item_or_none(sys.argv, 1)

    # simple checks for quality of life improvements
    if path == None:
        print("Specify a path please")
        sys.exit(0)

    # get file names in directory
    file_list = []

    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            file_list.append(os.path.join(path, file))

    for audio_path in file_list:
        audio, sr = librosa.load(audio_path, sr=11025)

        max_amplitude = np.max(np.abs(audio))
        normalized_audio = audio / max_amplitude

        # write to file
        sf.write("..\\NormalizedAudioFiles\\"+ get_file_name(audio_path) + '-normalized.wav', normalized_audio, 11025, format="wav", subtype='PCM_16')



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
