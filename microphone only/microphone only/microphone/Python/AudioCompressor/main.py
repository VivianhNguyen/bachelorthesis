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
Returns the file name without the extension

@param path: the path to the file
@return: the file name without the extension
"""
def get_file_name(path):
    file_name = os.path.basename(path)
    file_name_without_extension = os.path.splitext(os.path.splitext(file_name)[0])[0]
    return file_name_without_extension

"""
Compresses the audio using mu-law compression
"""
def main():
    # Get command line arguments
    path = get_item_or_none(sys.argv, 1)

    # simple checks for quality of life improvements
    if path == None:
        print("Specify a path please")
        sys.exit(0)

    audio, sr = librosa.load(path, sr=11025)

    compressed_audio = librosa.mu_compress(audio,mu=63, quantize=False)

    sf.write( get_file_name(path) + '-compressed.wav', compressed_audio, 11025,
             format="wav", subtype='PCM_16')


# Press the green button in the gutter to run the script.
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/