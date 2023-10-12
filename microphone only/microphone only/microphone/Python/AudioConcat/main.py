import librosa
import numpy as np
import os
import soundfile as sf
import sys

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
Concatenates all audio files in a directory into a single audio file
"""
def main():
    # Get command line arguments
    path = get_item_or_none(sys.argv, 1)
    fileName = get_item_or_none(sys.argv, 2)
    sampleSize = get_item_or_none(sys.argv, 3)

    # simple checks for quality of life improvements
    if path == None:
        print("Specify a path please")
        sys.exit(0)
    if not os.path.exists(path):
        print("Path does not exist")
        sys.exit(0)

    if fileName == None:
        print("Specify a name please for the file")
        sys.exit(0)


    #get file names in directory
    file_list = []

    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            file_list.append(os.path.join(path, file))


    #check if you wanted more files then there are here
    if sampleSize is not None:
        sampleSize = int(sampleSize)
    else:
        sampleSize = len(file_list)

    if not isinstance(sampleSize, int):
        print("Argument specified is not an integer")
        sys.exit(0)

    if np.size(file_list) < sampleSize:
        print(f"You requested more samples then there are in this directory, there are  {np.size(file_list)}")
        sys.exit(0)


    #set the sampling rate to 11 025Hz
    combined_audio = []
    for file in file_list[:sampleSize]:
        audio, sr = librosa.load(file, sr= 16000)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=11025)
        combined_audio = np.concatenate((combined_audio, audio))

    # Change quantization to 16-bit
    #combined_audio = librosa.util.fix_bit_depth(combined_audio, bit_depth=16)

    #write to file
    print(combined_audio.shape)
    sf.write(fileName + '.wav', combined_audio, 11025, format="wav", subtype='PCM_16')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()