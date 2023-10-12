import librosa
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


#thresholds = [7.554981932044029,13.131491057574749, 2.7914630299201235, 6.4703444426413625, 3.863683814648539]
"""
Returns the item at the specified index in the list, or None if the index is out of bounds

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
Returns the minimum and maximum energy slope of the audio file

@param path: the path to the audio file
@return: the minimum and maximum energy slope
"""
def getMinMax(path):

    # set frame length to 256
    frame_length = 256

    # get audio file
    audio, sr = librosa.load(path, sr=11025)

    # get absolute value for audio
    audio = np.abs(audio)

    # Use librosa.util.frame to frame the audio into frames, set the hop_length to the frame_length
    audio_chunks = librosa.util.frame(audio, frame_length=frame_length, hop_length=frame_length)

    # Calculate signal energy as the sum of the squared  signal values of each sample in a frame
    audio_chunks = np.transpose(audio_chunks)
    energy_list = np.sum(np.square(audio_chunks), axis=1)

    slopes = np.diff(energy_list, prepend=0)

    return np.min(slopes), np.max(slopes)

"""
Analyzes an audio file specified by the command line arguments.
It processes the audio in frames of length 256 and calculates the signal energy for each frame.
It then detects chew events based on changes in energy values (slopes) and a threshold.
The chew events are counted, and the total count is printed to the console.
"""
def main():
    # Get command line arguments
    path = get_item_or_none(sys.argv, 1)
    mintreshold = get_item_or_none(sys.argv, 2)

    # simple checks for quality of life improvements
    if path == None:
        print("Specify a path please")
        sys.exit(0)

    if mintreshold == None:
        print("Specify a minimum treshold please")
        sys.exit(0)

    if not os.path.exists(path):
        print("Path does not exist")
        sys.exit(0)

    # set frame length to 256
    frame_length = 256

    # get audio file
    audio, sr = librosa.load(path, sr=11025)

    # get absolute value for audio
    audio = np.abs(audio)

    # Use librosa.util.frame to frame the audio into frames, set the hop_length to the frame_length
    audio_chunks = librosa.util.frame(audio, frame_length=frame_length, hop_length=frame_length)

    # Calculate signal energy as the sum of the squared  signal values of each sample in a frame
    audio_chunks = np.transpose(audio_chunks)
    energy_list = np.sum(np.square(audio_chunks), axis=1)

    slopes = np.diff(energy_list, prepend=0)

    # set minimum treshold
    min_treshold = float(mintreshold)
   # min_treshold = np.sum(thresholds)/ len(thresholds)

    # set treshold
    treshold = 0

    from_last_max = 12

    chew_count = 0

    for i in range(1,len(slopes)-13):
        from_last_max += 1
        if from_last_max >= 12:
            if slopes[i] > slopes[i-1]:
                # only if the maximum energy value exceded the treshold value
                if slopes[i] > treshold:
                    # after the chew event was detected the treshold was set to the 12th subsequent, no lower than the minimum treshold
                    treshold = max(min(slopes[i + 13], min_treshold * 1.25), min_treshold)
                    chew_count += 1
                    from_last_max = 0

    print(chew_count)



if __name__ == '__main__':
    main()

