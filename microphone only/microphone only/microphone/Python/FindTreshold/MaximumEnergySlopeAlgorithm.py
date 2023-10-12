import librosa
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

"""
Returns the minimum and maximum value of the energy slopes

@param path: the path to the audio file
@return: the minimum and maximum value of the energy slopes
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
Prints the chew count of the energy slopes
"""
def main():
    # Get command line arguments
    path = sys.argv[1]
    min_thresh = sys.argv[2]

    # simple checks for quality of life improvements
    if path == None:
        print("Specify a path please")
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
    min_treshold = min_thresh

    # set treshold
    treshold = 0

    from_last_max = 12

    chew_count = 0

    for i in range(1,len(slopes)):
        from_last_max += 1
        if from_last_max >= 12:
            if slopes[i] > slopes[i-1]:
                # only if the maximum energy value exceded the treshold value
                if slopes[i] > treshold:
                    # after the chew event was detected the treshold was set to the 12th subsequent, no lower than the minimum treshold
                    treshold = max(np.min(energy_list[i + 13], min_treshold * 1.25), min_treshold)
                    chew_count += 1
                    from_last_max = 0

    print(chew_count)



if __name__ == '__main__':
    main()

