import librosa
import os
import sys
import numpy as np
import matplotlib.pyplot as plt



# Define the frequency bands
frequency_bands = [
    (0, 200),
    (200, 400),
    (400, 800),
    (800, 1600),
    (1600, 3200),
    (3200, 5512)
]

# set minimum treshold
min_treshold = [0, 0, 0, 0, 0, 0]

# set treshold
treshold = [0, 0, 0, 0, 0, 0]

# set frame length to 64
frame_length = 64

"""
Updates the treshold for the given indexes

@param indexes: the indexes to update the treshold for
@param ASV: the sum of absolute spectral values
"""
def updateTreshold(indexes, ASV):
    for i in indexes:
        treshold[i] = np.max(ASV[i], min_treshold[i])

"""
Returns the indexes of the bands that have more than 2 maxima in the given range

@param index: the index to check
@param sumASV: the sum of absolute spectral values
"""
def hasMoreThan2MaximaInRange(index, sumASV):
    indexes = []
    for i in range(6):
        #this gets the 7 before index and 6 after
        if sumASV[index, i] > np.max(np.concatenate((sumASV[index-8 : index, i], sumASV[index+1 : index+7, i]))):
            if sumASV[index, i] > treshold[i]:
                indexes.append(i)

    #retrun array of indexes where the value is bigger
    return indexes

"""
Returns the sum of absolute spectral values for the given frame

@param frame: the frame to get the sum of absolute spectral values for
@return the sum of absolute spectral values
"""
def sumOfAbsoluteSpectralValues(frame):
    # Compute the STFT
    stft = librosa.stft(frame)

    # Compute the magnitude spectrogram
    spectrogram = np.abs(stft) ** 2

    sums = np.array([])
    for band in frequency_bands:
        freq_range = librosa.core.fft_frequencies(sr=11025, n_fft=stft.shape[0])
        band_indices = np.where(np.logical_and(freq_range >= band[0], freq_range <= band[1]))[0]
        band_sum = np.sum(spectrogram[band_indices, :])
        sums = np.append(sums, band_sum)

    return sums

"""
Processes an audio file specified by the command line argument path.
It divides the audio into frames and calculates the sum of absolute spectral values for each frame.
It then analyzes each set of 14 frames, looking for patterns of more than 2 maxima within the range.
If such patterns are found, it increments the chewCount variable and updates the threshold.
The code finally prints the total number of chew events detected.
"""
def main():

    chewCount = 0

    # Get command line arguments
    path = sys.argv[1]

    # simple checks for quality of life improvements
    if path == None:
        print("Specify a path please")
        sys.exit(0)
    if not os.path.exists(path):
        print("Path does not exist")
        sys.exit(0)


    # get audio file
    audio, sr = librosa.load(path, sr=11025)

    # Use librosa.util.frame to frame the audio into frames, set the hop_length to the frame_length
    audio_chunks = librosa.util.frame(audio, frame_length=frame_length, hop_length=frame_length)
    audio_chunks = np.transpose(audio_chunks)

    #sumOfAbsoluteSpectralValues
    sumASV = np.array([0,0,0,0,0,0])

    #every frame
    for frame in audio_chunks:
        sumASV = np.vstack((sumASV, sumOfAbsoluteSpectralValues(frame)))
    #remove first row of zeros
    sumASV = sumASV[1:]

    from_last_max = 48

    #look into 14 frames at a time, 7 before 6 after
    for i in range(8, len(sumASV) - 6):
        from_last_max += 1
        indexes = hasMoreThan2MaximaInRange(i, sumASV)
        if from_last_max >= 48:
            if(np.size(indexes) >= 2):
                chewCount += 1
                updateTreshold(indexes, sumASV[i])
                from_last_max = 0

    print(chewCount)


if __name__ == "__main__":
    main()
