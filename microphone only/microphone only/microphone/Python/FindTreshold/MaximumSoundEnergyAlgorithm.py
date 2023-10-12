import librosa
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

"""
Returns the minimum and maximum value of the energy list

@param path: the path to the audio file
@return: the minimum and maximum value of the energy list
"""
def getMinMax(path):
    #set frame length to 256
    frame_length = 256

    #get audio file
    audio, sr = librosa.load(path, sr= 11025)

    # Use librosa.util.frame to frame the audio into frames, set the hop_length to the frame_length
    audio_chunks = librosa.util.frame(audio, frame_length=frame_length, hop_length=frame_length)

    # Calculate signal energy as the sum of the squared  signal values of each sample in a frame
    audio_chunks = np.transpose(audio_chunks)
    energy_list = np.sum(np.square(audio_chunks), axis=1)

    return np.min(energy_list), np.max(energy_list)

"""
Prints the chew count of the energy list
"""
def main():
    # Get command line arguments
    path = sys.argv[1]
    mintreshold = sys.argv[2]

    #simple checks for quality of life improvements
    if path == None:
        print("Specify a path please")
        sys.exit(0)

    if mintreshold == None:
        print("Specify a minimum treshold please")
        sys.exit(0)


    if not os.path.exists(path):
        print("Path does not exist")
        sys.exit(0)


    #set frame length to 256
    frame_length = 256

    #get audio file
    audio, sr = librosa.load(path, sr= 11025)

    # Use librosa.util.frame to frame the audio into frames, set the hop_length to the frame_length
    audio_chunks = librosa.util.frame(audio, frame_length=frame_length, hop_length=frame_length)

    # Calculate signal energy as the sum of the squared  signal values of each sample in a frame
    audio_chunks = np.transpose(audio_chunks)
    energy_list = np.sum(np.square(audio_chunks), axis=1)

    #frame counter for time stamps
    count_frame = 0
    frames = []

    #set minimum treshold
    min_treshold = float(mintreshold)

    #set tresholdwow
    treshold = 0

    frames_since_max = 12

    chew_count = 0

    # Check for maximum points based on the condition
    for i in range(len(energy_list) - 12):
        frames_since_max += 1
        count_frame += 1

        if frames_since_max >= 12:
            if energy_list[i] > np.max(energy_list[i+1:i+13]):
                #only if the maximum energy value exceded the treshold value
                if energy_list[i] > treshold:

                    #after the chew event was detected the treshold was set to the 12th subsequent, no lower than the minimum treshold
                    treshold = max(np.min(energy_list[i+13], min_treshold * 1.25), min_treshold)

                    #time in ms of chew event detection
                    frames.append(count_frame * 23)
                    chew_count += 1
                    frames_since_max = 0

    print(chew_count)

    # #Plot the audio signal waveform
    # plt.figure(figsize=(12, 4))
    # librosa.display.waveshow(audio, sr=sr)
    #
    # #plot lines where we detect chewing
    # for x in frames:
    #     plt.axvline(x=x/1000, color='red', linestyle='--')
    #
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.title('Audio Signal Waveform')
    # plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()