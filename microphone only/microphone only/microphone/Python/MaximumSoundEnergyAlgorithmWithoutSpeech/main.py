import librosa
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

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
Returns the minimum and maximum energy values of the audio file

@param path: the path to the audio file
@return: the minimum and maximum energy value
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
Processes an audio file specified by the command line arguments.
It divides the audio into frames of length 256 and calculates the energy for each frame.
It then detects chew events based on the maximum energy values within a window of 12 frames and a threshold.
The code counts the chew events and prints the total count to the console. The time in milliseconds of each chew event is stored in the frames list.
Additionally, it plots the audio signal waveform and marks the detected chew events with red dashed lines.
"""
def main():
    # Get command line arguments
    path = get_item_or_none(sys.argv, 1)
    mintreshold = get_item_or_none(sys.argv, 2)

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

    #set treshold
    treshold = 0


    #avg energy 12 steps ahead
    avg_energy = []

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
                    if np.sum(energy_list[i+5:i + 10])*10 < 7:

                        #avg_energy.append(np.sum(energy_list[i:i + 12]))
                        #after the chew event was detected the treshold was set to the 12th subsequent, no lower than the minimum treshold
                        treshold = max(min(energy_list[i+13], min_treshold * 1.25), min_treshold)

                        #time in ms of chew event detection
                        frames.append(count_frame * 23)
                        chew_count += 1
                        frames_since_max = 0

    print(chew_count)
    # print(np.max(avg_energy))
    # print(np.sum(avg_energy) / len(avg_energy))
    # print(sorted(avg_energy)[len(avg_energy)//2])
    # print(np.std(avg_energy))

    #Plot the audio signal waveform
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(audio, sr=sr)

    #plot lines where we detect chewing
    for x in frames:
        plt.axvline(x=x/1000, color='red', linestyle='--')

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Audio Signal Waveform')
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()