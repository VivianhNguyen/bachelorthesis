import librosa
import numpy as np
import soundfile as sf
import time
import os
import sys
import shutil

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
Splits an audio file into segments of the specified duration

@param audio_path: the path to the audio file
@param segment_duration: the duration of each segment (in seconds)
@param output_directory: the directory to save the segments to
"""
def split_audio(audio_path, segment_duration, output_directory):
    # Load the audio file
    audio, sr = librosa.load(audio_path, sr=16000)

    # Calculate the number of samples for the desired segment duration
    segment_samples = int(segment_duration * sr)

    # Calculate the number of segments
    num_segments = len(audio) // segment_samples

    fileName = get_file_name(audio_path)

    # Split the audio file into segments
    for i in range(num_segments):
        # Calculate the start and end samples for the segment
        start_sample = i * segment_samples
        end_sample = (i + 1) * segment_samples

        # Extract the segment
        segment = audio[start_sample:end_sample]

        # Save the segment to a file
        segment_path = os.path.join(output_directory, f"{fileName}-{i+1}.wav")

        # write to file
        sf.write(segment_path, segment, 16000, format="wav", subtype='PCM_16')

    print(f"{num_segments} segments created in the '{output_directory}' directory.")

"""
Splits an audio file into segments of the specified duration
"""
def main():
    # Get command line arguments
    audioSize = get_item_or_none(sys.argv, 3)
    fileName = get_item_or_none(sys.argv, 1)
    folderName = get_item_or_none(sys.argv, 2)

    #simple checks for quality of life improvements
    if fileName == None:
        print("Specify a name please")
        sys.exit(0)

    if folderName == None:
        print("Specify a name folder please")
        sys.exit(0)

    if audioSize == None:
        print("Specify audio size please")
        sys.exit(0)

    folder_path = os.path.join(os.getcwd(), folderName)
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Remove the existing folder
        shutil.rmtree(folder_path)

    # Create the folder
    folder = os.mkdir(folder_path)


    split_audio(fileName, int(audioSize), folder_path)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
