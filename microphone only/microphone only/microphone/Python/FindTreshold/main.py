import os
import sys
import subprocess
import numpy as np
import librosa
import math
from MaximumEnergySlopeAlgorithm import getMinMax as getMinMaxSlope
from MaximumSoundEnergyAlgorithm import getMinMax as getMinMaxMaximumSound
from LowPassFilterAlgorithm import getMinMax as getMinMaxLowpass

paths = ["C:\\Users\\Gras\\cse3000-earable-computing\\Python\AudioCompressor\\apple-1-resample-compressed.wav",
         "C:\\Users\\Gras\\cse3000-earable-computing\\Python\AudioCompressor\\apple-2-resample-compressed.wav",
         "C:\\Users\\Gras\\cse3000-earable-computing\\Python\AudioCompressor\\carrot-1-resample-compressed.wav",
         "C:\\Users\\Gras\\cse3000-earable-computing\\Python\AudioCompressor\\carrot-2-resample-compressed.wav",
         "C:\\Users\\Gras\\cse3000-earable-computing\\Python\AudioCompressor\\chip-2-resample-compressed.wav",
         "C:\\Users\\Gras\\cse3000-earable-computing\\Python\AudioCompressor\\chip-3-resample-compressed.wav"]

scripts = ["C:\\Users\\Gras\\cse3000-earable-computing\\Python\\MaximumEnergySlopeAlgorithm\\main.py",
          "C:\\Users\\Gras\\cse3000-earable-computing\\Python\\MaximumSoundEnegyAlgorithm\\main.py",
          "C:\\Users\\Gras\\cse3000-earable-computing\\Python\\LowPassFilterAlgorithm\\main.py"]

file_names = ["MaximumEnergySlopeAlgorithm", "MaximumSoundEnegyAlgorithm", "LowPassFilterAlgorithm"]

#output file, declared globaly
output = None

"""
Runs a script and returns the output

@param script_path: the path to the script to run
@param args: the arguments to pass to the script
@return: the output of the script
"""
def run_script(script_path, args):
    command = ['python', script_path] + args
    result = subprocess.run(command, capture_output=True, text=True)
    # 'result.stdout' contains the printed output of the script
    # You can save it to a file or use it in any other way you need

    #f.write(result.stdout)
    return result.stdout

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
Loads the audio files and returns them

@return: the audio files and the number of chews
"""
def get_training_dataset():
    audios = []

    #check if they exist
    for path in paths:
        if not os.path.exists(path):
            print("Path does not exist")
            sys.exit(0)

        # get audio file
        audio, sr = librosa.load(path, sr=11025)
        audios.append(audio)


    #nr of chews labeled
    #nr_of_chews = [59]
    nr_of_chews = [98, 96, 100, 97, 91, 92]

    return audios, nr_of_chews

"""
Returns the output of the algorithm based on the given threshold and min_threshold

@param script_path: the path to the script to run
@param audio_path: the path to the audio file
@param audio: the audio file
@param target: the target number of chews
@param tolerance: the tolerance for the number of chews
@return: the final treshhold and chew count
"""
def treshold_finder(script_path, audio_path, audio, target, tolerance):
    # Your algorithm implementation
    # Return the output of the algorithm based on the given threshold and min_threshold

    # Set the desired output
    desired_output = target  # Specify the desired output of your algorithm

    # Set the initial range for threshold and min_threshold
    # Each algorithm has a specific threshold range, use the getMinMax function to get them from the audio
    threshold_min,threshold_max = getMinMaxSlope(audio_path)


    # Perform binary search for threshold and min_threshold
    threshold = threshold_min + (threshold_max - threshold_min) / 2

    arguments = [audio_path, str(threshold)]
    chew_count = int(run_script(script_path, arguments))
    output.write(f"Target: {target} Tollerance: {tolerance}\n")

    while abs(chew_count - desired_output) > tolerance and not math.isclose(threshold_max, threshold_min, rel_tol=0.0001, abs_tol=0.0001):
        output.write(f"{threshold} {chew_count}\n")
        if chew_count < desired_output:
            threshold_max = threshold
        else:
            threshold_min = threshold

        threshold = threshold_min + (threshold_max - threshold_min) / 2
        arguments = [audio_path, str(threshold)]

        chew_count = int(run_script(script_path, arguments))

    # The final values of threshold and min_threshold that produce the desired output
    final_threshold = threshold

    return final_threshold, chew_count

"""
Performs a search for the best threshold and tolerance values to detect chew events in an audio file.
It iterates over different combinations of threshold and tolerance values, evaluates the chew count, 
and saves the combination and result to a text file. Finally, it prints the best chew count and threshold values.
"""
def main():

    # Get command line arguments
    # path of the output file
    # path = get_item_or_none(sys.argv,1)
    # global output
    # output = open(path+".txt", 'a')
    #information about the script
    #audio_path = get_item_or_none(sys.argv,2)

    # simple checks for quality of life improvements
    # if path == None:
    #     print("Specify a path please")
    #     sys.exit(0)

    # simple checks for quality of life improvements
    # if audio_path == None:
    #     print("Specify a path for the audio please")
    #     sys.exit(0)

    # if not os.path.exists(path):
    #     print("Path does not exist")
    #     sys.exit(0)

    # if not os.path.exists(audio_path):
    #     print("Path does not exist")
    #     sys.exit(0)

    audios, ground_truth = get_training_dataset()

    for i in range(3):
        for j in range(len(audios)):

            script = scripts[i]
            algorithm = file_names[i]
            filename = os.path.splitext(os.path.basename(paths[j]))[0]

            print(script)
            print(algorithm)
            print(filename)

            #create the file with the apropriate algorithm and filename
            global output
            output = open(f"{algorithm}-{filename}.txt", 'a')

            best_count = 0
            best_treshold = 0
            best_tollerance = 0
    

            for k in range(ground_truth[j]//2 - 1, -1, -1):
                treshold, chew_count = treshold_finder(script, paths[j], audios[j], ground_truth[j], k)
                #get the best difference in chew count, save both the count and treshold that achieved that count
                if(abs(ground_truth[j]- best_count) > abs(ground_truth[j] - chew_count)):
                    best_count = chew_count
                    best_treshold = treshold
                    best_tollerance = k

            #dont forget to close the file
            output.write(f"\n\n{best_count} {best_treshold} {best_tollerance}\n")
            output.close()
            print(best_count)
            print(best_treshold)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

