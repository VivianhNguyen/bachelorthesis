<p align="center">
  <img src="TUDelft_logo_rgb.png" width="150">
  <br />
  <br />
</p>

--------------------------------------------------------------------------------

This repository consists of the tools used for analysing, processing and training of the audio used for writing the paper for CSE3000-Chewing Detection on Low Power Embedded System. An electronic version of this thesis is available at http://repository.tudelft.nl/



### Features:

* SerialReader - records data from an arduino 33 BLE Sense PDM microphone through a specified COM port and saves to a file name give as command argument
* AudioDecimator - splits an audio file into segments of the specified duration
* AudioConcat - concatenates all audio files in a directory into a single audio file
* AudioPlotter - plots the audio signal waveform
* FindThreshold - performs a search for the best threshold and tolerance values to detect chew events in an audio file.
It iterates over different combinations of threshold and tolerance values, evaluates the chew count, 
and saves the combination and result to a text file. Finally, it prints the best chew count and threshold values.
* NormalizeAudio - processes a directory specified by the command line argument path.
 It retrieves the file names within the directory and iterates over each file.
 For each audio file, it loads the audio data using librosa, normalizes the audio by dividing it by
 the maximum absolute amplitude, and writes the normalized audio to a new file in the "NormalizedAudioFiles"
 directory with the same file name as the original file but appended with "-normalized.wav".

* MaximumSoundEnergyAlgorithm - processes an audio file specified by the command line arguments.
It divides the audio into frames of length 256 and calculates the energy for each frame.
It then detects chew events based on the maximum energy values within a window of 12 frames and a threshold.
The code counts the chew events and prints the total count to the console.
The time in milliseconds of each chew event is also stored in the frames list.
* MaximumEnergySlopeAlgorithm - analyzes an audio file specified by the command line arguments.
It processes the audio in frames of length 256 and calculates the signal energy for each frame.
It then detects chew events based on changes in energy values (slopes) and a threshold.
The chew events are counted, and the total count is printed to the console.
* LowPassFilterAlgorithm - processes an audio file, applies a Butterworth filter, and detects chew events based on amplitude thresholds.
* MaximumSoundEnergyWithoutSpeech - processes an audio file specified by the command line arguments.
It divides the audio into frames of length 256 and calculates the energy for each frame.
It then detects chew events based on the maximum energy values within a window of 12 frames and a threshold.
The code counts the chew events and prints the total count to the console. The time in milliseconds of each chew event is stored in the frames list.
Additionally, it plots the audio signal waveform and marks the detected chew events with red dashed lines.


# Requirements and Installation

* Python version 3.11
* librosa 0.10.0.post2
* numpy 1.24.3
* soundfile 0.12.1
* matplotlib 3.7.1

* **To install requirements** and develop locally:

``` bash
git clone https://gitlab.tudelft.nl/vkpdsouza/cse3000-earable-computing.git
cd cse3000-earable-computing
pip install requirements.txt
```

# Usage
Here is a guide to reaching the same results:
* Upload the code from TestRealTime.ino to the Arduino Nano 33 BLE Sense
* Open the SerialReader python file and run it with the name and size of the recording as parameters
* Normalize the recording using the AudioNormalizer
* Manually count the chewing events in the audio; for ease, use the AudioDecimator and AudioConcat to separate the audio into smaller chunks making it easier to count.
* Apply a mu-law filter using the AudioCompressor to reduce dynamic range.
* Find overfitted minimum threshold using the FindThreshold program
* Plot the audio and find the RMS using AudioPlotter
* Test your audio against the different algorithms and minimum threshold values.

# Citation

Please cite as:

``` bibtex
@inproceedings{CSE3000-2023-p3Cutitei,
  title = {Automatic Chewing Detection on an Embedded Platform},
  author = {p3 Cutitie, Przemysław Pawełczak, Vivian Dsouza },
  booktitle = {Reasearch Project CSE3000-Automatic Chewing Detection},
  year = {2023},
}
```
