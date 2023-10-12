##############
## Script listens to serial port and writes contents into a file
##############
## requires pySerial and librosa to be installed
import serial  # sudo pip install pyserial should work
import librosa
import numpy as np
import soundfile as sf
import time
import os
import sys

"""
Gets an item from the list if it exists, otherwise returns None

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
Reads audio data from a serial port and saves it to a WAV file. It takes two command line arguments: fileName,
which specifies the name of the output file, and audioSize, which
determines the duration of the audio recordingin seconds.

The code initializes a serial connection with the specified serial port and baud rate.
It then reads data from the serial port in chunks of 512 bytes and appends it to the data list.
The reading continues until the specified audioSize duration is reached.

Afterward, the byte array data is transformed into signed 16-bit integer arrays using NumPy.
Finally, the audio data is written to a WAV file with the specified fileName and a sample rate of 16000 Hz.
"""
def main():

    # Get command line arguments
    audioSize = get_item_or_none(sys.argv, 2)
    fileName = get_item_or_none(sys.argv, 1)

    #simple checks for quality of life improvements
    if fileName == None:
        print("Specify a name please")
        sys.exit(0)


    serial_port = "COM4"
    baud_rate = 500000  # In arduino, Serial.begin(baud_rate)

    ser = serial.Serial(serial_port, baud_rate)
    data = []

    t_end = time.time() + int(audioSize)
    while time.time() < t_end:
        if ser.in_waiting > 0:
            data.append(ser.read(512))

    # transform byte array to signed 16-bit integer arrays
    audio = np.frombuffer(np.array(data).flatten(), dtype=np.int16)

    # write to file
    sf.write(fileName + ".wav", audio, 16000, format="wav", subtype='PCM_16')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()