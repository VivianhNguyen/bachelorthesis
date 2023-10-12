# This is a sample Python script.
import numpy as np

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

thresholds = [11.55748151, 19.79555337, 3.531689784, 9.615999344, 6.240250865, 30.80079659]
rms = [0.18800406, 0.25034463, 0.10509852, 0.16899946, 0.12451721, 0.29877555]

def main():
    print(np.corrcoef(thresholds, rms)[0, 1])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
