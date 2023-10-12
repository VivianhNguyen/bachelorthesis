from matplotlib import pyplot as plt
import numpy as np
from numpy import ogrid
from csvconverter import *

def plot_readings(ax,ay,az,gx,gy,gz, timeSeconds=None, filename=""):
    if timeSeconds!=None:
        transparency = 0.8
        plt.subplot(2,1,1)

        timeArrayA = range(len(ax)) #np.array(range(len(ax))).reshape(-1, 1).tolist()
        # print("1: ", timeArrayA)
        timeArrayA = np.divide(timeArrayA, (len(timeArrayA)))
        #print("2: ", timeArray)
        timeArrayA = np.multiply(timeArrayA, timeSeconds)

        timeArrayG = range(len(gx)) 
        timeArrayG = np.divide(timeArrayG, (len(timeArrayG)))
        timeArrayG = np.multiply(timeArrayG, timeSeconds)

        plt.plot(timeArrayA, ax, label='ax',alpha=transparency)
        plt.plot(timeArrayA, ay, label='ay',alpha=transparency)
        plt.plot( timeArrayA, az,label='az',alpha=transparency)
        plt.legend()
        plt.title('Acceleration - ' + filename)

        plt.subplot(2,1,2)
        plt.plot( timeArrayG, gx,label='gx',alpha=transparency)
        plt.plot( timeArrayG, gy,label='gy',alpha=transparency)
        plt.plot( timeArrayG, gz,label='gz',alpha=transparency)
        plt.legend()
        plt.title('Angular velocity')
        plt.xlabel('Time')
        plt.show()
    else:
        transparency = 0.8
        plt.subplot(2,1,1)
        plt.plot(ax,label='ax',alpha=transparency)
        plt.plot(ay,label='ay',alpha=transparency)
        plt.plot(az,label='az',alpha=transparency)
        plt.legend()
        plt.title('Acceleration - ' + filename)

        plt.subplot(2,1,2)
        plt.plot(gx,label='gx',alpha=transparency)
        plt.plot(gy,label='gy',alpha=transparency)
        plt.plot(gz,label='gz',alpha=transparency)
        plt.legend()
        plt.title('Angular velocity')
        plt.xlabel('Time')

        plt.show()




def plotGraph():
    # mat = readCSVFile("./1206/data/1ontbijtkoek1206.txt", (1, 2, 3, 4, 5, 6))
    # mat = readCSVFile("./1106/20hz/butteredAndNorm/BN1tosti1106_20hz.txt", (0, 1, 2, 3, 4, 5))
    # mat = readCSVFile("./1106/20hz/butteredAndNorm/BN1icecream1106_20hz.txt", (0, 1, 2, 3, 4, 5))
    # mat = mat[500:1200,:]

    # mat = readCSVFile("./1106/20hz/butteredAndNorm/BN1tosti1106_20hz.txt", range(0, 6))
    # mat = readCSVFile("./1106/20hz/butteredAndNorm/BN1tosti1106_20hz.txt", range(0, 6))

    # mat = readCSVFile("./1106/20hz/butteredAndNorm/BN1cookies1106_20hz.txt", range(0, 6))
    files1106 = [("1cookies", 90, 1), 
        ("1icecream", 90, 1),
        # ("1tosti", 3*60, 1),
        # ("1video", 6*60, 0 ),
        # ("1walking", 8*60, 0 ),
        ("2icecream", 90, 1 ),
        # ("2tosti", 3*60, 1),
        # ("3tosti", 2*60, 1),
        # ("4tosti", 1*60, 1)
        ]

    ## -----1206
    files1206 = [("1cucumber", 3 * 60, 1),
            ("1ontbijtkoek", 1 * 60, 1),
            ("1phone", 4 * 60, 0),
            # ("1sauzijs", 4 * 60, 1),
            # ("1studying", 7* 60, 0),
            ("2ontbijtkoek", 1 * 60, 1)]

    # ----- 1306

    files1306 = [
            # ("1brushingteeth", 2 * 60, 0),
            # ("1studying", 7 * 60, 0),
            # ("1tosti", 2.5 * 30, 1),
            ("1reading", 4 * 60, 0),
            ("1walkinginroom", 3 * 60, 0),
            ("2reading", 3 * 60, 0)]


    # ---- 2006

    files2006 = [("1acrackerp3", 2 * 60, 1),
                # ("1applep3", 2 * 60, 1),
                ("1crackerp3", 2 * 60, 1),
                ("1crackerp3", 2 * 60, 1),
                ("1cucumberp3", 2* 60, 1),
                ("1phonep3", 5 * 60, 0),
                ("1pindap3", 2 * 60, 1),
                ("1readingp3", 4 * 60, 0),
                ("1rijstwafelp3", 1.5 * 60, 1),
                ("1talkingp3", 4 * 60, 0),
                ("1videop3", 5 * 60, 0),
                ("1videop3", 7 * 60, 0),
                ("1walkingp3", 7 * 60, 0),
                ("1walkingp3", 5 * 60, 0),
                ("2talkingp3", 4 * 60, 0),
                ("2walkingp3", 4 * 60, 0)]
    
    filename = files1106[3][0] + "1106"
    

    mat = readCSVFile("./usedData/butteredAndNorm/BN" + filename + "_20hz.txt", range(0,6))[300:,:]

    # mat = mat[20*30:,: ]
    mat = mat[:,: ]
    ax = mat[:,0]
    ay = mat[:,1]
    az = mat[:,2]
    gx = mat[:,3]
    gy = mat[:,4]
    gz = mat[:,5]
    plt.title(filename)
    plot_readings(ax,ay,az,gx,gy,gz, filename=filename)

# plotGraph()