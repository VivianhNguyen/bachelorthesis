import serial
import time
from settings import *
from plotting import plot_readings

if __name__ == '__main__':


    serial_device = serial.Serial(port, baudrate, timeout=1)
    print('Connected')

    # accelerometer readings in x,y,z axes
    ax = []
    ay = []
    az = []
    # gyroscope readings in x,y,z axesard
    gx = []
    gy = []
    gz = []

    mic = []

    # clearing the file 
    if save_to_file:
        with open(filename,'w') as file:
            pass
    
    start = time.time()
    print('Sample start')
    while (time.time() - start) < sample_time:
        try:
            line  = (serial_device.read_until(b"\n")).decode("utf-8")
        except:
            #print('Skipping non ASCII character')
            continue
       #print(len(line.split(' ')))
        line_valid = (line[0] == 'i') and (len(line.split(' ')) == 8)

        if save_to_file and line_valid:
            with open('2ep42106.txt','a') as file:
                file.writelines(line)
        #7 MIN WALKING C
        # 4 MIN TLAKING C
        # 5 MIN VIDEO C
        # 2 MIN EAT appple
        # 2 MIN CRACKER C

        # 7 MIN VIDEO G
        # 1 CRACKER 2 MIN EACH G
        # 1 WALKING 5 MIN G
        # 2 WALKING 4 G
        # CUCMBER 2 MIN
        # RIp4STWAFEL 1.5

        #pinda 2?
        #phone 4?
        #reading2 
        #reading 2

        #ik rijstwafel 4 min
        # lopen 5 min
        # studying 4 min
        # hardopvoorlezen 4 min
        #phone 4 min

        if line_valid:
            a = line.split(' ')
            ax.append(int(a[1]))
            ay.append(int(a[2]))
            az.append(int(a[3]))
            #az.append(0)
            gx.append(int(a[4]))
            gy.append(int(a[5]))
            gz.append(int(a[6])) 

            # mic.append(int(a[7]))

    print('Sampling end')

    plot_readings(ax,ay,az,gx,gy,gz)
