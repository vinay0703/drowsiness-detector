#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

fig=plt.figure()
ax1=fig.add_subplot(111)
def refreshGroupData():
    print("refreshing data")
    graphdata=open("example.txt","r").read()
    lines=graphdata.split("\n")
    x_val=[]
    y_val=[]
    for line in lines:
        if len(line) >1:
            x,y=line.split(",")
            x_val.append(x)
            y_val.append(y)
    ax1.clear()
    ax1.plot(x_val,y_val)
ani=animation.FuncAnimation(fig,refreshGroupData(),interval=1000)
plt.show()