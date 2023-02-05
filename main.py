# My name is Anthony O'Neal and this is my work
# Runge-Kutta-Fehlberg for ODE

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.integrate import odeint

start = time.time()

#track amount of computational steps
compSteps=0
#initial values for x0, y0, and step size
x=2
y=1
h=0.3
xlist=[2]
ylist=[1]
print("Initial values: ")
print("x0=2\n" + "y0=1 \n" + "h=0.3 \n")

print("Iterating through RFK method...")

#iterate through RFK
for i in range(1000):

    #will be used to store incoming xn value from k4
    list=[]

    #ODE
    def dydx(x,y):
        return np.log(x)-y

    #solve for k values
    def k1(x,y,h):
        return dydx(x,y)
    k1=k1(x,y,h)
    compSteps +=1

    def k2(x,y,h):
        a = x+(h/2)
        b=y+(h/2)*k1
        return dydx(a,b)
    k2=k2(x,y,h)
    compSteps +=1

    def k3(x,y,h):
        a = x+(h/2)
        b=y+(h/2)*k2
        return dydx(a,b)
    k3=k3(x,y,h)
    compSteps +=1

    def k4(x,y,h):
        a=x+h
        b=y+h*k3
        return dydx(a,b),a


    #store next x value in list
    list = k4(x,y,h)
    k4=list[0]
    compSteps +=1

    #calculate T4
    T4=(k1+2*k2+2*k3+k4)/6
    compSteps +=1

    #next y value
    y=y+h*T4

    #next x value
    x=list[1]

    #add xn and yn to list of resulting x and y values
    xlist.append(x)
    ylist.append(y)

    #display k values
    print("k1 is " + str(k1))
    print("k2 is " + str(k2))
    print("k3 is " + str(k3))
    print("k4 is " + str(k4))


    #display resulting xn and yn values
    print("x" + str(i+1) + "=" + str(x))
    print("y" + str(i+1) + "=" + str(y))
    print("RFK iteration " + str(i+1))
    print("\n")



#print number of computational steps
print("Computational steps: "+ str(compSteps))

#RFK plot
plt.title("RFK plot")
plt.xlabel("xn values")
plt.ylabel("yn values")
plt.plot(xlist, ylist)
plt.show()


#display plot using odeint
ys = odeint(dydx, 1, xlist,tfirst=True)
ys = np.array(ys).flatten()
plt.title("ODEINT plot")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(xlist, ys)
plt.show()

#plot comparing RFK method and ODEint
plt.title("RFK vs. ODEINT")
plt.xlabel("xn values")
plt.ylabel("yn values")
plt.plot(xlist, ys)
plt.plot(xlist,ylist, linestyle='dashed')
plt.show()


#error plot
xlist = np.array(xlist).flatten()
y_exact = xlist - 1 + 2*np.exp(-xlist)
y_difference = ys - y_exact
y_diff = np.abs(y_exact - ys)
plt.semilogy(xlist, y_diff)
plt.ylabel("Error")
plt.xlabel("x")
plt.title("Error in numerical integration");
plt.show()

#Display computing time
end = time.time()
compTime = end-start
print("Computing time: " + str(compTime))

