#! /usr/bin/python3.5

from math import radians as rad
from math import degrees as deg
from math import *
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import os
from os import system as cmd
import random
import stats as st
import sys
import time

#plt.ion() # enables interactive mode
global x,yrange
x = np.arange(-10,10,.001)
yrange=10

## Setting up the 3d plotter
"""fig = plt.figure(figsize=(50,50))
ax = fig.add_subplot(111, projection="3d")
u = np.linspace(0, 2*pi, 100)
v = np.linspace(0, pi,100)

x = 10*np.outer(np.cos(u), np.sin(v))
y = 10*np.outer(np.sin(u), np.sin(v))
z = 10*np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x,y,z,cmap=cm.Blues)
plt.show() """

def cls():
    if sys.platform=="win32":
        cmd("cls")
    elif sys.platform=="linux":
        cmd("clear")
#### Polar ####
def rectangular(p):
	return (p[0]*cos(p[1]),p[0]*sin(p[1]))	
def polar(r):
	aVal=np.array(r)
	hypo = hypot(aVal[0],aVal[1])
	return (hypo,acos(aVal[0]/hypo))
#### Vectors #### 	
def spaceVectors(v,w):
	if (v.find("-") is -1 and v.find("+") is -1) and (w.find("-") is -1 and w.find("+") is -1):
		print("Not computable.")
		raise ValueError("No operators found.")
def plotVector(v): ## Only works for 2D vectors
	plt.arrow(0,0,v[0],v[1])
def dot(v,w):
	componentCount=len(v)
	currentSum=0
	for i in range(componentCount):
		currentSum+=v[i]*w[i]
	return currentSum
def angle(u,v):
	return acos(dot(u,v)/magnitude(u)*magnitude(v))
def magnitude(v,clean=False):
	total = np.array(np.array(v)**2).sum()
	if clean:
		return str(rRad(total)) + " or " + str(np.sqrt(total))
	else:
		return np.sqrt(total)
def unitVector(v):
	return str(v) + "/" + rRad(np.array(np.array(v)**2).sum())
def dAngles(v):
	mag = magnitude(v)
	angles=[]
	for i in v:
		angles.append(acos(i/mag))
	return str(np.array(angles)) + " or degrees(" + str(np.degrees(np.array(angles)))+")"
def cross(v,u,clean=False):
	a=v[1]*u[2]-u[1]*v[2]
	b=v[0]*u[2]-u[0]*v[2]
	c=v[0]*u[1]-u[0]*v[1]
	if clean:
		return "("+str(a)+")i-("+str(b)+")j+("+str(c)+")k"
	else:
		return (a,-b,c)
def decompose(v,w):
	numer=dot(v,w)
	scalar=numer/magnitude(w)**2
	v1=scalar*np.array(w)
	v2=np.array(v)-v1
	return "v1: " + str(v1) + "\nv2: " + str(v2)
### Matrix Algebra ###

def det(matrix):
	if len(matrix) is not 2:
		raise ValueError("god")
	return matrix[0][0]*matrix[1][1]-matrix[1][0]*matrix[0][1]
### Simplification and Exact Values ###
def simpfrac(numer, denom):
	
	if numer > denom:
		for i in range(int(abs(denom)),0,-1):
			if numer % i == 0 and denom % i == 0:
				denom /= i
				numer /= i
				if denom == 1:
					return numer
	elif numer < denom:
		for i in range(int(abs(numer)),0,-1):
			if numer % i == 0 and denom % i == 0:
				denom /= i
				numer /= i
				if denom == 1:
					return numer
	else:
		return 1
	return str(int(numer)) + "/" + str(int(denom))
def rRad(radical):
	factors = []
	for i in range(radical//2,1,-1):
		if radical % i == 0 and sqrt(i) % 1 == 0:
			factors.append(sqrt(i))
			radical /= i
	reduced_integer = 1
	for i in range(len(factors)):
		if factors[i] is not 0:
			reduced_integer *= factors[i]
	return str(reduced_integer) + "*sqrt(" + str(radical) + ")"
	
### Testing the parity of a function ###
def parity(func):

	table = {}
	
	err = "Error at x = "
	try:
		for i in range(-10,11,1):
			table.__setitem__(i, func(i))
	except ValueError:
		print(err + str(i))
	except ZeroDivisionError:
		print(err + str(i))

	if -func(100) == func(-100):
		return "Odd"
	elif func(-100) == func(100):
		return "Even"
	else:
		return "Neither"
		
def sec(x):
	return 1/np.cos(x)
def csc(x):
	return 1/np.sin(x)
def cot(x):
	return np.cos(x)/np.sin(x)
	
def DMS(val, toMin = True,minut = None, sec = None): # Takes the entire value in degrees and converts it to minutes and seconds
	
	if minut is not None or sec is not None:
		toMin = False
		return round(val + minut/60 + sec/3600,2)
	if toMin:	
		curr_val = float("." + str(val).split(".")[1])
		minutes,seconds = str(curr_val * 60).split(".")
		seconds = "." + seconds
		print(seconds)
		seconds = str(round(float(seconds)*60))
		
		return str(int(val)) + "\u00b0" + minutes + "'" + seconds + '"'
		
#### Plotting and graphing
def getMesh(x=(-10,10),y=(-10,10),resolution=200):
    return np.meshgrid(np.linspace(x[0],x[1],resolution),np.linspace(y[0],y[1],resolution))
def dash():
	style.use("dark_background")
	plt.axhline(ls = 'solid')
	plt.axvline(ls = 'solid')
	plt.xlabel("x")
	plt.ylabel("y")
	plt.grid(visible=True)
	plt.ylim(-yrange,yrange)
	
def implicit():
    dash()
    plt.contour(x,y,F-G,[1])
    plt.show()

def graph(title="",length=5):
	x,y = np.meshgrid(np.linspace(-length,length,500),np.linspace(-length,length,500))
	fig = plt.figure(figsize=(150,150))
	ax = fig.add_subplot(111,projection="3d")
	ax.set_title(title)
	ax.set_zlabel("Inches Long")
	ax.set_xlabel("Girth")
	ax.set_ylabel("Girth")
	ax.plot_wireframe(x,y,f(x,y))
	plt.show()
	
def summation(k,n,func):
	sum = 0
	
	for i in range(k, n+1,1):
		sum += func(i)
	return sum
def tangentize(f):
    dash()
    inc=-5
    plt.plot(x,f(x))
    while inc < 5:
        plt.plot(x,deriv(f,inc)*(x - inc) + f(inc))
        inc += .01
    plt.show()
def deriv(f,x,order=1): # actually returns the approximate slope at a specific point
    h = .000001
    if order is 1:
        return (f(x+h)-f(x))/h
    return deriv(lambda val : deriv(f,val,order=1),x,order-1)
def partial(f,*args):
    partials = []
    def resetArgs(args):
            newAr = []
            for i, val in enumerate(args[0]):
                newAr.append(val)
            return newAr
    for i, x in enumerate(args[0]):
        newArgs = resetArgs(args)
        newArgs[i] += .000001
        partials.append((f(newArgs)-f(args[0]))/.000001)
    return partials
def cylinder(f,a,b):    # finding the volume with cylindrical shells
    def newfunc(x):
        return 2*pi*x*f(x)
    return Riemann(newfunc,a,b,2000)
def washer(f,a,b):
    def newfunc(x):
        return pi*f(x)**2
    return Riemann(newfunc,a,b,2000)
def integral(f,a,b):
	total = 0
	limit = 10000000*(b-a)
	delta_x = (b-a)/(limit)
	current_val = a
	for i in range(limit):
		total += f(current_val)*delta_x
		current_val += delta_x
	print(current_val)
	return total
def Newtonian(f,x,val,iterations=10):             # Select an x that is slightly greater than the suspected value
    for i in range(iterations):
        if np.abs(f(x) - val)> .000001:
            x = x - f(x)/deriv(f,x)
    else:
        return x
    return x
def Riemann(f,a,b,n,side="mid"):
	delta_x = (b-a)/n
	total=0
	if side=="left":
		current_x = a
	elif side=="right":
		current_x = a + delta_x
	elif side=="mid":
		current_x = (a + a + delta_x)/2
	for i in range(n):
		total += delta_x*f(current_x)
		current_x += delta_x
	return total
def trapezoid(f,a,b,n):
    delta_x = (b-a)/n
    return (1/2)*(Riemann(f,a,b,n,side='left') + Riemann(f,a,b,n,side='right'))
def simpson(f,a,b,n):
    if not n % 2 == 0:
        raise ValueError("n must be an even number.")
    current_x = a
    delta_x = (b-a)/n
    total = f(a)
    current_x += delta_x
    for i in range(1,n,1):
        if i % 2 == 1:
            total += 4*f(current_x)
        else:
            total += 2*f(current_x)
        current_x += delta_x
    total += f(current_x)
    total *= (delta_x/3)
    return total
def error(f,a,b,n,tms):
    pass
def favg(f,a,b):
    return (1/(b-a))*Riemann(f,a,b,2000)
def fact(x):
	product = 1
	for i in range(1,x+1,1):
		product *=i
	return product
def bino(n,j):
	return fact(n)/(fact(j)*fact(n - j))
def expansion(x, a, deg):
	outputStr = ""
	for i in range(deg+1):
		if deg - i == 0:
			outputStr += str(bino(deg,i)*a**i*x**(deg - i))
		else:
			outputStr += str(bino(deg,i)*a**i*x**(deg - i)) + "x^"+str(deg-i)+"+ "
	return outputStr
	
def aroc(x1,x2, func):
	m = (func(x2) - func(x1))/(x2-x1)
	return m

def plotter(arr, func):
	newar = []
	for i in range(len(arr)):
		newar.append(func(arr[i]))
	return np.array(newar)
	
def divis(val):
	for i in range(1, val//2):
		if val % i == 0:
			print (str(i) + "\t:\t" + str(val / i))
			
def idk(x1,x2,func):
	m = aroc(x1,x2,func)
	
	y1 = func(x1)
	y2 = func(x2)
	
	equ = "y - " + str(y1) + " = " + str(m) + "(x - " + str(x1) + ")"
	return equ
def quadratic(a,b,c):
	try:
		sqrt(b**2-4*a*c)
	except ValueError:
		return "No solution."
	return "("+str(-b)+"\u00b1"+rRad(b**2-4*a*c)+")/"+str(2*a)
#### Derive a function based on the data points ((Incomplete)) ####
def getfunc():

	val = []
	a = b = c = None
	funcstr = input("Enter function: ")
	for i in range(len(funcstr)):
		try:
			val.append(int(funcstr[i]))
			if a == None:
				a = val[i]
			if b == None:
				b = val[i]
			if c == None:
				c = val[i]
		except ValueError:
			val.append(funcstr[i])

def vertex(a,b,c,f):
	h = -b/(2*a)
	k = f(h)
	return "(" + str(h) + ", " + str(k	) + ")"
	
def dist(p1,p2,retSqrt=False):
	total=np.array(np.array(np.array(p1)-np.array(p2))**2).sum()
	if retSqrt:
		return str(rRad(total)) + " or " + str(np.sqrt(total))
	else:
		return np.sqrt(total)
def triangle(a,b,c):
	leg1=dist(a,b)
	leg2=dist(a,c)
	leg3=dist(b,c)
	sArr=[leg1,leg2,leg3]
	hypIndex=sArr.index(max(leg1,leg2,leg3))
	
	for i in range(len(sArr)):
		if i is hypIndex:
			print("Hypotenuse: " + str(sArr[i]))
			
		else:
			print("Leg " + str(i) + ": " + str(sArr[i]))
	return sArr
def midpoint(a,b):
	a,b=np.array(a),np.array(b)
	return (a+b)/2
def fibbonaci(length):
	fib = [0,1]
	for i in range(1, length):
		fib.append(fib[i]+fib[i-1])
	return fib
def pythag(leg1 = None, leg2 = None, hypo = None):
	if leg1 == None:
		return "sqrt(" + str(hypo**2 - leg2**2) + ")"
	elif leg2 == None:
		return "sqrt(" + str(hypo**2 - leg1**2) + ")"
	else:
		return "sqrt(" + str(leg1**2 + leg2**2) + ")"
def quadrant(ang):
	if cos(ang) < 0 and sin(ang) < 0:
		return True
def lawOfCosines(sas=None,sss=None): ## Takes a side, angle (radians), and a side in an array, then returns the remaining values in an array
	if sas is not None:
		return np.sqrt(sas[0]**2+sas[2]**2-2*sas[0]*sas[2]*cos(sas[1]))
	else:
		return acos((sss[1]**2+sss[2]**2-sss[0]**2)/(2*sss[1]*sss[2]))
	sas[0]
def hero(a,b,c):
	s=(a+b+c)/2
	return sqrt(s*(s-a)*(s-b)*(s-c))
def timer(val, func = None, fVal = None, compute = None):
	tm_val = time.time() + val
	while tm_val - time.time() > .00001:
		if compute is None:
			pass
		else:
			compute()
	if func is not None and fVal is not None:
		return func(fVal)
	elif func is not None and fVal is None:
		return func()
		
##################################################################################################
### Function for finding whether two given sides and an angle form two, one, or zero triangles ###
def SSA(side1,side2,angle1):
	try:
		angle2i=asin(side2*sin(angle1)/side1)
	except ValueError:
		print("No solution.")
	if pi-angle2i+angle1 < pi:
		angle3i=pi-(angle2i+angle1)
		side3i=sin(angle3i)*(side1/sin(angle1))
	
		angle2j=pi-angle2i
		angle3j=pi-(angle1+angle2j)
		tri1=[side1,angle1,side2,angle2i,side3i,angle3i]
		tri2=[side1,angle1,side2,angle2j,side3j,angle3j]
		triDisp(tri1)
		triDisp(tri2)
		return [tri1,tri2]
		
def SAS(side1,angle3,side2):
	side3=np.sqrt(side1**2+side2**2-2*side1*side2*np.cos(angle3))
	angle2=acos((side1**2+side3**2-side2**2)/(2*side1*side3))
	angle1=pi-(angle2+angle3)
	tri=[side1,angle1,side2,angle2,side3,angle3]
	triDist(tri)
	return tri
def triDisp(tri):
	output=str(tri[0]) + "\n" + str(tri[1]) + "\n\n" + str(tri[2]) + "\n" + str(tri[3]) + "\n\n" + str(tri[4]) + "\n" + str(tri[5]) + "\n\n\n\n"
	return output
##################################################################################################
class Timer:
	def __init__(self, val, func = None, fVal = None, countdown = False):
		self._val = val
		self._func = func
		self._fVal = fVal
		self._countdown = countdown
	def __call__(self):
		return "asdf"
	def start(self):
		tm_val = self._val + time.time()
		while tm_val - time.time() > .00001:
			pass
		if self._func is not None and self._fVal is not None:
			return self.callback(self._fVal)
		elif self._func is not None and self._fVal is None:
			return self.callback()
	def callback(self,val = None):
		if val is not None:
			return self._func(val)
		else:
			return self._func()
	def setFunction(self,func):
		self._func = func
	def setVal(self,val):
		self._val = val
	def getVal(self):
		return self._val
	def setFVal(self, val):
		self._fVal = val
	def getFVal(self):
		return self._fVal
	def setCountdown(self,isTrue):
		self._countdown = isTrue
	def getCountdown(self):
		return self._countdown
		
class Polar_Equation:
	def __init__(self, start = 0, end = 2*pi, interval = .0001, r = 1):
		self.__func = None
		self.__theta = np.arange(start, end, interval)
		if self.__isFunc(r):
			self.__func = r
			self.__r = [0.0]*len(self.__theta)
			for i in range(len(self.__theta)):
				self.__r[i] = self.__func(self.__theta[i])
		else:
			self.__r = [r]*len(self.__theta)
	def __isFunc(self, r):
		try:
			r(0)
			return True
		except TypeError:
			return False
	def setup(self, start, end, interval = .0001, r = 1):
		self.__theta = np.arange(start,end,interval)
		if self.__isFunc(r):
			self.__r = [0.0]*len(self.__theta)
			for i in range(len(self.__theta)):
				self.__r[i] = r(self.__theta[i])
		else:
			self.__r = [r]*len(self.__theta)
		print (len(self.__theta))
		print (len(self.__r))
	def theta(self):
		return self.__theta
	def r(self):
		return self.__r
	def show(self):
		plt.polar(self.theta(), self.r())
		plt.show()
	def symmetry(self):
		polar_axis_sym = False
		pole_sym = False
		y_sym = False
		if self.__func is not None:
			randInt = random.randint(0,10)
		else:
			print ("Despotism and Godless Terrorism on the Home Front")	

class Secant: # class for the secant line of a function between two points
	def __init__(self, a, b, func):
		self.__a = a
		self.__b = b
		self.__func = func
		self.__x = np.arange(a,b,1/(10*np.abs(a-b)))
		self.__y = func(self.__x)
		self.__slope = (func(b) - func(a))/(b - a)
	def plot(self):
		plt.plot(self.__x, self.__y,self.__x,self.__slope*self.__x+self.__y[0])
		dash()
		plt.show()
	def chVal(self,a,b):
		self.__a = a
		self.__b = b
		self.__x = np.arange(self.__a,self.__b,1/(10*(np.abs(self.__a - self.__b))))
		self.__y = self.__func(self.__x)
		self.__slope = (self.__func(self.__b) - self.__func(self.__a))/(self.__b-self.__a)
	def slope(self):
		return self.__slope	
def Vector(object):
	def __init__(self,*args):
		# Components defined by args array
		self.__obj_comp = args
		# String representation
# Binary to decimal (Assembly practic)		
def btod(val):
	val = str(val)
	total = 0
	for i, x in enumerate(val):
		total += int(x)*2**(len(val)-(i+1))
	return total
	

# Physics
class kinematic: ### Assumes constant acceleration
    def __init__(self,delta_x = None,v0 = None,vf = None,t = None,a = None):
        self.delta_x = delta_x
        self.v0 = v0
        self.vf = vf
        self.t = t
        self.a = a
    def __check(self):
        if self.delta_x is None:
            pass
    def displacement(self):
            return self.v0*self.t + .5*self.a*self.t**2
    def final_v(self):
        return 
class vector:
    def __init__(self,components = None):
        self.__components = components
        self.__dimensions = len(self.__components)
    def __add__(self,other):
        pass

def find_g(L,h,t): # Calculate g for the given values of L (length of the incline), h (vertical distance from one photogate to the other), t (elapsed time)
    return 2*L/(t**2*h)

# General utility functions
def find(val, arr, caseInsensitive=True):
    if caseInsensitive:
        val = val.lower()
    matchArr = []
    currentIndex = 0
    prevIndex = 0
    for arrIndex, string in enumerate(arr):
        originalString=string
        if caseInsensitive:
            string = string.lower()
        for charIndex, char in enumerate(string):
            if char == val[currentIndex]:
                currentIndex += 1
                prevIndex = charIndex
            if charIndex-prevIndex > 1:
                #print(str(charIndex-prevIndex) + ":\tWord: " + string)
                currentIndex=0
            if currentIndex == len(val):
                matchArr.append(originalString)
                currentIndex=0
                break
    return matchArr
