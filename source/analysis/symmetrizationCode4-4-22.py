# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 10:06:02 2022

@author: zachh
"""


import numpy as np
import matplotlib.pyplot as plt # plotting
from matplotlib import cm
import pandas as pd
from scipy.interpolate import interp1d
from scipy.misc import derivative
from scipy.interpolate import InterpolatedUnivariateSpline
#from scipy.misc import derivative
#plt.style.use('custom')


#################################
#################################
#######input information#########
#################################
#################################
#filename = 'Sn110921_ch1_Sn110521_ch3Rxxvs magnetic field.csv'#raw data from PPMS
Bmax = 14 #max field used; important for interpolation fucntion
thickness = 40*10**-7#thickness in centimeters
geo = 4.35236 #thickness in centimeters Van der Pauw: pi/ln(2) =4.53236; Hall bar = lengh/Area
saturation = 6 # % of data used to fit the linear background (i.e. for a 10T scan, 10 would give a fit over 9-10T data range)



########## USER ENTERED INFORMATION #########################
filename = 'RxyvsH18tempd.csv'#raw data from PPMS
datafull = np.genfromtxt(filename,delimiter=',',skip_header=6)
Cols = 4 #number of data columns per B field(include B field column)
#below is the column number of each of the data points beginning at position zero
#this lets the code know which columns are what
Tpos = 0
Bpos = 1
Rxx1pos = 2
Rxx2pos =2
Rxypos =3 #set Rxypos = 99 if there is no Rxy data
savedFileName = 'AnjSn645.csv' #a csv file with this name will be saved. two files of the exact same name cannot exist in the folder
n2dFileName = 'n2dDataoutout.csv'
#now enter geometries
l = 600 #in um
w = 200 #in um
t = 2 #in nm
#are you using Ryy data in place of Rxx2?
#if yes, put Ryypos = 101
#if no, put Ryypos = 201
Ryypos = 201


#########################################################################
#########################################################################
################# Function for Symmetrization ###########################
#########################################################################
#########################################################################
#########################################################################
def symRxxs(B,Rxx1,Rxx2,Bkey,Rxy):
    "symmetrization process"
    Bup = B[:Bkey[0][0]]
    Bdown = B[Bkey[0][0]:]
    Bup = B[:Bkey[0][0]]
    Bdown = B[Bkey[0][0]:]

    #resistance upsweep/downsweep
    Rxx1up = Rxx1[:Bkey[0][0]] #upsweeps
    Rxx2up = Rxx2[:Bkey[0][0]]

    Rxx1down = Rxx1[Bkey[0][0]:] #downsweeps
    Rxx2down = Rxx2[Bkey[0][0]:]
    
    Bupm = Bup*-1 #creates B field opposite sweep
    Bdownm = Bdown*-1
    Rxx1int_up = interp1d(Bup, Rxx1up, kind='slinear',fill_value="extrapolate",bounds_error=False) #interpolates -B values
    Rxx2int_up = interp1d(Bup, Rxx2up, kind='slinear',fill_value="extrapolate",bounds_error=False)

    Rxx1int_down = interp1d(Bdown, Rxx1down, kind='slinear',fill_value="extrapolate",bounds_error=False) #interpolates +B values
    Rxx2int_down = interp1d(Bdown, Rxx2down, kind='slinear',fill_value="extrapolate",bounds_error=False)

    ##For symmetrization, interpolation is made so f(Rxx_upsweep) == f(-Rxx_downsweep), and f(Rxx_downsweep) == f(-Rxx_upsweep).  Always use Bup when plotting 
    Rxx1_upm = Rxx1int_up(Bupm) #interps -B values
    Rxx2_upm = Rxx2int_up(Bupm)

    Rxx1_downA = Rxx1int_down(Bup) #interps -B values
    Rxx2_downA = Rxx2int_down(Bup)

    Rxx1_downm = Rxx1int_down(Bupm)
    Rxx2_downm = Rxx2int_down(Bupm) #interps -B values

    #averaging
    Rxx1sym_up = (Rxx1_downm + Rxx1up)/2
    Rxx2sym_up = (Rxx2_downm + Rxx2up)/2

    Rxx1sym_down = (Rxx1_upm + Rxx1_downA)/2
    Rxx2sym_down = (Rxx2_upm + Rxx2_downA)/2

    #combining
    Rxx1sym = np.append(Rxx1sym_up, np.flipud(Rxx1sym_down))
    Rxx2sym = np.append(Rxx2sym_up, np.flipud(Rxx2sym_down))
    
    #normalizing
    MRpct1up = (Rxx1sym_up/Rxx1int_up(0))*100
    MRpct1down = (Rxx1sym_down/Rxx1int_up(0))*100
    MRpct2up = (Rxx2sym_up/Rxx2int_up(0))*100
    MRpct2down = (Rxx2sym_down/Rxx2int_up(0))*100
    
    #Rxy sym process
    if Rxypos != 99:
        Rxyup = Rxy[:Bkey[0][0]]
        Rxydown = Rxy[Bkey[0][0]:]
        Rxyint_up = interp1d(Bup, Rxyup, kind='slinear',fill_value="extrapolate",bounds_error=False)
        Rxyint_down = interp1d(Bdown, Rxydown, kind='slinear',fill_value="extrapolate",bounds_error=False)
        Rxy_upm = Rxyint_up(Bupm)
        Rxy_downA = Rxyint_down(Bup)
        Rxy_downm = Rxyint_down(Bupm) #for some reason there are nan for i =6
        Rxy_symup = (-Rxy_downm+Rxyup)/2
        Rxy_symdown = (-Rxy_upm+Rxy_downA)/2
        Rxysym = np.append(Rxy_symup, np.flipud(Rxy_symdown))
    elif Rxypos == 99:
        Rxy_symup = np.zeros(len(Rxx1up))
        Rxy_symdown = np.zeros(len(Rxx1up))
        Rxysym = np.zeros(len(Rxx1up))
    
    return(Bup,Rxx1sym_up,Rxx1sym_down,Rxx2sym_up,Rxx2sym_down,MRpct1up,MRpct1down,MRpct2up,MRpct2down,Rxy_symup,Rxy_symdown,Rxysym) #


#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################

#defining data sets
ylen = np.int(len(datafull[:,0]))
xlen = len(datafull[0,:])
num_datasets = int(len(datafull[0,:])/Cols)
n2dout = np.zeros(num_datasets)
titlesT = np.zeros(num_datasets)

if Rxypos != 99:#creating empty vector to store final symmed data
    outputDatasets = 12 #different number of output data columns for Rxy vs just Rxxs
    symCols = np.int(outputDatasets*num_datasets) + 1
    SymmedData = np.zeros((ylen,symCols))#if Rxy is included, need more data
elif Rxypos == 99:
    outputDatasets = 10
    symCols = np.int(outputDatasets*num_datasets) + 1
    SymmedData = np.zeros((ylen,symCols))



for i in range(num_datasets): #indexing through the data file
    Tempz = datafull[:,(i*Cols)] #grabbing full temp column 
    locate = np.where(np.isnan(Tempz) == True) #finding where it becomes nan
    if len(locate[0]) == 0: #finds if locate has any nan
        position = len(Tempz) - 1 #if no nan in Temp, stores position and continues
    elif len(locate) > 0:
        position = locate[0][0] - 1 #minus 1 to last real number position
    
    #storing data
    Temp = datafull[:position,(i*Cols)+Tpos] #data stores with only real numbers
    B = datafull[:position,(i*Cols)+Bpos] 
    Rxx1 = datafull[:position,(i*Cols)+Rxx1pos]
    Rxx2 = datafull[:position,(i*Cols)+Rxx2pos]
    if Rxypos != 99: #if Rxy data exists, save it
        Rxy = datafull[:position,(i*Cols)+Rxypos]
    elif Rxypos == 99:
        Rxy = 99
    

    #finding upsweeps and downsweeps
    if np.sign(B[0]) != np.sign(B[-1]):    #if this is true, it only has a downsweep or only has an upsweep
        Bkey = np.where(B == np.min(np.abs(B)))
        SWEEPS = 1 #sweeps lets if statements below know 
    elif B[0] < 0:
        Bkey = np.where(B == np.max(B)) #place where sweeps change
        SWEEPS = 0
    elif B[0] > 0:
        Bkey = np.where(B == np.min(B)) #place where sweeps change
        SWEEPS = 0
    
    #saving data
    SymmedData[0:position,(i*outputDatasets)] = Temp #saving temp data
    SymmedData[0:Bkey[0][0],(i*outputDatasets)+1] = symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[0] #Bup data
    SymmedData[0:Bkey[0][0],(i*outputDatasets)+2] = symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[1] #Rxx1up Data
    SymmedData[0:Bkey[0][0],(i*outputDatasets)+3] = symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[2] #Rxx1down data
    SymmedData[0:Bkey[0][0],(i*outputDatasets)+4] = symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[3] #Rxx2up
    SymmedData[0:Bkey[0][0],(i*outputDatasets)+5] = symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[4] #Rxx2down
    SymmedData[0:Bkey[0][0],(i*outputDatasets)+6] = symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[5] #Rxx1up MR
    SymmedData[0:Bkey[0][0],(i*outputDatasets)+7] = symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[6] #Rxx1down MR
    SymmedData[0:Bkey[0][0],(i*outputDatasets)+8] = symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[7] #Rxx2up MR
    SymmedData[0:Bkey[0][0],(i*outputDatasets)+9] = symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[8] #Rxx2down MR
    if Rxypos != 99:
        SymmedData[:Bkey[0][0],(i*12)+10] = symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[9] #Rxyup
        SymmedData[:Bkey[0][0],(i*12)+11] = symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[10] #Rxydown
    
    
    #now, call symmetrization function and store the data##################
    if SWEEPS == 1: #for data that only has down/upsweep
        plt.figure(1)
        plt.plot(symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[0],symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[1]) 
        
        plt.figure(2)
        plt.plot(symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[0],symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[3])
        
        plt.figure(3)
        plt.plot(symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[0],symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[5])
        
    elif SWEEPS == 0: #for data that has upsweeps and downsweeps
        legends = ['2K','5K','10K','20K','30K','40K','50K','60K','70K','80K','90K','100K','300K']
        plt.figure(1) #plotting Rxx1 symmetrized
        plt.plot(symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[0],symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[1],label=legends[i]) #,label=legends[i]
        plt.plot(symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[0],symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[2],label=legends[i])
        plt.xlabel('Magnetic Field (Oe)')
        plt.ylabel('Resistance')
        #plt.legend()
    
        plt.figure(2) #plotting Rxx2 symmetrized
        plt.plot(symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[0],symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[3])
        plt.plot(symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[0],symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[4])
        plt.xlabel('Magnetic Field (Oe)')
        plt.ylabel('Resistance')
    
        plt.figure(3) #plotting normalized data
        plt.plot(symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[0],symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[5])
        plt.plot(symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[0],symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[6])
        plt.xlabel('Magnetic Field (Oe)')
        plt.ylabel('Resistance')
        plt.title('% Magnetoresistance')
        
    if Rxypos != 99:
        plt.figure(10) #plotting Rxy symmetrized
        plt.plot(symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[0],symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[9])
        plt.plot(symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[0],symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[10])
        plt.xlabel('Magnetic Field (Oe)')
        plt.ylabel('Resistance')
        plt.title('Rxy Symmetrized')
        ## finding n2d
        # Rxysym = symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[11]
        # Bn2d = np.append(symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[0],np.flipud(symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[0]))
        # z = np.polyfit(Bn2d, Rxysym, 1);
        #z1 = np.polyfit(symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[0],symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[9],1)
        z1 = (symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[9][-1]-symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[9][0])/(symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[0][-1]-symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[0][0])
        R_H = z1 * 10000 * t * 10**-9 #rh in m(thats why 10^-9)
        n3d = 1/(z1 * (1.602*10**-19) * 10**6) #times 10^6 to get into cm^3 from m^3
        n2d = n3d*(t*10**-7) #converting to 2d density. 10^-7 to turn nm to cm
        n2dout[i] = n2d
        print('For T=',np.round(np.mean(Temp),1),'K, n2d=',np.round(n2d,3))
        titlesT[i] = np.round(np.mean(Temp),1)
    
   # if Ryypos == 101:
        # R_0 = symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[9][np.argmin(abs(symRxxs(B,Rxx1,Rxx2,Bkey,Rxy)[0]))]
        # Rsheet = R_0 *w/l #sheet resistance
        # resisitivity = (Rsheet*t*(10**-9)*w/l)*10**5 #in units milliOhm cm
        # mobili = (1/(resisitivity*(1.602*10**-19)*n3d))*1000 #cm^2/(Vs))
        # print(mobili)
        

    

    
###### storing data
#creating title vector
titles = ['Temp.','Bup','Rxx1up','Rxx1down','Rxx2up','Rxx2down','Rxx1up MR','Rxx1down MR','Rxx2up MR','Rxx2down MR']
titles2 = ['Temp.','Bup','Rxx1up','Rxx1down','Rxx2up','Rxx2down','Rxx1up MR','Rxx1down MR','Rxx2up MR','Rxx2down MR']
rxytitles = ['Rxyup','Rxydown']
if Rxypos != 99: #adds titles for rxy if they exist
    titles = np.append(titles,rxytitles)
    titles2 = np.append(titles2,rxytitles)

for i in range(num_datasets-1):
    titles = np.append(titles,titles2)

end = 'end'
titles = np.append(titles,end)
SymmedData = np.vstack([titles,SymmedData])
#saving to csv. 
np.savetxt(savedFileName,SymmedData, delimiter = ',',fmt = "%s")

n2dout = np.vstack([titlesT,n2dout])

np.savetxt(n2dFileName,n2dout,delimiter=',',fmt = "%s")
        

    


