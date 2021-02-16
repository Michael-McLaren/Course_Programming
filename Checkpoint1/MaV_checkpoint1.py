# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def create_array(lx,ly):
    #Init array of spins
    spin_array = np.zeros((lx,ly))
    
    for x in range(lx):
        for y in range(ly):
            split = random.random()
            if split < 0.5:
                spin_array[x][y] = 1
            else:
                spin_array[x][y] = -1
    return spin_array

def get_NN(array,coords):
    #Input numpy array and coordinates in tuple form (x,y)
    #Outputs tuple of the spins of the nearest neighbours 
    #Taking into account that the boundary wraps around
    shape = array.shape
    lx = shape[0] - 1
    ly = shape[1] - 1
    xx = coords[0]
    yy = coords[1]
    #north
    if xx == 0:
        s_north = array[lx][yy]
    else:
        s_north = array[xx-1][yy]
    #east
    if yy == ly:
        s_east = array[xx][0]
    else:
        s_east = array[xx][yy+1]
    #south
    if xx == lx:
        s_south = array[0][yy]
    else:
        s_south = array[xx+1][yy]
    #west
    if yy == 0:
        s_west = array[xx][ly]
    else:
        s_west = array[xx][yy-1]
        
    return (s_north, s_east, s_south, s_west)

def get_NN_half(array,coords):
    #Used to stop double counting when calculating the energy
    #Same premise as get_NN, just only for north and east
    shape = array.shape
    lx = shape[0] - 1
    ly = shape[1] - 1
    xx = coords[0]
    yy = coords[1]
    #north
    if xx == 0:
        s_north = array[lx][yy]
    else:
        s_north = array[xx-1][yy]
    #east
    if yy == ly:
        s_east = array[xx][0]
    else:
        s_east = array[xx][yy+1]
    return (s_north, s_east)
    

def coords_list(coords):
    #Inputs coordinates in form of tuple (x,y)
    #Outputs list of coordinate tuples corresponding to nearest neighbours of the input coord
    xx = coords[0]
    yy = coords[1]
    coords_list = [(xx+1,yy),(xx,yy+1),(xx-1,yy),(xx,yy-1)]
    return coords_list

def glauber_energy_change(array, coords):
    #Input array and coords
    #Output the energy change if you were flip the spin at input coordinates
    s = array[coords]
    NN = get_NN(array, coords)
    energy = 2*s*(sum(NN))
    return energy


def glauber_sim_step(array, temp):
    #Input array and temperature, temp is just a scalar number
    #Outputs the array 
    #Once its been tested if the energy change will allow for the flip to happen
    shape = array.shape
    lx = shape[0] - 1
    ly = shape[1] - 1
    xx = random.randint(0,lx)
    yy = random.randint(0, ly)
    coords = (xx,yy)
    
    energy_change = glauber_energy_change(array, coords)
    rand = random.random()
    prob = np.exp(-1*energy_change / temp)
    
    if energy_change < 0 or rand <= prob:
        array[coords] = -1*array[coords]
    return array

def kawasaki_energy_change(array, coords1 , coords2):
    #Chooses two coordinates and test the energy change if they switch places
    #The energy change will be different they're right next to each other or the same spin
    #This is what the if, elif statements are for
    s1 = array[coords1]
    s2 = array[coords2]
    
    NN1 = get_NN(array, coords1)
    NN2 = get_NN(array, coords2)
    #tests to see if the spins are the same, in which case nothing changes
    if s1 == s2:
        return 0
    #Checks to see if they two chosen points are nearest neighbours
    #In which case the energy change equation is different
    elif (coords1 in coords_list(coords2)):
        energy1 = 2*s1*(sum(NN1) - array[coords1])
        energy2 = 2*s2*(sum(NN2) - array[coords2])
        energy = energy1 + energy2
        return energy
    #The normal energy change
    else:
        energy1 = 2*s1*(sum(NN1))
        energy2 = 2*s2*(sum(NN2))
        energy = energy1 + energy2
        return energy



def kawasaki_sim_step(array, temp):
    #Input array and temperature, temp is just a scalar number
    #Outputs the array 
    #Once its been tested if the energy change will allow for the swap to happen
    shape = array.shape
    lx = shape[0] - 1
    ly = shape[1] - 1
    #picking random coordinates
    xx1 = random.randint(0,lx)
    yy1 = random.randint(0,ly)
    coords1 = (xx1,yy1)
    
    xx2 = random.randint(0,lx)
    yy2 = random.randint(0,ly)                  
    coords2 = (xx2,yy2)

    #Check same spot hasnt been picked twice
    while coords1 == coords2:
        xx2 = random.randint(0,lx)
        yy2 = random.randint(0, ly)
        coords2 = (xx2,yy2)
    #Swap the two chosen spins, if negative  energy chyange or according to a probability
    energy_change = kawasaki_energy_change(array, coords1, coords2)
    rand = random.random()
    prob = np.exp(-1*energy_change / temp)
    #The conditions on the energy change
    if energy_change < 0  or rand <= prob: 
        old1 = array[coords1].copy()
        array[coords1] = array[coords2]
        array[coords2] = old1
        
    return array 
    
    
    

def array_energy(array):
    #Inputs array
    #Outputs total energy of the array
    shape = array.shape
    lx = shape[0] - 1
    ly = shape[1] - 1
    
    E = 0
    for x in range(lx):
        for y in range(ly):
            E -= array[(x,y)]*sum( get_NN_half(array, (x,y)) )  
    return E 

def jackknife_error(E_list, N, T):
    #Inputs E_list = list of Energies of array, N = size of array, T = temperature
    #Outputs approx error of specific heat capacity at one temperature
    sum_diff = 0
    #Calculate specific heat capacity
    c = np.var(E_list) / (N*T**2)
    for i in range(len(E_list)-1):
        E_list_copy = E_list.copy() #to remove the first element of the list
        E_list_copy.pop(i)
        c_i = np.var(E_list_copy) / (N*T**2)
        diff = (c_i - c)**2
        sum_diff += diff
    error = np.sqrt(sum_diff)
    return error
    
        
        

def main():
    #Choose glauber or kawasaki. Set choice as 0 or 1 respectively.
    #Also changes datafile and plot names
    choice = 1
    step_func_choice = [glauber_sim_step, kawasaki_sim_step]
    string_change_choice = ['_glauber', '_kawasaki']
    step_func = step_func_choice[choice]
    string_change = string_change_choice[choice]
    
    #set animation to equal 1 to turn it on
    animation = 0

    #Set up initial array
    lx = 50
    ly = 50
    N = lx*ly
    array = create_array(lx, ly)
    
    #Wait till equilibrium
    init_temp = 1
    sweeps = 0
    while sweeps < 5000:
        #One sweep is 2500 flips
        for i in range(2500):
            array = step_func(array, init_temp)
            
        #animation
        if sweeps%100 == 0 and animation == 1:
            plt.ion()
            plt.title(sweeps)
            plt.imshow(array)
            plt.draw()
            plt.pause(0.0001)
            plt.clf()
            
        sweeps += 1
    
    #set up data lists
    average_E_list = []
    average_mag_list = []
    diff_mag_list = []
    diff_E_list = []
    error_list = []
    temp_list = np.linspace(1,3,21)
    #main simulation loop through temperatures
    with open('data'+string_change+'.txt', 'w') as file:
        file.write('temp, average_E, average_mag, diff_E, diff_mag, error'+'\n')
        for temp in temp_list:
            print(temp)
            sweeps = 0
            #magnetic
            mag_list = []
            #energy
            E_list = []
            while sweeps < 10000:
                #One sweep is 2500 flips
                for i in range(2500):
                    array = step_func(array, temp)
                    
                #Animation every 100 sweeps
                
                if sweeps%100 == 0 and animation == 1:
                    plt.ion()
                    plt.title(sweeps)
                    plt.imshow(array)
                    plt.draw()
                    plt.pause(0.0001)
                    plt.clf()
                
                
                #waiting for equilibrium then taking values every 10
                if (sweeps > 100 and sweeps % 10 == 0):
                    #magnetic
                    mag = abs((np.sum(array)))
                    mag_list.append(mag)
                    #energy
                    E = array_energy(array)
                    E_list.append(E)  
                    
                sweeps += 1
            
            n = len(E_list)
            average_E= str(sum(E_list)/n)
            average_mag = str(sum(mag_list)/n)
            #Variance of magnetic
            diff_mag = str(np.var(mag_list) ) 
            #Variance of Energy
            diff_E = str(np.var(E_list))
            #error
            error = str(jackknife_error(E_list, N, temp))
            #Write to file every temp with the data 
            #in form (average E, average mag, variance of E, variance of mag, error)
            string = str(temp)+', '+average_E+', '+average_mag+', '+diff_E+', '+diff_mag+', '+error+'\n'
            file.write(string)
            
    with open('data'+string_change+'.txt', 'r') as file:
        next(file)
        for line in file:
            token = line.split(', ')
            average_E_list.append(float(token[1]))
            average_mag_list.append(float(token[2]))
            diff_E_list.append(float(token[3]))
            diff_mag_list.append(float(token[4]))
            error_list.append(float(token[5]))
        
    one_over_T = np.reciprocal(temp_list)
    sus_list = ( one_over_T*np.array(diff_mag_list) ) / (N)
    specific_heat_list = ( (one_over_T**2)*np.array(diff_E_list) ) / (N)


    plt.plot(temp_list, sus_list)
    plt.title('Susceptibility')
    plt.xlabel('Temperature(T)')
    plt.ylabel('Susceptibility')
    plt.savefig('susceptibility_graph'+string_change+'.png')
    plt.clf()
    
    plt.errorbar(temp_list, specific_heat_list, yerr=error_list, ecolor='red', elinewidth=1, capsize=3)
    plt.title('Specific Heat')
    plt.xlabel('Temperature')
    plt.ylabel('Specific Heat')
    plt.savefig('specific_heat_graph'+string_change+'.png')
    plt.clf()
    
    plt.plot(temp_list, average_mag_list)
    plt.title('Average Magnetisation')
    plt.xlabel('Temperature')
    plt.ylabel('Average Magnetisation')
    plt.savefig('average_magnetisation_graph'+string_change+'.png')
    plt.clf()
    
    plt.plot(temp_list, average_E_list)
    plt.title('Average Energy')
    plt.xlabel('Temperature')
    plt.ylabel('Average Energy')
    plt.savefig('average_energy'+string_change+'.png')
    plt.clf()
 


main()