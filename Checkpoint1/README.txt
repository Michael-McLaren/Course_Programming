#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 23:19:41 2021

@author: michaelmclaren
"""
When run it will create a data file filled the temperatures, average energy, 
average magnetic, energy variance, magnetic variance and error for specific 
heat capacity.

It will also create 4 plots:
average energy vs temperature
average magnetic vs temperature
susceptibility vs temperature
specific heat capacity vs temperature, with error


MaV_checkpoint1.py is where the simulation is run and the datafiles and plots 
generated

In the main function: 
if you want to turn on animation set the variable animation = 1
if you want to use glauber set choice = 0
if you want to use kawasaki set choice = 1

Functions:

    create_array:
    Inputs = lx, ly: which are the x and y dimensions of the array
    Output = spin_array: (lx,ly) sized array with elements either -1 or +1, 
    which are the spins
    
    get_NN:
    Input = array, coords: numpy array and coordinates in tuple form (x,y)
    Output = (s_north, s_east, s_south, s_west):  tuple of the spins of the 
    nearest neighbours 
    Notes: takes into account that the boundary wraps around
    
    get_NN_half:
    Notes: Same as get_NN except it only outputs (s_north, s_east)
    
    coords_list:
    Input = coords: coordinates in form of tuple (x,y)
    Output = coords_list: list of coordinate tuples corresponding to nearest 
    neighbours of the input coord
    
    glauber_energy_change:
    Input = array, coords: numpy array and coordinates in tuple form (x,y)
    Output = energy: the energy change if you were flip the spin at input 
    coordinates

    glauber_sim_step:
    Input = array, temp: numpy array and temperature, temp is just a scalar number
    Output = array: array with one spin possibly flipped
    Notes: picks random coordinate and then test to see if it will flip
    using glauber_energy_change.
    
    kawasaki_energy_change:
    Input = array, coords1 , coords2: numpy array and coordinates in tuple form (x,y)
    Output = energy: the energy change if you were to swap the two coordinates
    Notes:
    Chooses two coordinates and test the energy change if they switch places
    The energy change will be different they're right next to each other or the same spin
    This is what the if, elif statements are for

    kawasaki_sim_step:
    Input = array, temp: numpy array and temperature, temp is just a scalar number
    Output = array: numpy array, with two coordinates possibly swapped
    
    array_energy:
    Input = array: numpy array
    Output = E: total energy of the array
    
    jackknife_error:
    Input = E_list, N, T: E_list is a list of Energies of array, N is the size 
    of array, T is the temperature
    Output = error: approx error of specific heat capacity at one temperature















