"""CMod Astronomical N-Body Sim: velocity Verlet time integration of
an N-Body system

Produces an output file of the format required for VMD trajectories
Also prints the peri- and apo- apses for all bodies except the sun
to the terminal

To run the code type
<program><desired output file name><parameters file><particles file>
The parameters file should be of the format
no. of steps, time-step, Gravitational constant
The particles file should have each Body on a separate line and
the bodies should be in the order
Sun, Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, 
Neptune, Pluto, Halley's Comet, Earth's moon
and the data format for the bodies should agree with that of Particle3D

Author: K. Farmer, M. McLaren
"""

import sys
import numpy as np
import matplotlib.pyplot as pyplot
from Particle3D import Particle3D
from ParticleList import ParticleList

# Begin main code
def main():

    # Read name of output and input files from command line
    if len(sys.argv) != 4:
        print("Wrong number of arguments.")
        print("Require: " + sys.argv[0] + " <output_file><input_parameters><input_particles>")
    else:
        outfile_name = sys.argv[1]
        parameters = sys.argv[2]
        particle_list = sys.argv[3]

    # Open output files
    outfile1 = open(outfile_name, "w")

    # Open first input file
    with open(parameters, "r") as file:
        # Set up simulation parameters
        line = file.readline()
        token = line.split(",")
        num_step = int(token[0])
        dt = float(token[1])
        G = float(token[2])

    # Open second input file
    with open(particle_list, "r") as file:
        # Read particles into list
        p_list = []
        for line in file:
            p_list.append(Particle3D.create_particle(line))

    # Initialise ParticleList and initial conditions
    Par_list = ParticleList(p_list)
    time = 0
   
    # Initial force
    force = Par_list.return_force(G)
    
    # Initialise data lists
    time_list = [0]
    energy_list = [Par_list.output_energy(G)]
    particle_separation = []
    
    # Create list of initial vector separations & append
    # initial vector separations to particle_separation list
    vec_initial_separation = Par_list.vec_separation()
    particle_separation.append(vec_initial_separation)

    # Update initial velocities to account for non-vanishing linear momentum of system
    Par_list.com_velocity_update()
    
    # Main loop
    for i in range(num_step):
         # Update particle positions
        Par_list.update_position(dt, force)

        # Update forces
        force_new = Par_list.return_force(G)
        # Update particle velocity by averaging
        # current and new forces
        Par_list.update_velocity(dt, 0.5*(force + force_new))

        # Re-define force value
        force = force_new

        # Increase time
        time += dt

        # Output particle information
        if i%10==0:
            # Append information to lists
            time_list.append(time)
            energy_list.append(Par_list.output_energy(G))
            # Write information to output file
            Par_list.VMD_write(outfile1, i)
            
            # Append separation from Sun (Earth for Earth's moon) to ParticleSeparation
            timestep_vec_separation = Par_list.vec_separation()
            particle_separation.append(timestep_vec_separation)
            
                
    # Post-simulation:
    # Close output file
    outfile1.close()
    
    # Plot energy vs time to screen
    pyplot.plot(time_list, energy_list)
    pyplot.title("Energy vs Time")
    pyplot.xlabel("Time (days)")
    pyplot.ylabel("Energy (x10^24 kg (AU)^2 (days)^-2")
    pyplot.show()
                

    #Apo- & Peri- apses calculations, print to terminal
    print("Body apoapses periapses")
    for i in range(11):
        list1 = []
        for j in range(len(particle_separation)):
            list1.append(np.linalg.norm(particle_separation[j][i]))
        apo = max(list1)
        peri = min(list1)
        print(Par_list.p_list[i+1].label,apo,peri)
        
    # Period calculations, print to terminal
    print("Body period")
    for i in range(11):
        inner_products = [] 
        period_times = []
        periods = []
        for j in range(len(particle_separation)):
            # Calculate inner product of initial position and timestep position
            inner_products.append(np.inner(vec_initial_separation[i],particle_separation[j][i]))
        for k in range(1,len(inner_products)):
            if inner_products[k] < 0 and inner_products[k-1] >= 0:
                # Append times when inner product goes from positive to strictly negative
                period_times.append(time_list[k])
        for l in range(1,len(period_times)):
            # Append adjacent times in period_times to list of periods
            periods.append((period_times[l]-period_times[l-1]))
        if len(periods) != 0:
            # Average entries in period to determine average period
            Period = sum(periods)/len(periods)
            print(Par_list.p_list[i+1].label,Period)
        
              
# Execute main method, but only when directly invoked
if __name__ == "__main__":
    main()
