""" CMod Astronomical N-Body Sim: ParticleList, a class to describe lists of 3D Particles

    Author: K. Farmer, M. McLaren
"""
from Particle3D import Particle3D
import numpy as np

class ParticleList(object):
    """
    Class to describe lists of 3D particles.
    
    Properties:
    p_list (list) - list of Particle3D objects
    size (float) - length of list of Particle3D objects
    
    Methods:
    * Gravitational force acting on each particle due to all other particles
    * second-order position update for all particles
    * first-order velocity update for all particles
    * centre-of-mass velocity update for all particles
    * total energy of system
    * relative vector separation of particles
    * formatted output
    """

    def __init__(self, list_of_particles):
        """
        Initialise a ParticleList instance
        
        :param p_list: list of Particle3D objects
        :param size: length of p_list as float
        """
        self.p_list = list_of_particles
        self.size = len(list_of_particles)
        
    def return_force(self,G):
        """
        Return force acting on each particle as sum of 
        gravitational forces acting on particle due to all other particles
        
        Uses Particle3D method force_gravitational
        """
        force_list = np.zeros([self.size,3])
        for i1 in range(self.size):
            for i2 in range(i1+1,self.size):
                force_on_p = Particle3D.force_gravitational(self.p_list[i1], self.p_list[i2], G)
                force_list[i1] += force_on_p
                force_list[i2] -= force_on_p
        return force_list

    def update_position(self, dt, force):
        """
        Second-order position update for all particles
        
        Uses Particle3D method leap_pos2nd
        
        :param dt: timestep
        :param force: numpy array of force acting on each particle
        """
        for i in range(self.size):
            particle = self.p_list[i]
            force_on_p = force[i] 
            particle.leap_pos2nd(dt, force_on_p)
            
    def update_velocity(self, dt, force):
        """
        First-order velocity update for all particles
        
        Uses Particle3D method leap_velocity
        
        :param dt: timestep as float
        :param force: numpy array of force acting on each particle
        """
        for i in range(self.size):
            particle = self.p_list[i]
            force_on_p = force[i]
            particle.leap_velocity(dt, force_on_p)
            
    def com_velocity_update(self):
        """
        Updates velocity of all particles to account for velocity of 
        centre-of-mass
        
        Centre-of-mass velocity is given by
        (total linear momentum of system)/(total mass of system)
        
        Uses Particle3D method linear_momentum
        """
        momentum, mass = 0, 0
        for particle in self.p_list:
            momentum += particle.linear_momentum()
            mass += particle.mass
        com_vel = momentum/mass
        for particle in self.p_list:
            particle.velocity -= com_vel

    def output_energy(self, G):
        """
        Return total energy of system, given by sum of kinetic energy of
        each particle and the potential energy of each particle - particle
        interaction
        
        Uses Particle3D methods kinetic_energy and pe_gravitational
        
        :Param G: Gravitational constant
        """
        energy = 0
        for particle in self.p_list:
            energy += particle.kinetic_energy()

        for x in range(self.size):
            for i in range(self.size-x-1):
                energy += Particle3D.pe_gravitational(self.p_list[x], self.p_list[i+x+1],G)
        return energy
        
    def vec_separation(self):
        """
        Return list of vector separation of each particle in list
        from first particle in list. For 12th particle in list, separation
        is calculated relative to 4th particle in list.
        
        Uses Particle3D method separation()
        """
        vec_sep = []
        for i in range(1,self.size):
            if i == 11:
                vec_sep1 = Particle3D.separation(self.p_list[i],self.p_list[3])
                vec_sep.append(vec_sep1)
            if i != 11:
                vec_sep2 = Particle3D.separation(self.p_list[i],self.p_list[0])
                vec_sep.append(vec_sep2)
        return vec_sep
	
    def VMD_write(self, outfile, point):
        """
        Write to file in format required for VMD. Format is
        <no. of particles>
        <point>
        <particle1 label> <particle1 x_pos> <particle1 y_pos> <particle1 z_pos>
        
        :param outfile: filehandle of output file to be written to
        :param point: point in simulation as int
        """
        outfile.write(str(self.size) + "\n")
        outfile.write('Point = ' + str(point+1)+ "\n")
        for particle in self.p_list:
            labels = particle.__str__()
            outfile.write("{0} {1:12.8f} {2:12.8f} {3:12.8f}\n".format(labels[0],labels[1],labels[2],labels[3]))


