""" CMod Astronomical N-Body Sim: Particle3D, a class to describe 3D particles
    
    Author: K. Farmer, M. McLaren
    """

import numpy as np

class Particle3D(object):
    """
    Class to describe 3D particles.

    Properties:
    label (string) - name/label for the particle
    position(numpy array, float) - position along the x, y and z axis
    velocity(numpy array, float) - velocity along the x, y and z axis
    mass(float) - particle mass

    Methods:
    * formatted output
    * kinetic energy
    * linear momentum
    * first-order velocity update
    * first- and second- order position update
    * create particle from a file
    * relative vector separation of 2 particles
    * gravitational potential energy of 2 particles
    * gravitational force of a particle due to another particle
    """

    def __init__(self, label, x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, mass):
        """
        Initialise a Particle3D instance

        :param label: label for particle as string
        :param x_ pos/ y_pos/ z_pos: position along x/y/z axis as float
        :param x_vel/ y_vel/ z_vel: velocity along x/y/z axis as float
        :param mass: mass as float
        """
        self.label = label
        self.position = np.array([x_pos, y_pos, z_pos], dtype=float)
        self.velocity = np.array([x_vel, y_vel, z_vel], dtype = float)
        self.mass = float(mass)
    

    def __str__(self):
        """
        Define output format.
        For particle p=(particle1, 1.0, 2.0, 3.0, 0.5, 1.5, 2.5, 1.0) this will print as
        "particle1 [1.0 2.0 3.0]"
        """
        return(self.label, self.position[0], self.position[1], self.position[2])

    
    def kinetic_energy(self):
        """
        Return kinetic energy as
        1/2*mass*(x_vel^2 + y_vel^2 + z_vel^2)
        """
        return 0.5*self.mass*np.inner(self.velocity,self.velocity)


    def linear_momentum(self):
        """
        Return linear momentum as
        mass*(x_vel^2 + y_vel^2 + z_vel^2)^(1/2)
        """
        return self.mass*self.velocity
        

    # Time integration methods
    def leap_velocity(self, dt, force):
        """
        First-order velocity update,
        v(t+dt) = v(t) + dt*F(t)

        :param dt: timestep as float
        :param force: force on particle as float
        """
        self.velocity += dt*force/self.mass


    def leap_pos1st(self, dt):
        """
        First-order position update,
        r(t+dt) = r(t) + dt*v(t)

        :param dt: timestep as float
        """
        self.position += dt*self.velocity


    def leap_pos2nd(self, dt, force):
        """
        Second-order position update,
        r(t+dt) = r(t) + dt*v(t) + 1/2*dt^2*F(t)

        :param dt: timestep as float
        :param force: current force as float
        """
        self.position += dt*self.velocity + 0.5*dt**2*force/self.mass
    
    @staticmethod
    def create_particle(line):
        """
        Return a Particle3D object using data from a file
        File should have data listed as
        label, x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, m
        """
        tokens = line.split(",")
        label, x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, m = tokens[0:8]
            
        return Particle3D(label, x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, m)
    
    @staticmethod 
    def separation(particle1, particle2):
        """
        return separation of two particles,
        r2(t) - r1(t)
        
        :param particle1: Particle3D instance
        :param particle2: Particle3D instance
        """
        return (particle2.position - particle1.position)
     
    @staticmethod
    def pe_gravitational(particle1, particle2, G):
        """
        return Gravitational potential energy for two interacting particles
        U = -G*m1*m2/r12

        :param particle1: Particle3D instance
        :param particle2: Particle3D instance
        :param G: Gravitational constant as float
        """
        
        vector_r12 = particle1.separation(particle1, particle2)
        r12 = np.linalg.norm(vector_r12)
        return -G*particle1.mass*particle2.mass/r12
        
    @staticmethod
    def force_gravitational(particle1, particle2, G):
        """
        return the gravitational force acting on particle 1 due to particle 2
        Force has magnitude
        F = G*m1*m2/(r12)^2
        in the direction
        r2 - r1

        :param particle1: Particle3D instance
        :param particle2: Particle3D instantce
        :param G: Gravitational constant as float
        """
    
        vector_r12 = particle1.separation(particle1,particle2)
        r12 = np.linalg.norm(vector_r12)
        unit_r12 = vector_r12/r12
        return ((G*particle1.mass*particle2.mass)/r12**2)*unit_r12

        



