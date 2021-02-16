
Parameters.txt
Text file containing the simulation parameters in format
<number of steps>,<timestep>,<Gravitational constant>
without the “<>”.
The units of the timestep are days and the units of the Gravitational constant are 
x10-24AU3kg-1(days)-1.

Particles.txt
Text file containing initial data for all bodies. The data for each body must be on a new line, and each line must be of the format
<label>,<x pos>,<y pos>,<z pos>,<x vel>,<y vel>,<z vel>,<mass>
without the “<>”.
The units of the components of position are days and the units of the components of velocity are AU/day.
The data for the bodies must be given in the order Sun, Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto, Halley’s Comet, Earth’s Moon.

NBodySim.py
Simulation code. Requires Particle3D and ParticleList methods to run. To run in command line type
python3 NBodySim.py Output.xyz Parameters.txt Particles.txt
Output.xyz produces data required to plot trajectories in VMD. The units of the positions are in AU.
A plot of total energy of the system against time is plotted. The units of time are days and the units of energy are x1024 kg AU2 (days)-2.
