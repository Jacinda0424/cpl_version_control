import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from nbody import Particles, NbodySimulation
from numba import jit, njit, prange, set_num_threads

def initial_NormalDistribution(N = 100000, total_mass = 20):
    particles = Particles(N=N)
    masses = particles.get_masses()
    mass = total_mass/particles.nparticles
    particles.get_kinetic_energy()
    particles.get_potential_energy()
    particles.set_masses((masses*mass))
    positions = np.random.randn(N,3)
    velocities = np.random.randn(N,3)
    accelerations = np.random.randn(N,3)
    particles.set_positions(positions)
    particles.set_velocities(velocities)
    particles.set_accelerations(accelerations)
    particles.set_kinetic_energy(Ek,mode="init")
    particles.set_potential_energy(Eu,mode="init")
    return particles

time          = 0                           
num_particles = 100000                      
masses        = np.ones((num_particles,1))
positions     = np.zeros((num_particles,3))
velocities    = np.zeros((num_particles,3))
accelerations = np.zeros((num_particles,3))
tags          = np.linspace(1,num_particles,num_particles)
particles = Particles(N=num_particles)
particles.output(fn="data_particles_100000.txt",time=time)
t,m,x,y,z,vx,vy,vz,ax,ay,az,Ek,Eu = np.loadtxt("data_particles_100000.txt")

set_num_threads(4)

problem_name = "N_particles_100000"
G = 6.67428e-8
particles = initial_NormalDistribution(N = 100000, total_mass = 20)
sim = NbodySimulation(particles=particles)
sim.setup(G=G,rsoft=0.01,method="RK4",io_freq=20,io_title=problem_name,io_screen=True,visualized=False)
sim.evolve(dt=0.01,tmax=10)

import glob
fns = "data_"+problem_name+"/data_"+problem_name+"_[0-9][0-9][0-9][0-9][0-9].txt"
fns = glob.glob(fns)
fns.sort()

scale = 3 

fig, ax =plt.subplots()
fig.set_size_inches(10.5, 10.5, forward=True)
fig.set_dpi(72)
line, = ax.plot([],[],'o')

def init():
    ax.set_xlim(-1*scale,1*scale)
    ax.set_ylim(-1*scale,1*scale)
    ax.set_aspect('equal')
    ax.set_xlabel('X [code unit]')
    ax.set_ylabel('Y [code unit]')
    return line,

def updateParticles(frame):
    fn = fns[frame]
    m,t,x,y,z,vx,vy,vz,ax,ay,az,Ek,Eu = np.loadtxt(fn)
    #print("loadtxt done",fn)
    line.set_data(x,y)
    plt.title("Step ="+str(frame))
    return line,

ani = animation.FuncAnimation(fig, updateParticles, frames=len(fns),init_func=init, blit=True)
ani.save('movie_'+problem_name+'.mp4',fps=10)
