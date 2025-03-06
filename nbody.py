import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from numba import jit, njit, prange, set_num_threads

"""

This program solve 3D direct N-particles simulations 
under gravitational forces. 

This file contains two classes:

1) Particles: describes the particle properties
2) NbodySimulation: describes the simulation

Usage:

    Step 1: import necessary classes

    from nbody import Particles, NbodySimulation

    Step 2: Write your own initialization function

    
        def initialize(particles:Particles):
            ....
            ....
            particles.set_masses(mass)
            particles.set_positions(pos)
            particles.set_velocities(vel)
            particles.set_accelerations(acc)

            return particles

    Step 3: Initialize your particles.

        particles = Particles(N=100)
        initialize(particles)


    Step 4: Initial, setup and start the simulation

        simulation = NbodySimulation(particles)
        simulation.setip(...)
        simulation.evolve(dt=0.001, tmax=10)


Author: Kuo-Chuan Pan, NTHU 2022.10.30
For the course, computational physics lab

"""

class Particles:
    """
    
    The Particles class handle all particle properties

    for the N-body simulation. 

    """
    def __init__(self,N:int=100000):
        """
        Prepare memories for N particles

        :param N: number of particles.

        By default: particle properties include:
                nparticles: int. number of particles
                _masses: (N,1) mass of each particle
                _positions:  (N,3) x,y,z positions of each particle
                _velocities:  (N,3) vx, vy, vz velocities of each particle
                _accelerations:  (N,3) ax, ay, az accelerations of each partciel
                _tags:  (N)   tag of each particle
                _time: float. the simulation time 

        """
        self.nparticles = N
        self._time = 0
        self._masses = np.ones((N,1))
        self._positions = np.zeros((N,3))
        self._velocities = np.zeros((N,3))
        self._accelerations = np.zeros((N,3))
        self._tags = np.linspace(1,N,N)
        self._Ek = np.ones((N,1))
        self._Eu = np.ones((N,1))
        return
    """
    @property
    def time(self):
        return self._time
    @time.setter
    def time(self,time):
        self._time = time
        return
    """

    # getter
    def get_time(self):
        return self._time
    
    def get_masses(self):
        return self._masses

    def get_positions(self):
        return self._positions
    
    def get_velocities(self):
        return self._velocities
    
    def get_accelerations(self):
        return self._accelerations
    
    def get_tags(self):
        return self._tags

    def get_kinetic_energy(self):
        return self._Ek

    def get_potential_energy(self):
        return self._Eu

    # setter
    def set_time(self,time):
        self._time = time
        return
    
    def set_masses(self,mass):
        self._masses = mass
        return

    def set_positions(self,pos):
        self._positions = pos
        return

    def set_velocities(self,vel):
        self._velocities = vel
        return
    
    def set_accelerations(self,acc):
        self._accelerations = acc
        return
    
    def set_tags(self,tag):
        self._tags = tag
        return

    def set_kinetic_energy(self,Ek,mode):
        mass = self._masses
        vel = self._velocities
        N = self.nparticles
        # print("wtf = ", vel[:,0].shape)
        # V_square = (vel[:,0]*vel[:,0] + vel[:,1]*vel[:,1] + vel[:,2]*vel[:,2])
        # M = mass[:,0]
        # print("V_square shape = ", V_square.shape)
        # print("M shape = ", M.shape)
        # self._Ek += 0.5*np.multiply(M, V_square)
        # print("MV shape = ", (M*V_square[:,0]).shape)
        # print("kinetic shape = ", self._Ek.shape)
        velx = vel[:,0]
        vely = vel[:,1]
        velz = vel[:,2]
        for i in range(N):
            value = 0.5*mass[i,0]*(velx[i]**2 + vely[i]**2 + velz[i]**2)
            self._Ek[i] += value
        # print("kinetic energy = ", self._Ek)
        return
    
    def set_potential_energy(self,Eu,mode):
        G = 6.67428e-8
        N = self.nparticles
        mass = self._masses
        pos = self._positions
        posx = pos[:,0]
        posy = pos[:,1]
        posz = pos[:,2]
        for i in range(N):
            for j in range(N):
                if (j>i):
                    dx = (posx[i]-posx[j])
                    dy = (posy[i]-posy[j])
                    dz = (posz[i]-posz[j])
                    r = np.sqrt(dx**2 + dy**2 + dz**2)
                    value = 0.5*(-G*mass[i,0]*mass[j,0]/r)
                    #print("value ", value)
                    self._Eu[i] += value
                    self._Eu[j] += value
        # print("potential energy = ", self._Eu)
            # print("potential shape = ", Eu.shape)
            # if mode == "init":
            #     print("SETTING INIT CONDITION, POTENTIAL1 = ", self._Eu)
            #     self._Eu = self._Eu * Eu
            #     print("SETTING INIT CONDITION, POTENTIAL2 = ", self._Eu)
            #     print("INIT CONDITION, POTENTIAL SHAPE = ", self._Eu.shape)
        return   

    def output(self,fn, time):
        """
        Write simulation data into a file named "fn"

        """
        mass = self._masses
        pos  = self._positions
        vel  = self._velocities
        acc  = self._accelerations
        tag  = self._tags
        Ek   = self._Ek
        Eu   = self._Eu
        header = """
                ----------------------------------------------------
                Data from a 3D direct N-body simulation. 

                rows are i-particle; 
                coumns are :mass, tag, x ,y, z, vx, vy, vz, ax, ay, az, Ek, Eu
                ----------------------------------------------------
                """
        header += "Time = {}".format(time)
        # print("EK shape = ", Ek.shape)
        # print("Eu shape = ", Eu.shape)
        np.savetxt(fn,(tag[:],mass[:,0],pos[:,0],pos[:,1],pos[:,2],
                            vel[:,0],vel[:,1],vel[:,2],
                            acc[:,0],acc[:,1],acc[:,2],Ek[:,0],Eu[:,0]),header=header)
        return

# @njit(parallel=True)
# def _potential_energy_kernal(N,posx,posy,posz,G,mass,Eu):
#     for i in prange(N):
#         for j in prange(N):
#             if (j>i):
#                 dx = (posx[i]-posx[j])
#                 dy = (posy[i]-posy[j])
#                 dz = (posz[i]-posz[j])
#                 r = np.sqrt(dx**2 + dy**2 + dz**2)
#                 Eu += 0.5*(-G*mass**2/r)
#     return Eu

class NbodySimulation:
    """
    
    The N-body Simulation class.
    
    """

    def __init__(self,particles:Particles):
        """
        Initialize the N-body simulation with given Particles.

        :param particles: A Particles class.  
        
        """

        # store the particle information
        self.nparticles = particles.nparticles
        self.particles  = particles

        # Store physical information
        self.time  = 0.0  # simulation time

        # Set the default numerical schemes and parameters
        self.setup()
        
        return

    def setup(self, G=1, 
                    rsoft=0.01, 
                    method="RK4", 
                    io_freq=10, 
                    io_title="particles",
                    io_screen=True,
                    visualized=False):
        """
        Customize the simulation enviroments.

        :param G: the graivtational constant
        :param rsoft: float, a soften length, used to aviod numerical issue when 2 particles are close to each other.
        :param meothd: string, the numerical scheme
                       support "Euler", "RK2", and "RK4"

        :param io_freq: int, the frequency to outupt data.
                        io_freq <=0 for no output. 
        :param io_title: the output header
        :param io_screen: print message on screen or not.
        :param visualized: on the fly visualization or not. 
        
        """
        self.G = G
        self.rsoft = rsoft
        self.method = method
        self.io_freq = io_freq
        self.io_title = io_title
        self.io_screen = io_screen
        self.visualized = visualized
        return

    def evolve(self, dt:float=0.01, tmax:float=10):
        """

        Start to evolve the system

        :param dt: time step
        :param tmax: the finial time
        
        """
        method = self.method
        if method=="Euler":
            _update_particles = self._update_particles_euler
        elif method=="RK2":
            _update_particles = self._update_particles_rk2
        elif method=="RK4":
            _update_particles = self._update_particles_rk4
        elif method=="LeapFrog":
            _update_particles = self._update_particles_leapfrog    
        else:
            print("No such update meothd", method)
            quit()
                
        # prepare an output folder for lateron output
        io_folder = "data_"+self.io_title
        Path(io_folder).mkdir(parents=True, exist_ok=True)
        
        # ====================================================
        # The main loop of the simulation
        # =====================================================
        
        time = self.time
        
        # np.ceil: 無條件進位；np.floor: 無條件捨去
        nsteps = np.ceil((tmax-time)/dt)
        
        for n in range(int(nsteps)):
            # print("============================>evolving time = #{}/{}\n".format(n + 1, int(nsteps)))
            # 包含最後一個時間點
            if (time + dt) > tmax:
                dt = tmax - time
            
            # update particles
            particles = self.particles
            _update_particles(dt, particles)


            # print("============================>getting particles Ek\n")
            particles.set_kinetic_energy(particles._Ek, "evolve")
            # print("============================>getting particles Eu\n")
            particles.set_potential_energy(particles._Eu, "evolve")
            # check & visualization
            if (n % self.io_freq == 0):
                if self.io_screen:
                    print("n =", n, "time =", time, "dt =", dt)

                    # visualize
                    if self.visualized:
                        # TODO:
                        pass
                    
                    # output data
                    fn = io_folder + "/data_" + self.io_title + "_" + str(n).zfill(5) + ".txt" # 存取的filename
                    self.particles.output(fn,time)
            
            # update time
            time += dt


        print("Done!")
        return

    def _calculate_acceleration(self, mass, pos):
        """
        Calculate the acceleration.
        """

        posx = pos[:,0]
        posy = pos[:,1]
        posz = pos[:,2]
        G = self.G
        npts = self.nparticles
        acc = np.zeros((npts,3)) # reset acc
        acc = _calculate_acceleration_kernel(mass,posx,posy,posz,acc,G,npts)

        # for i in range(npts):
        #     for j in range(npts):
        #         if (j>i):
        #             x = (posx[i]-posx[j])
        #             y = (posy[i]-posy[j])
        #             z = (posz[i]-posz[j])
        #             rsquare = x**2 + y**2 + z**2
        #             req = np.sqrt(x**2 + y**2)
        #             force = -G*mass[i,0]*mass[j,0]/rsquare
        #             theta = np.arctan2(y,x)
        #             phi = np.arctan2(z,req)
        #             fx = force*np.cos(theta)*np.cos(phi)
        #             fy = force*np.sin(theta)*np.cos(phi)
        #             fz = force*np.sin(phi)
                    
        #             acc[i,0] += fx/mass[i,0]
        #             acc[i,1] += fy/mass[i,0]
        #             acc[i,2] += fz/mass[i,0]

        #             acc[j,0] += -fx/mass[j,0]
        #             acc[j,1] += -fy/mass[j,0]
        #             acc[j,2] += -fz/mass[j,0]
        return acc

    def _update_particles_euler(self, dt, particles:Particles):
        #t1 = time.time()
        mass = particles._masses
        pos = particles._positions
        vel = particles._velocities
        #t2 = time.time()
        acc = self._calculate_acceleration(mass, pos)
        #t3 = time.time()
        pos = pos + dt*vel
        vel = vel + dt*acc
        #t4 = time.time()
        #print("Acc",t3-t2)
        #print("Pos/Vel",t4-t3)
        acc = self._calculate_acceleration(mass, pos)

        particles._positions = pos
        particles._velocities = vel
        particles._accelerations = acc

        #tend = time.time()
        #print("Time in one step = ", tend-t1)
        return particles

    def _update_particles_rk2(self, dt, particles:Particles):
        mass = particles._masses
        pos = particles._positions                          # y0[0]
        vel = particles._velocities                         # y0[1], k1[0]
        acc = self._calculate_acceleration(mass, pos)       # k1[1]

        pos1 = pos + dt*vel                                 # y1[0]
        vel1 = vel + dt*acc                                 # y1[1], k2[0]
        acc1 = self._calculate_acceleration(mass, pos1)     # k2[1]

        pos2 = pos + 0.5*dt*(vel+vel1)
        vel2 = vel + 0.5*dt*(acc+acc1)

        particles._positions = pos2
        particles._velocities = vel2
        
        return particles

    def _update_particles_rk4(self, dt, particles:Particles):
        mass = particles._masses
        pos = particles._positions
        vel = particles._velocities
        acc = self._calculate_acceleration(mass, pos)

        pos1 = pos + 0.5*dt*vel
        vel1 = vel + 0.5*dt*acc
        acc1 = self._calculate_acceleration(mass, pos1)

        pos2 = pos + 0.5*dt*vel1                            # y2[0]
        vel2 = vel + 0.5*dt*acc1                            # y2[1], k3[0]
        acc2 = self._calculate_acceleration(mass, pos2)     # k3[1]

        pos3 = pos + dt*vel2
        vel3 = pos + dt*acc2
        acc3 = self._calculate_acceleration(mass, pos3)

        particles._positions = pos + (1/6)*dt*(vel + 2*vel1 + 2*vel2 + vel3)
        particles._velocities = vel2 + (1/6)*dt*(acc + 2*acc1 + 2*acc2 + acc3)
    
        return particles

    def _update_particles_leapfrog(self, dt, particles:Particles):
        mass = particles._masses
        pos = particles._positions
        vel = particles._velocities
        acc = self._calculate_acceleration(mass, pos)

        vel = vel + 0.5*dt*acc
        pos = pos + dt*vel

        particles._positions = pos
        particles._velocities = vel
        particles._accelerations = acc

        return particles

@njit(parallel=True)
def _calculate_acceleration_kernel(mass,posx,posy,posz,acc,G,npts):
    for i in prange(npts):
            for j in prange(npts):
                if (j>i):
                    x = (posx[i]-posx[j])
                    y = (posy[i]-posy[j])
                    z = (posz[i]-posz[j])
                    rsquare = x**2 + y**2 + z**2
                    req = np.sqrt(x**2 + y**2)
                    force = -G*mass[i,0]*mass[j,0]/rsquare
                    theta = np.arctan2(y,x)
                    phi = np.arctan2(z,req)
                    fx = force*np.cos(theta)*np.cos(phi)
                    fy = force*np.sin(theta)*np.cos(phi)
                    fz = force*np.sin(phi)
                    
                    acc[i,0] += fx/mass[i,0]
                    acc[i,1] += fy/mass[i,0]
                    acc[i,2] += fz/mass[i,0]

                    acc[j,0] += -fx/mass[j,0]
                    acc[j,1] += -fy/mass[j,0]
                    acc[j,2] += -fz/mass[j,0]
    return acc

if __name__=='__main__':

    # test Particles() here
    # particles = Particles(N=100)
    # test NbodySimulation(particles) here
    # sim = NbodySimulation(particles=particles)
    # test setup() and evolve()
    # sim.setup(G=6.67428e-8, method="Euler", io_freq=10, io_screen=True, io_title="solar_2D")
    # sim.evolve(dt=86400,tmax=365*86400)

    print("Done")
