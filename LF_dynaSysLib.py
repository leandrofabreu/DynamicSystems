import numpy as np #numerical computation package
import scipy as sp #library of scientific algorithms
from numba import jit #library to compile some functions

####################################################################################################
# Function to integrate ODE using Runge Kutta - v002 (faster) - intRK2(odeFun, x0, t, args=()) 
def funcaoTeste(x=1):
    return x + np.abs(-32)

####################################################################################################
# Function to integrate ODE using Runge Kutta - v002 (faster) - intRK2(odeFun, x0, t, args=()) 
def intRK2(odeFun, x0, t, args=()):
    """
    Runge-Kutta integration (same parameters as scipy.integrate.odeint)
    
    Inputs:
       - odeFun: (callable) ordinary differential equation, e.g. "x, t: odeFun(x, t)"
       - x0: vector of initial states, ex. "x0 = np.array([1.0, 0.3, 3.1])" for a 3D system
       - t: np.array containing the time stamps, ex "t=np.arange(0, 100, 0.02)"
    
    Outputs:
       - x: array of states, shape (len(t), len(x0))
       - t: array of time stamps, shape (len(t))
    
    IFMG - v002 - Leandro Freitas (dez-2020)
    """
    
    # pre-allocate state vector
    x = np.empty((len(t),len(x0))) #(len(t), len(x0))
    
    @jit(nopython=True)
    def auxFunc(x0, t, x, args):
        # initial state
        x[0, :] = x0

        # integration step
        dt = t[1]-t[0]

        # loop to compute the states
        for k in range(1, len(t)):
            k1F = dt*odeFun(x[k-1, :], t[k], *args)
            k2F = dt*odeFun(x[k-1, :] + k1F/2., t[k], *args)
            k3F = dt*odeFun(x[k-1, :] + k2F/2., t[k], *args)
            k4F = dt*odeFun(x[k-1, :] + k3F, t[k], *args)
            # compute the actual state
            x[k, :] = x[k-1, :] + (k1F+2.*k2F+2.*k3F+k4F)/6.

        return x
    
    return auxFunc(x0, t, x, args)

####################################################################################################
# Time between crossing of Poincaré section - f_timeBetCrossPoin(x, t, f_poin) 
def f_timeBetCrossPoin(x, t, f_poin):
    """
    Estimate time between crossings of a Poincaré section.
      Inputs:
        - x[M,n]: state variables of the system (M: #of points, n: #of states)
        - t[M]: time variable of the system (M: #of points)
        - f_poin(x[k,:], x[k-1,:]): function that receives two consecutive points
          and returns TRUE or FALSE accordding to the passage through the section.
          'TRUE' indicate that the section is between x[k] and x[k-1]
    
      Outputs:
        - timeBetCross[#of crossings-1]: computed time between crossings
        - crossIdx[#of crossings]: indices of crossing (size: #of crossings)
          (ex.: 'x[crossIdx]' shows the crossing points in the variable 'x')
        - numPass[M]: register #of passages through the section for each sample
          (ex.: 'numPass[80]' show how many crossing occured before the 80th sample)
    
    Versions:
        > MACSIN - v000 - LFA jul-2017
        > PRATICAR/IFMG - v001 - LFA dez-2020: vectorized implementation (@jit does not work)
    """
    
    # Passages through the Poincaré section
    passages = f_poin(x[:-1,:], x[1:,:]) # Note: len(passages) = len(x) - 1 (one less sample)
    crossTime = t[1:][passages] # must start in the second sample because shape of "passages"
    
    # compute the time between crossings (time to complete a cycle)
    timeBetCross = crossTime[1:]-crossTime[:-1]
    
    # index of crossings in the trajectory "x" and time "t"
    crossIdx = np.where(passages)[0]+1
    
    # number of passages through the section for each point
    numPass = np.cumsum(passages)
    numPass = np.append(numPass[0], numPass)
    
    return timeBetCross, crossIdx, numPass

####################################################################################################
# Phase calculation (Vector Field Phase) - f_estimateVFP(x, t, kkP1, kkP2, f_ode=None) 
def f_estimateVFP(x, t, kkP1, kkP2, f_ode=None):
    """
    Estimate VFP phase.
    
      Inputs:
        - x[N,2*n]: state variables of the system (N: #of samples, n: #of states)
        - t[N]: time variable of the system (M: #of samples)
        - kkP1: array of indexes that are in the Poincare section of the system (1)
        - kkP2: array of indexes that are in the Poincare section of the system (2)
        - f_ode: function that calculate vector field with 'x' as input argument f_ode(x)
                 if "f_ode" is "None", then the vector field is estimated
            
      Outputs:
        - phi_1, phi_2: phase variables of each subsystem
        - ell1, ell2: computed arc length of each revolution
        - freq1, freq2: frequency of each subsystem
    
    MACSIN - v000 - LFA mar-2018
    PRATICAR/IFMG - v001 - LFA dez-2020: faster implementation
    PRATICAR/IFMG - v002 - LFA fev-2021: bug fix
    """
    
    # interesting variables
    N = np.size(x, 0) #number of samples
    n = int(np.size(x, 1)/2) #order of each subsystem
    
    # Compute the arc-length (perimeter) for each revolution...
    #  - Oscillator (1)
    ell1 = np.empty(N)
    dist1 = np.linalg.norm(x[1:,:n]-x[:-1,:n], axis=1) #distance between each subsequent points
    for k in np.arange(len(kkP1)-1):
        ell1[kkP1[k]:kkP1[k+1]] = np.sum(dist1[kkP1[k]:kkP1[k+1]-1])
    # complete missing lengths
    ell1[:kkP1[0]]=ell1[kkP1[0]]; ell1[kkP1[-1]:]=ell1[kkP1[-1]-1];

    #  - Oscillator (2)
    ell2 = np.empty(N)
    dist2 = np.linalg.norm(x[1:,n:]-x[:-1,n:], axis=1) #distance between each subsequent points
    for k in np.arange(len(kkP2)-1):
        ell2[kkP2[k]:kkP2[k+1]] = np.sum(dist2[kkP2[k]:kkP2[k+1]-1])
    # complete missing lengths
    ell2[:kkP2[0]]=ell2[kkP2[0]]; ell2[kkP2[-1]:]=ell2[kkP2[-1]-1];
    
    # Compute the instantaneous frequency
    if f_ode is not None:
        # when vector field is available
        f = np.empty((N, 2*n))
        dt = np.mean(t[1:]-t[:-1])
        for k in np.arange(N): # len(k) => 0 to N-1
            f[k,:] = f_ode(x[k,:]) #using a "for loop" because "f_ode" is not vectorized...
        
        # Oscillator (1)
        freq1 = ( (2*np.pi)/ell1 ) * np.linalg.norm(f[:,:n], axis=1)
        f1 = freq1*np.append(dt, t[1:]-t[:-1]) #numerical integration (zero-order hold)
        
        # Oscillator (2)
        freq2 = ( (2*np.pi)/ell2 ) * np.linalg.norm(f[:,n:], axis=1)
        f2 = freq2*np.append(dt, t[1:]-t[:-1]) #numerical integration (zero-order hold)
        
    else:
        # when vector field is NOT available
        dt = np.mean(t[1:]-t[:-1])
        
        # Oscillator (1)
        f1 = ( (2*np.pi)/ell1[1:] ) * np.linalg.norm(x[1:,:n]-x[:-1,:n], axis=1) #estimate vector field
        f1 = np.append(f1[0], f1) #copy first element to keep len(f1) = N
        freq1 = f1/np.append(dt, t[1:]-t[:-1]) #instantaneous frequency
        
        # Oscillator (2)
        f2 = ( (2*np.pi)/ell2[1:] ) * np.linalg.norm(x[1:,n:]-x[:-1,n:], axis=1) #estimate vector field
        f2 = np.append(f2[0], f2) #copy first element to keep len(f2) = N
        freq2 = f2/np.append(dt, t[1:]-t[:-1]) #instantaneous frequency
    
    # Compute the VECTOR FIELD PHASE
    #   - Oscillator 1
    phi_1 = np.cumsum(f1)-f1[0]
    
    #   - Oscillator 2
    phi_2 = np.cumsum(f2)-f2[0]
    
    # return phase values (phi_1, phi_2), perimeters (ell1, ell2) and frequency (freq1, freq2)
    return phi_1, phi_2, ell1, ell2, freq1, freq2

####################################################################################################
# VFP calculation for 1 oscillator alone - f_estimateVFP_1(x, t, kkP1, f_ode=None) 
def f_estimateVFP_1(x, t, kkP1, f_ode=None):
    """
    Estimate VFP phase.
    
      Inputs:
        - x[N,n]: state variables of the system (N: #of samples, n: #of states)
        - t[N]: time variable of the system (M: #of samples)
        - kkP1: array of indexes that are in the Poincare section of the system
        - f_ode: function that calculate vector field with 'x' as input argument f_ode(x)
                 if "f_ode" is "None", then the vector field is estimated
            
      Outputs:
        - phi_1: phase variables of the system
        - ell1: computed arc length of each revolution
        - freq1: frequency of the system
    
    PRATICAR/IFMG - v000 - LFA jan-2021: initial version
    PRATICAR/IFMG - v001 - LFA fev-2021: bug fix
    """
    
    # interesting variables
    N = np.size(x, 0) #number of samples
    n = int(np.size(x, 1)) #order of the system
    
    # Compute the arc-length (perimeter) for each revolution...
    ell1 = np.empty(N)
    dist1 = np.linalg.norm(x[1:,:n]-x[:-1,:n], axis=1) #distance between each subsequent points
    for k in np.arange(len(kkP1)-1):
        ell1[kkP1[k]:kkP1[k+1]] = np.sum(dist1[kkP1[k]:kkP1[k+1]-1])
    # complete missing lengths
    ell1[:kkP1[0]]=ell1[kkP1[0]]; ell1[kkP1[-1]:]=ell1[kkP1[-1]-1];

    # Compute the instantaneous frequency
    if f_ode is not None:
        # when vector field is available
        f = np.empty((N, n))
        dt = np.mean(t[1:]-t[:-1])
        for k in np.arange(N): # len(k) => 0 to N-1
            f[k,:] = f_ode(x[k,:]) #using a "for loop" because "f_ode" is not vectorized...
        
        # Oscillator frequency
        freq1 = ( (2*np.pi)/ell1 ) * np.linalg.norm(f[:,:n], axis=1)
        f1 = freq1*np.append(dt, t[1:]-t[:-1]) #numerical integration (zero-order hold)
        
    else:
        # when vector field is NOT available
        dt = np.mean(t[1:]-t[:-1])
        
        # Oscillator frequency
        f1 = ( (2*np.pi)/ell1[1:] ) * np.linalg.norm(x[1:,:n]-x[:-1,:n], axis=1) #estimate vector field
        f1 = np.append(f1[0], f1) #copy first element to keep len(f1) = N
        freq1 = f1/np.append(dt, t[1:]-t[:-1]) #instantaneous frequency
        
    # Compute the VECTOR FIELD PHASE
    phi_1 = np.cumsum(f1)-f1[0]
    
    # return phase values (phi_1), perimeters (ell1) and frequency (freq1)
    return phi_1, ell1, freq1

####################################################################################################


