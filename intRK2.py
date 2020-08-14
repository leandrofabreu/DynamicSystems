# Function to integrate ODE using Runge Kutta - v002 (faster)
def intRK2(odeFun, x0, t, args):
    """
    Runge-Kutta integration (same parameters as scipy.integrate.odeint)
    
    Inputs:
       - odeFun: (callable) ordinary differential equation, e.g. "x, t: odeFun(x, t)"
       - x0: vector of initial states, ex. "x0 = np.array([1.0, 0.3, 3.1])" for a 3D system
       - t: np.array containing the time stamps, ex "t=np.arange(0, 100, 0.02)"
    
    Outputs:
       - x: array of states, shape (len(t), len(x0))
       - t: array of time stamps, shape (len(t))
    
    MACSIN - v001 - Leandro Freitas (dez-2017)
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