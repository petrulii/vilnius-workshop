import matplotlib.pyplot as plt
import numpy as np
import timeit
from scipy.optimize import line_search

"""
FIRST ORDER ALGORITHMS

For minimizing a differentiable function $f:\mathbb{R}^n \to \mathbb{R}$, given:
* the function to minimize `f`
* a 1st order oracle `f_grad` (see `problem1.ipynb` for instance)
* an initialization point `x0`
* the sought precision `PREC` 
* a maximal number of iterations `ITE_MAX` 
 
these algorithms perform iterations of the form
$$ x_{k+1} = x_k - \gamma_k \nabla f(x_k) $$
where $\gamma_k$ is a stepsize to choose.
"""

def gradient_algorithm(f, f_grad, x0, step, PREC, ITE_MAX):
    """ Constant stepsize gradient algorithm """
    x = np.copy(x0)
    stop = PREC*np.linalg.norm(f_grad(x0) )
    x_iterates = np.copy(x)
    print("------------------------------------\n Constant Stepsize gradient\n------------------------------------\nSTART    -- stepsize = {:0}".format(step))
    t_s =  timeit.default_timer()
    for k in range(ITE_MAX):
        # Get the gradient of f wrt. x
        g = f_grad(x)
        # Perform a step
        x = x # !!! TO FILL !!!
        # Save the current iterate to the list of iterates
        x_iterates = np.vstack((x_iterates,x))
        # Check for the stopping criteria
        if np.linalg.norm(g) < stop:
            break
    t_e =  timeit.default_timer()
    print("FINISHED -- {:d} iterations / {:.6f}s -- final value: {:f}\n\n".format(k,t_e-t_s,f(x)))
    return x,x_iterates


def gradient_adaptive_algorithm(f, f_grad, x0, PREC, ITE_MAX, step=0.1):
    """ Adaptive stepsize gradient algorithm """
    x = np.copy(x0)
    stop = PREC*np.linalg.norm(f_grad(x0) )
    x_iterates = np.copy(x)
    print("------------------------------------\nAdaptative Stepsize gradient\n------------------------------------\nSTART    -- stepsize = {:0}".format(step))
    t_s =  timeit.default_timer()
    for k in range(ITE_MAX):
        x_prev = np.copy(x)
        # Get the gradient of f wrt. x
        g = f_grad(x)
        # Perform a step
        x = x # !!! TO FILL !!!
        # Find the adaptive stepsize
        if f(x)>f(x_prev):
            x = np.copy(x_prev)
            step = step/2
            print("stepsize: = {:0}".format(step))
        # Save the current iterate to the list of iterates
        x_iterates = np.vstack((x_iterates,x))
        # Check for the stopping criteria
        if np.linalg.norm(g) < stop:
            break
    t_e =  timeit.default_timer()
    print("FINISHED -- {:d} iterations / {:.6f}s -- final value: {:f}\n\n".format(k,t_e-t_s,f(x)))
    return x,x_iterates


def gradient_Wolfe(f, f_grad, x0, PREC, ITE_MAX):
    """ Wolfe Line search """
    x = np.copy(x0)
    stop = PREC*np.linalg.norm(f_grad(x0) )
    x_iterates = np.copy(x)
    print("------------------------------------\n Gradient with Wolfe line search\n------------------------------------\nSTART")
    t_s =  timeit.default_timer()
    for k in range(ITE_MAX):
        # Get the gradient of f wrt. x
        g = f_grad(x)
        # Find the stepsize using the linesearch procedure
        res = line_search(f, f_grad, x, -g, gfk=None, old_fval=None, old_old_fval=None, args=(), c1=0.0001, c2=0.9, amax=50)
        # Perform a step
        x = x - res[0]*g
        # Save the current iterate to the list of iterates
        x_iterates = np.vstack((x_iterates,x))
        # Check for the stopping criteria
        if np.linalg.norm(g) < stop:
            break
    t_e =  timeit.default_timer()
    print("FINISHED -- {:d} iterations / {:.6f}s -- final value: {:f}\n\n".format(k,t_e-t_s,f(x)))
    return x,x_iterates


def fast_gradient_algorithm(f, f_grad, x0, step, PREC, ITE_MAX):
    """ Nesterov's fast gradient algorithm """
    x = np.copy(x0)
    y = np.copy(x0)
    # Initialize lambda_0 = 0
    lbd = 0
    stop = PREC*np.linalg.norm(f_grad(x0))
    x_iterates = np.copy(x)
    print("------------------------------------\n Fast gradient\n------------------------------------\nSTART    -- stepsize = {:0}".format(step))
    t_s =  timeit.default_timer()
    for k in range(ITE_MAX):
        # Perform a step
        x = x # !!! TO FILL !!!
        # Save the current iterate to the list of iterates
        x_iterates = np.vstack((x_iterates,x))
        # Check for the stopping criteria
        if np.linalg.norm(g) < stop:
            break
    t_e =  timeit.default_timer()
    print("FINISHED -- {:d} iterations / {:.6f}s -- final value: {:f}\n\n".format(k,t_e-t_s,f(x)))
    return x,x_iterates


"""
SECOND ORDER NEWTON ALGORITHM
 
For minimizing a *twice* differentiable function $f:\mathbb{R}^n \to \mathbb{R}$, given:
* the function to minimize `f`
* a 2nd order oracle `f_grad_hessian` (see `problem1.ipynb` for instance)
* an initialization point `x0`
* the sought precision `PREC` 
* a maximal number of iterations `ITE_MAX` 
 
these algorithms perform iterations of the form
$$ x_{k+1} = x_k - [\nabla^2 f(x_k) ]^{-1} \nabla f(x_k) .$$
"""

def newton_algorithm(f, f_grad_hessian, x0, PREC, ITE_MAX):
    """ Newton algorithm """
    x = np.copy(x0)
    g0,H0 = f_grad_hessian(x0)
    stop = PREC*np.linalg.norm(g0)
    x_iterates = np.copy(x)
    print("------------------------------------\nNewton's algorithm\n------------------------------------\nSTART")
    t_s =  timeit.default_timer()
    for k in range(ITE_MAX):
        # Get the Hessian of f wrt. x
        g,H = f_grad_hessian(x)
        # Perform a step
        x = x # !!! TO FILL !!!
        # Save the current iterate to the list of iterates
        x_iterates = np.vstack((x_iterates,x))
        # Check for the stopping criteria
        if np.linalg.norm(g) < stop:
            break
    t_e =  timeit.default_timer()
    print("FINISHED -- {:d} iterations / {:.6f}s -- final value: {:f}\n\n".format(k,t_e-t_s,f(x)))
    return x,x_iterates


"""
QUASI NEWTON ALGORITHMS
 
**BFGS.** (Broyden-Fletcher-Goldfarb-Shanno, 1970) The popular BFGS algorithm consist in performing the following iteration

$$ x_{k+1}=x_k - \gamma_k W_k \nabla f(x_k)$$

where $\gamma_k$ is given by Wolfe's line-search and positive definite matrix $W_k$ is computed as
$$ W_{k+1}=W_k - \frac{s_k y_k^T W_k+W_k y_k s_k^T}{y_k^T s_k} +\left[1+\frac{y_k^T W_k y_k}{y_k^T s_k}\right]\frac{s_k s_k^T}{y_k^T s_k} $$
with $s_k=x_{k+1}-x_{k}$ and $y_k=\nabla f(x_{k+1}) - \nabla f(x_{k})$.
"""

def bfgs(f, f_grad, x0, PREC, ITE_MAX):
    """ BFGS """
    x = np.copy(x0)
    n = x0.size
    g =  f_grad(x0)
    sim_eval = 1
    stop = PREC*np.linalg.norm( g )
    W = np.eye(n)
    x_iterates = np.copy(x)
    print("------------------------------------\n BFGS\n------------------------------------\nSTART")
    t_s =  timeit.default_timer()
    for k in range(ITE_MAX): 
        prev_x = x
        d = - W @ g
        res = line_search(f, f_grad, x, d, g)
        step = res[0]
        # Perform a step
        x = x + step * d
        # Compute W
        prev_g = g
        g = f_grad(x)
        s = x - prev_x
        y = g - prev_g
        syTW = np.outer(s, y) @ W 
        sTy = np.dot(s, y) + 1e-6
        W = (sTy * W - (syTW + syTW.T) + ((1 + np.linalg.multi_dot((y, W, y))/sTy) * np.outer(s, s))) / sTy
        # Save the current iterate to the list of iterates
        x_iterates = np.vstack((x_iterates,x))
        # Check for the stopping criteria
        if np.linalg.norm(g) < stop:
            break
    t_e =  timeit.default_timer()
    print("FINISHED -- {:d} iterations / {:.6f}s -- final value: {:f}\n\n".format(k,t_e-t_s,f(x)))
    return x,x_iterates




""" Functions to plot the iterations of the algorithms """

def compare_loss(pb, nb_comparisons, x_iterates, labels):
    """ Plotting the iterates of two algorithms
    param pb : logistic regression problem class
    param nb_comparisons : how many algorithms will be plotted
    param x_iterates : a list of lists of iterates of the algorithms to be compared
    param labels : a list of names of the algorithms to be compared
    """
    # Plot the value of f wrt to x
    plt.figure(figsize=(3,3))
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim(0, 0.75)
    for k in range(nb_comparisons):
        F = []
        for i in range(x_iterates[k].shape[0]):
            F.append(pb.f(x_iterates[k][i]))
        plt.plot(F, label=labels[k])
    plt.legend()
    plt.show()
    # Plot the norms of the gradient of f wrt to x
    plt.figure(figsize=(3,3))
    plt.xlabel('$x$')
    plt.ylabel(''r'\nabla f(x)$')
    for k in range(nb_comparisons):
        GF = []
        for i in range(x_iterates[k].shape[0]):
            GF.append( np.linalg.norm(pb.f_grad(x_iterates[k][i])))
        plt.plot(GF, label=labels[k])
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.show()