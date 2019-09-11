#!/usr/bin/env python
# coding: utf-8

# # Requirements

# In[1]:


# !pip install tqdm

import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from tqdm import tqdm


# # Exercises

# ## Solve Exercise 3.4. Explain how you obtained the table. Your solution may be hand-written.

# Let $R_{search}$ be the rv of reward when the robot is searching. <br>
# $P_{R_{search}}(r) = P[R_{search}= r]$ <br> 
# <br>
# Let $R_{wait}$ be the rv of reward when the robot is waiting. <br>
# $P_{R_{wait}}(r) = P[R_{wait}= r]$ <br> 
# <br>
# Let there distribution be s.t. <br>
# $E[R_{search}] = r_{search}$ <br>
# $E[R_{wait}] = r_{wait}$ <br>
# <br>
# Then, <br>
# 
# 
# | s    | a        | s'   | r  | p(s',r\|s,a)                       | 
# |:-----|:---------|:-----|:---|:----------------------------------:|
# | high | search   | high | r  | $\alpha$ . $P_{R_{search}}(r)$     |
# | high | search   | low  | r  | $(1-\alpha)$ . $P_{R_{search}}(r)$ | 
# | high | wait     | high | r  | $P_{R_{wait}}(r)$                  | 
# | low  | search   | high | -3 | $(1-\beta)$                        |
# | low  | search   | low  | r  | $\beta$ . $P_{R_{search}}(r)$      |
# | low  | wait     | low  | r  | $P_{R_{wait}}(r)$                  |
# | low  | recharge | high | 0  |  1.0                               |
# 

# ## Write code that solves the linear equations required to find v$_π$(s) and generate the values in the table in Figure 3.2. Note that the policy π picks all valid actions in a state with equal probability. Add comments to your code that explain all your steps.

# In[9]:


#Grid size
n = 5
#Number of states
n_s = n*n
#Number of actions
n_a = 4
#Position A
A_r, A_c = 0, 1
#Position A'
A1_r, A1_c = 4, 1
#Position B
B_r, B_c = 0, 3
#Position B'
B1_r, B1_c = 2, 3
#discount
gamma = 0.9

#Direction convention => 0:East, 1:North, 2:West, 3:South
#Policy: First 2 coordinates signify state and the third signifies action
#All actions equi-probable in all states
policy = np.ones((n, n, n_a)) / n_a

#Transition Fucntion: given current state and action taken returns new state(s1_c, s1_r) and reward(r) earned
def transit(s_r, s_c, a):
    s1_r, s1_c, r = -10, -10, -10
    #Position A: leads to A' for all actions with +10 reward
    if s_r == A_r and s_c == A_c:
        s1_r, s1_c = A1_r, A1_c
        r = 10
    #Position B: leads to B' for all actions with +5 reward    
    elif s_r == B_r and s_c == B_c:
        s1_r, s1_c = B1_r, B1_c
        r = 5
    #East Boundary: going East -1 reward    
    elif s_c == 0 and a == 0:
        s1_r, s1_c = s_r, s_c
        r = -1
    #North Boundary: going North -1 reward    
    elif s_r == 0 and a == 1:
        s1_r, s1_c = s_r, s_c
        r = -1
    #West Boundary: going West -1 reward    
    elif s_c == n-1 and a == 2:
        s1_r, s1_c = s_r, s_c
        r = -1
    #South Boundary: going South -1 reward    
    elif s_r == n-1 and a == 3:
        s1_r, s1_c = s_r, s_c
        r = -1
    #Staying inside the grid with 0 reward
    else:
        #Going East
        if a == 0:
            s1_r, s1_c = s_r, s_c - 1
        #Going North
        elif a == 1:
            s1_r, s1_c = s_r - 1, s_c
        #Going West
        elif a == 2:
            s1_r, s1_c = s_r, s_c + 1
        #Going South
        else:
            s1_r, s1_c = s_r + 1, s_c
        #Gaining 0 reward
        r = 0
    
    return s1_r, s1_c, r

#An equation for each state
eqn = np.zeros((n_s, n_s))
#Constant for each equation
c = np.zeros(n_s)

#Row-Major Style
for s_r in range(n):
    for s_c in range(n):
        #co-efficients of all v_pi(s')
        weights = np.zeros((n, n))
        #co-efficient of v_pi(s)
        weights[s_r, s_c] = -1
        #scalar additives in equation
        scals = 0
        
        for a in range(n_a):
            #Policy realisation
            poly = policy[s_r, s_c, a]
            #New states (s')
            s1_r, s1_c, r = transit(s_r, s_c, a)
            #co-efficient of v_pi(s')
            weights[s1_r, s1_c] += poly * gamma
            #scalars
            scals +=  poly * r
        #Storing equations
        eqn[s_r*n + s_c] = weights.flatten(order='C')
        c[s_r*n + s_c] = -scals
        
v_pi = npla.solve(eqn, c).reshape((n, n))


# In[10]:


print('Value Function')
print(np.round(v_pi, decimals=2))


# ## Solve Exercises 3.15 and 3.16.

# $v_\pi(s) = E_\pi[G_t | S_t=s]$ <br>

# ### Solve Exercises 3.15

# **Continuous** <br>
# $v_\pi(s) = E_\pi[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} .... | S_t=s]$ <br>
# <br>
# Adding c to all rewards<br>
# <br>
# $v^c_\pi(s)  = E_\pi[(R_{t+1}+c) + \gamma (R_{t+2}+c) + \gamma^2 (R_{t+3}+c) .... | S_t=s]$ <br>
# $\quad  = E_\pi[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} .... | S_t=s] + E_\pi[c + \gamma c + \gamma^2 c .... | S_t=s]$ <br><br>
# $\quad  = v_\pi(s) + \frac{c}{1-\gamma}$ <br>
# <br>
# The value function doesn't change relatively for states. Each of them have same scalar added to them.

# ### Solve Exercises 3.16

# **Episodic** <br>
# $v_\pi(s) = E_\pi[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... + \gamma^{n-1} R_{t+n} | S_t=s]$ <br>
# <br>
# Adding c to all rewards<br>
# <br>
# $v^c_\pi(s)  = E_\pi[(R_{t+1}+c) + \gamma (R_{t+2}+c) + \gamma^2 (R_{t+3}+c) + ... + \gamma^{n-1} (R_{t+n}+c) | S_t=s]$ <br>
# $\quad  = E_\pi[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... + \gamma^{n-1} R_{t+n} | S_t=s] + E_\pi[c + \gamma c + \gamma^2 c + ... + \gamma^{n-1} c | S_t=s]$ <br><br>
# $\quad  = v_\pi(s) + c \frac{1-\gamma^{n}}{1-\gamma}$ <br>
# <br>
# The value function does change relatively for states. Now it depends upon the time($n$) after which the episode ends when started in state $s$. It increases as n increases. And as the time after which the episode ends for different start states, the value of the additive term changes and hence the value functions of the term aren't relatively same as before.  

# ## Write code that generates the optimal state-value function and the optimal policy for the Gridworld in Figure 3.5. You want to solve the corresponding system of non-linear equations. Explain all your steps.

# In[19]:


#Grid size
n = 5
#Number of states
n_s = n*n
#Number of actions
n_a = 4
#Position A
A_r, A_c = 0, 1
#Position A'
A1_r, A1_c = 4, 1
#Position B
B_r, B_c = 0, 3
#Position B'
B1_r, B1_c = 2, 3
#discount
gamma = 0.9

#Direction convention => 0:East, 1:North, 2:West, 3:South

#Transition Fucntion: given current state and action taken returns new state(s1_c, s1_r) and reward(r) earned
def transit(s_r, s_c, a):
    s1_r, s1_c, r = -10, -10, -10
    #Position A: leads to A' for all actions with +10 reward
    if s_r == A_r and s_c == A_c:
        s1_r, s1_c = A1_r, A1_c
        r = 10
    #Position B: leads to B' for all actions with +5 reward    
    elif s_r == B_r and s_c == B_c:
        s1_r, s1_c = B1_r, B1_c
        r = 5
    #East Boundary: going East -1 reward    
    elif s_c == 0 and a == 0:
        s1_r, s1_c = s_r, s_c
        r = -1
    #North Boundary: going North -1 reward    
    elif s_r == 0 and a == 1:
        s1_r, s1_c = s_r, s_c
        r = -1
    #West Boundary: going West -1 reward    
    elif s_c == n-1 and a == 2:
        s1_r, s1_c = s_r, s_c
        r = -1
    #South Boundary: going South -1 reward    
    elif s_r == n-1 and a == 3:
        s1_r, s1_c = s_r, s_c
        r = -1
    #Staying inside the grid with 0 reward
    else:
        #Going East
        if a == 0:
            s1_r, s1_c = s_r, s_c - 1
        #Going North
        elif a == 1:
            s1_r, s1_c = s_r - 1, s_c
        #Going West
        elif a == 2:
            s1_r, s1_c = s_r, s_c + 1
        #Going South
        else:
            s1_r, s1_c = s_r + 1, s_c
        #Gaining 0 reward
        r = 0
    
    return s1_r, s1_c, r

def equations(v):
    v = v.reshape((n, n))
    eqn = np.zeros((n, n))
    
    for s_r in range(n):
        for s_c in range(n):
            #exected returns on all actions
            g = np.zeros(n_a)
            for a in range(n_a):
                #New states (s')
                s1_r, s1_c, r = transit(s_r, s_c, a)
                #exected return
                g[a] = r + gamma*v[s1_r, s1_c]
            eqn[s_r, s_c] = v[s_r, s_c] - np.amax(g)
    
    return eqn.flatten(order='F')

def optimal_policy(v):
    pi = np.zeros((n, n))
    #Building optimal policy
    for s_r in range(n):
        for s_c in range(n):
            #Expected Return for each action
            g = np.zeros(n_a)
            for a in range(n_a):
                #new state and reward
                s1_r, s1_c, r = transit(s_r, s_c, a)
                #expected return
                g[a] = r + (gamma * v[s1_r, s1_c])

            #Policy Improvement
            #Update value function
            pi[s_r, s_c] = np.argmax(g)

    return pi


# In[20]:


v = np.zeros(n*n)
v_star = fsolve(equations, v).reshape(n, n)
pi_star = optimal_policy(v_star)
print('Optimal Value Function')
print(np.round(v_star, decimals=2))
print()
print('Optimal Policy')
print(pi_star)


# ## Given an equation for v∗ in terms of q∗.

# $q_*(s, a) = \max\limits_\pi q_\pi(s, a)$ <br>
# $v_*(s) = \max\limits_{a\in A(s)} \max\limits_\pi q_\pi(s, a)$ <br>
# $v_*(s) = \max\limits_{a\in A(s)} q_*(s, a)$ <br>

# ## Code policy iteration and value iteration (VI) to solve the Gridworld in Example 4.1. Your code must log output of each iteration. Pick up a few sample iterations to show policy evaluation and improvement at work. Similarly, show using a few obtained iterations that every iteration of VI improves the value function. Your code must include the fix to the bug mentioned in Exercise 4.4.

# In[2]:


#grid Size
n = 4
#No. of states
n_s = (n*n) - 2
#Terminal State
st_r, st_c = 0, 0
#No. of actions
n_a = 4
#discount
gamma = 1
#convergence
theta = 1e-10

#Actions => 0:left, 1:up, 2:right, 3:down
def transit_grid(s_r, s_c, a):
    s1_r = -1
    s1_c = -1
    #Reward
    r = -1
    
    #Terminal State
    if s_r==0 and s_c==1 and a==0:
        s1_r, s1_c = st_r, st_c
    elif s_r==1 and s_c==0 and a==1:
        s1_r, s1_c = st_r, st_c
    elif s_r==n-1 and s_c==n-2 and a==2:
        s1_r, s1_c = st_r, st_c
    elif s_r==n-2 and s_c==n-1 and a==3:
        s1_r, s1_c = st_r, st_c
        
    #Going outside the left boundary
    elif s_c==0 and a==0:
        s1_r, s1_c = s_r, s_c
    #Going outside the top boundary
    elif s_r==0 and a==1:
        s1_r, s1_c = s_r, s_c
    #Going outside the right boundary
    elif s_c==n-1 and a==2:
        s1_r, s1_c = s_r, s_c
    #Going outside the botton boundary
    elif s_r==n-1 and a==3:
        s1_r, s1_c = s_r, s_c
    
    #Move left
    elif a==0:
        s1_r, s1_c = s_r, s_c-1
    #Move up
    elif a==1:
        s1_r, s1_c = s_r-1, s_c
    #Move right
    elif a==2:
        s1_r, s1_c = s_r, s_c+1
    #Move down
    elif a==3:
        s1_r, s1_c = s_r+1, s_c

    return s1_r, s1_c, r


# ### Policy Iteration

# The bug mentioned in Exercise 4.4 is dealt by using numpy.argamx() which internally handles it, as it has follows a consistent convention when selecting between equal values. It always selects the one that has a lower index and since indexes of actions don't change in code, therefor it can't oscillate between equally favaourable policies as it will always choose the action with a lower index.

# In[10]:


def policy_iteration(n_1, n_2, n_a, theta, gamma, transit):
    
    #Optimal Value Function
    v = np.zeros((n_1, n_2))
    #Optimal Policy
    pi = np.ones((n_1, n_2, n_a))/n_a

    i = 0
    while True:
        print('Iteration:', i)
        #Policy Evaluation
        print('Policy Evaluation')
        while True:
            #Improvement Measure
            delta = 0
            for s_r in range(n_1):
                for s_c in range(n_2):
                    #avoiding Terminal state
                    if (s_r==0 and s_c==0) or (s_r==n_1-1 and s_c==n_2-1):
                        continue
                    #current Value function
                    v_old = v[s_r, s_c]
                    v_ = 0
                    for a in range(n_a):    
                        #new state and reward
                        s1_r, s1_c, r = transit(s_r, s_c, a)
                        #Expected Return
                        print('s', s_r, s_c, ', a', a-5, 's1', s1_r, s1_c)
                        v_ += pi[s_r, s_c, a] * (r + (gamma * v[s1_r, s1_c]))
                    #update Value function
                    v[s_r, s_c] = v_
                    #max-improvement
                    delta = max(delta, abs(v_old - v[s_r, s_c]))
            print('del', delta)
            if delta < theta:
                print('delta < theta')
                break
        print('Value Function')
        print(v)

        #Policy Improvement
        print('Policy Improvement')
        stable_poly = True
        for s_r in range(n_1):
            for s_c in range(n_2):
                #avoiding Terminal state
                if (s_r==0 and s_c==0) or (s_r==n_1-1 and s_c==n_2-1):
                    continue
                #action from old policy
                a_old = np.random.choice(n_a, p=pi[s_r, s_c])
                #exected returns on all actions
                g = np.zeros(n_a)
                for a in range(n_a):
                    #new state and reward
                    s1_r, s1_c, r = transit(s_r, s_c, a)
                    #expected return
                    g[a] = r + (gamma * v[s1_r, s1_c])            
                #best action
                a_star = np.argmax(g)
                #update policy
                pi[s_r, s_c] *= 0.
                pi[s_r, s_c, a_star] = 1.
                #stability check
                if a_old != a_star:
                    stable_poly = False

        print('Policy')
        print(np.argmax(pi, axis=2))

        if stable_poly:
            print()
            print('Policy Stable')
            break
        i+=1
        print()

    return v, pi


# In[32]:


v_star, pi_star = policy_iteration(n_1=n, n_2=n, n_a=n_a, theta=theta, gamma=gamma, transit=transit_grid)
print('v_star')
print(v_star)
print()
print('pi_star')
print(np.argmax(pi_star, axis=2))


# ### Value Iteration

# In[5]:


#Optimal Value Function
v = np.zeros((n, n))
#Optimal Policy
pi = np.zeros((n, n))
#precision
theta = 1e-10

i=0
while True:
    print('Iteration:', i)
    #Improvement measure
    delta = 0
    
    for s_r in range(n):
        for s_c in range(n):
            
            #avoiding Terminal state
            if (s_r==0 and s_c==0) or (s_r==n-1 and s_c==n-1):
                continue
            
            #current Value function
            v_old = v[s_r, s_c]

            #Policy Evaluation
            #Expected Return for each action
            g = np.zeros(n_a)
            for a in range(n_a):
                #new state and reward
                s1_r, s1_c, r = transit_grid(s_r, s_c, a)
                #expected return
                g[a] = r + (gamma * v[s1_r, s1_c])
            
            #Policy Improvement
            #Update value function
            v[s_r, s_c] = np.amax(g)
            #max-improvement
            delta = max(delta, abs(v_old - v[s_r, s_c]))
    
    print('Value Function')
    print(v)

    if delta < theta:
        break
    
    i+=1
    print()
        
#Building optimal policy
for s_r in range(n):
    for s_c in range(n):
        
        #avoiding Terminal state
        if (s_r==0 and s_c==0) or (s_r==n-1 and s_c==n-1):
            continue
        
        #Expected Return for each action
        g = np.zeros(n_a)
        for a in range(n_a):
            #new state and reward
            s1_r, s1_c, r = transit_grid(s_r, s_c, a)
            #expected return
            g[a] = r + (gamma * v[s1_r, s1_c])

        #Policy Improvement
        #Update value function
        pi[s_r, s_c] = np.argmax(g)


# In[6]:


print('v_star')
print(v)
print()
print('pi_star')
print(pi)


# ## Code exercise 4.7.

# ### Example 4.2

# In[11]:


#max cars allowed 
n_1, n_2 = 21, 21

#Expectations
lren_1 = 3
lren_2 = 4
lret_1 = 3
lret_2 = 2

#Rewards
r_mov = -2
r_ren = 10

#number of actions: -5 to 5 cars moved (a to b is positive)
n_a = 11

#discount
gamma = 0.9

#convergence
that = 0.1

def transit_jack(s_1, s_2, a):
    
    ns_1, ns_2, r = s_1, s_2, 0
    
    #If none of the 2 stations are closed cars can be moved 
    if s_1 !=0 and s_2!=0:   
        #number of cars to move
        c = a-5
        #cars moved
        c = np.clip(c, s_1-(n_1-1), s_1)
        c = np.clip(c, -s_2, n_2-1-s_2)
        ns_1, ns_2 = s_1-c, s_1+c
        #reward
        r = abs(c)*r_mov
        print('moved', c)
    
    #cars returned
    if s_1 != 0:
        ret_1 = min(n_1-1-ns_1, np.random.poisson(lret_1))
        ns_1 += ret_1
        print('return_1', ret_1)
    
    if s_2 != 0:
        ret_2 = min(n_2-1-ns_2, np.random.poisson(lret_2))
        ns_2 += ret_2
        print('return_2', ret_2)

    #cars rented
    if s_1 != 0:
        ren_1 = min(ns_1, np.random.poisson(lren_1))
        ns_1 -= ren_1
        r += r_ren * ren_1
        print('rented_1', ren_1)
    
    if s_2 != 0:
        ren_2 = min(ns_2, np.random.poisson(lren_2))
        ns_2 -= ren_2
        r += r_ren * ren_2
        print('rented_2', ren_2)
            
    return ns_1, ns_2, r


# In[12]:


v_star, pi_star = policy_iteration(n_1=n_1, n_2=n_2, n_a=n_a, theta=theta, gamma=gamma, transit=transit_jack)
print('v_star')
print(v_star)
print()
print('pi_star')
print(np.argmax(pi_star, axis=2))


# In[ ]:




