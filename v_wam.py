import numpy as np
import hidden_markov as hm

def vstate(pos,len):
    pie = np.zeros(len)
    pie[pos] = 1
    return pie

def Viterbi(M,Z,T,s0):
    t = Z.shape[0]
    q = s0.shape[0]
    delt = np.zeros((t,q))
    pre = np.zeros((t,q))
    #initialization
    delt[0,:] = np.matmul(s0,M[:,int(Z[0]-1)])
    pre[0,:] = -np.ones(q)

    for i in range(1,t):
        for j in range(0,q):
            delt[i,j] = np.max(np.multiply(delt[i-1,:],T[:,j]))*M[j,int(Z[i]-1)]
            pre[i,j] = np.argmax(np.multiply(delt[i-1,:],T[:,j]))
    s_t = np.argmax(delt[-1,:])+1
    path = np.zeros(t)
    path[-1] = s_t
    for k in range(t-2,-1,-1):
        path[k] = pre[k+1,int(path[k+1]-1)]+1
    p = delt[-1,s_t]
    return path,p

if __name__=="__main__":
    print("whack a mole")
    M = np.array([[0.5, 0.5],[0.9, 0.1], [0.1, 0.9]])
    Z = np.array([1,1,1,2,2,2,1,2,2,1])
    T = np.array([[0.1,0.4,0.5],[0.4,0,0.6],[0,0.6,0.4]])
    s0 = np.array([0,0,1])

    path,p = Viterbi(M,Z,T,s0)
    print("Most Likely Path:")
    print(path)
    print("Joint probability")
    print(p)
    states = [1,2,3]
    obs = [12]
    h = hm.hmm(states,obs,np.asmatrix(s0),np.asmatrix(T),np.asmatrix(M))
    print(h.viterbi(Z.tolist()))
