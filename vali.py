import numpy as np
import matplotlib.pyplot as plt

def valiter(V0,R,T,l,S,y):
    b,n,m = T.shape
    V = V0
    for i in range(0,l):
        next = np.zeros((1,len(S)))

        for j in range(0,len(S)):
            max = 0
            for k in range(0,b):
                m = 0
                for l in range(0,len(S)):
                    m += T[k,j,l]*(R[j]+y*V[i,l])
                if m > max:
                    max = m
            next[0,j] = max
        V = np.vstack((V,next))

    return V

if __name__=="__main__":
    S = [1,2,3,4]
    pie = [1,0,0,0]
    V = np.zeros((1,len(S)))
    T = np.array([[[0.1,0.9,0,0],[0.9,0.1,0,0],[0,0,0.1,0.9],[0,0,0,1]],[[0.9,0.1,0,0],[0,0.1,0,0.9],[0.9,0,0.1,0],[0,0,0,1]]])
    R = [0,0,0,1]
    y = 0.95
    VI = valiter(V,R,T,25,S,0.95)
    VI = VI/(np.max(VI))
    r = range(0,26)
    print(VI)
    plt.plot(r,VI[:,0] , r,VI[:,1] , r,VI[:,2] , r,VI[:,3])
    plt.xlabel("Iteration i")
    plt.ylabel("V_i(s)")
    plt.legend(["S_1", "S_2","S_3", "S_4"])
    plt.show()
