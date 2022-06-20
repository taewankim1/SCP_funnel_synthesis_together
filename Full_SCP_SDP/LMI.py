import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import numpy as np
import cvxpy as cvx
import time
import random
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    # print ("Values are: \n%s" % (x))

import cost
import model
import IPython

from Scaling import TrajectoryScaling

class Q_update :
    def __init__(self,ix,iu,iq,ip,iw,N,delT,S,R) :
        self.ix = ix
        self.iu = iu
        self.iq = iq
        self.ip = ip
        self.iw = iw
        self.N = N
        self.small = 1e-6
        self.delT = delT

        self.Cv = np.vstack((np.sqrt(S),np.zeros((iu,ix))))
        self.Dv = np.vstack((np.zeros((ix,iu)),np.sqrt(R)))

        self.myScale = TrajectoryScaling()
    def initialize(self,x,u,A,B,C,D,E,F,G) :
        ix,iu,N = self.ix,self.iu,self.N
        self.myScale.update_scaling_from_traj(x,u)
        self.Sx,self.iSx,self.sx,self.Su,self.iSu,self.su = self.myScale.get_scaling()

        self.Q_list = []
        self.Y_list = []
        for i in range(N+1) :
            self.Q_list.append(cvx.Variable((ix,ix), PSD=True))
            if i < N :
                self.Y_list.append(cvx.Variable((iu,ix)))
        self.nu_Q = cvx.Variable(N+1)
        self.nu_K = cvx.Variable(N)
        self.nu_p = cvx.Variable(1)


        self.x = x
        self.u = u

        self.A = A
        self.B = B 
        self.C = C 
        self.D = D 
        self.E = E 
        self.F = F 
        self.G = G

    def solve_type_1(self,gamma,xi_neighbor,Qf,Qbar,Ybar) :
        ix,iu,N,delT = self.ix,self.iu,self.N,self.delT
        iq,ip,iw = self.iq,self.ip,self.iw

        lambda_w = 1
        constraints = []

        for i in range(N) :
            Qi = self.Sx@self.Q_list[i]@self.Sx
            Yi = self.Su@self.Y_list[i]@self.Sx
            Qi_next = self.Sx@self.Q_list[i+1]@self.Sx
            Q_dot = (Qi_next - Qi) / delT
            
            LMI11 = Qi@self.A[i].T+Yi.T@self.B[i].T+self.A[i]@Qi+self.B[i]@Yi-Q_dot+lambda_w*Qi
            LMI12 = self.E*self.nu_p
            LMI13 = self.F[i]
            LMI14 = Qi@self.C.T+Yi.T@self.D.T
            LMI15 = Qi@self.Cv.T+Yi.T@self.Dv.T
            
            # LMI22 = self.nu_p*-np.eye(iq*ip)
            LMI22 = self.nu_p*-np.eye(ip)
            LMI23 = np.zeros((ip,iw))
            LMI24 = np.zeros((ip,iq))
            LMI25 = np.zeros((ip,ix+iu))
            
            LMI33 = -lambda_w * np.eye(iw)
            LMI34 = self.G.T
            LMI35 = np.zeros((iw,ix+iu))
            
            LMI44 = -self.nu_p * (1/gamma[i]**2) * np.eye(iq)
            LMI45 = np.zeros((iq,ix+iu))
            
            LMI55 = -np.eye(ix+iu)
            
            row1 = cvx.hstack((LMI11,LMI12,LMI13,LMI14,LMI15))
            row2 = cvx.hstack((LMI12.T,LMI22,LMI23,LMI24,LMI25))
            row3 = cvx.hstack((LMI13.T,LMI23.T,LMI33,LMI34,LMI35))
            row4 = cvx.hstack((LMI14.T,LMI24.T,LMI34.T,LMI44,LMI45))
            row5 = cvx.hstack((LMI15.T,LMI25.T,LMI35.T,LMI45.T,LMI55))
            LMI = cvx.vstack((row1,row2,row3,row4,row5))

            constraints.append(LMI << 0)
            constraints.append(self.nu_p>=self.small)
            
        for i in range(N+1) :
            Qi = self.Sx@self.Q_list[i]@self.Sx
            # constraints.append(Qi << Qbar[i] + self.nu_Q[i]*np.eye(ix))
            constraints.append(Qi << self.nu_Q[i]*np.eye(ix))
            constraints.append(Qi >> np.eye(ix)*self.small) # PD
            
        for i in range(N) :
            Yi = self.Su@self.Y_list[i]@self.Sx
            Qi = self.Sx@self.Q_list[i]@self.Sx
            tmp1 = cvx.hstack((self.nu_K[i]*np.eye(iu),Yi))
            tmp2 = cvx.hstack((Yi.T,Qi))
            constraints.append( cvx.vstack((tmp1,tmp2)) >> 0)
            
        # initial condition
        Qi = self.Sx@self.Q_list[0]@self.Sx    
        for xp in xi_neighbor : 
            tmp1 = cvx.hstack(([[1]],np.expand_dims(self.x[0]-xp,0)))
            tmp2 = cvx.hstack((np.expand_dims(self.x[0]-xp,1),Qi))
            constraints.append( cvx.vstack((tmp1,tmp2)) >> 0)
        # final condition
        Qi = self.Sx@self.Q_list[-1]@self.Sx   
        constraints.append(Qi << Qf )

        # trust region
        l_t = []
        for i in range(N) :
            Yi = self.Su@self.Y_list[i]@self.Sx
            Qi = self.Sx@self.Q_list[i]@self.Sx
            l_t.append(cvx.norm(Qi-Qbar[i],'fro') + cvx.norm(Yi-Ybar[i],'fro'))
            tmp1 = cvx.hstack((self.nu_K[i]*np.eye(iu),Yi))
            tmp2 = cvx.hstack((Yi.T,Qi))
            constraints.append( cvx.vstack((tmp1,tmp2)) >> 0)
        Qi = self.Sx@self.Q_list[-1]@self.Sx   
        l_t.append(cvx.norm(Qi-Qbar[-1],'fro'))



        l = cvx.sum(self.nu_Q) + cvx.sum(self.nu_K) + 1e0 * cvx.sum(l_t)
        # l = 1
        prob = cvx.Problem(cvx.Minimize(l), constraints)
        prob.solve(solver=cvx.MOSEK)#verbose=False,solver=cvx.ECOS,warm_start=True)
        print("Funnel update cost is",l.value)

        # print(self.nu_Q.value)
        # print(self.nu_K.value)
        # print(self.nu_p.value)

        Qnew = []
        Ynew = []
        for i in range(N+1) :
            Qnew.append(self.Sx@self.Q_list[i].value@self.Sx)
            if i < N :
                Ynew.append(self.Su@self.Y_list[i].value@self.Sx)
        Knew = []
        for i in range(N) :
            K = Ynew[i]@np.linalg.inv(Qnew[i])
            Knew.append(K)
        Knew = np.array(Knew)
        Qnew = np.array(Qnew)

        return Qnew,Knew,Ynew,prob.status







#     LMI11 = Qi@A[i].T+Yi.T@B[i].T+A[i]@Qi+B[i]@Yi-Q_dot+lambda_w*Qi
#     LMI12 = E
#     LMI13 = F
#     LMI14 = Qi@C.T+Yi.T@D.T
#     LMI15 = np.zeros((ix,iq*ip))
#     LMI16 = Qi@Cv.T+Yi.T@Dv.T
    
#     LMI22 = np.zeros((iq*ip,iq*ip))
#     LMI23 = np.zeros((iq*ip,iw))
#     LMI24 = np.zeros((iq*ip,iq*ip))
#     LMI25 = np.eye(iq*ip)
#     LMI26 = np.zeros((iq*ip,ix+iu))
    
#     LMI33 = -lambda_w * np.eye(iw)
#     LMI34 = G.T
#     LMI35 = np.zeros((iw,iq*ip))
#     LMI36 = np.zeros((iw,ix+iu))
    
#     LMI44 = -1/lambda_p * np.diag(1/gamma[i]**2)
#     LMI45 = np.zeros((iq*ip,iq*ip))
#     LMI46 = np.zeros((iq*ip,ix+iu))
    
#     LMI55 = -1/lambda_p * -np.eye(iq*ip)
#     LMI56 = np.zeros((iq*ip,ix+iu))
    
#     LMI66 = -np.eye(ix+iu)
    
#     row1 = cvx.hstack((LMI11,LMI12,LMI13,LMI14,LMI15,LMI16))
#     row2 = cvx.hstack((LMI12.T,LMI22,LMI23,LMI24,LMI25,LMI26))
#     row3 = cvx.hstack((LMI13.T,LMI23.T,LMI33,LMI34,LMI35,LMI36))
#     row4 = cvx.hstack((LMI14.T,LMI24.T,LMI34.T,LMI44,LMI45,LMI46))
#     row5 = cvx.hstack((LMI15.T,LMI25.T,LMI35.T,LMI45.T,LMI55,LMI56))
#     row6 = cvx.hstack((LMI16.T,LMI26.T,LMI36.T,LMI46.T,LMI56.T,LMI66))
#     LMI = cvx.vstack((row1,row2,row3,row4,row5,row6))

#     row1 = cvx.hstack((LMI11,LMI13))
#     row2 = cvx.hstack((LMI13.T,LMI33))
#     LMI = cvx.vstack((row1,row2))
    
#     row1 = cvx.hstack((LMI11,LMI13,LMI16))
#     row2 = cvx.hstack((LMI13.T,LMI33,LMI36))
#     row3 = cvx.hstack((LMI16.T,LMI36.T,LMI66))
#     LMI = cvx.vstack((row1,row2,row3))
################################################################################################







################################################################################################
#     LMI11 = Qi@A[i].T+Yi.T@B[i].T+A[i]@Qi+B[i]@Yi-Q_dot+(alpha+lambda_w)*Qi
#     LMI12 = E*nu_p
#     LMI13 = F
#     LMI14 = (Qi@C.T+Yi.T@D.T)@np.diag(gamma[i])
#     LMI22 = nu_p*-np.eye(iq*ip)
#     LMI23 = np.zeros((iq*ip,iw))
#     LMI24 = np.zeros((iq*ip,iq*ip))
#     LMI33 = -lambda_w * np.eye(iw)
#     LMI34 = G.T
#     LMI44 = -nu_p * np.eye(iq*ip)
    
#     row1 = cvx.hstack((LMI11,LMI12,LMI13,LMI14))
#     row2 = cvx.hstack((LMI12.T,LMI22,LMI23,LMI24))
#     row3 = cvx.hstack((LMI13.T,LMI23.T,LMI33,LMI34))
#     row4 = cvx.hstack((LMI14.T,LMI24.T,LMI34.T,LMI44))
#     LMI = cvx.vstack((row1,row2,row3,row4))

#     row1 = cvx.hstack((LMI11,LMI12,LMI14))
#     row2 = cvx.hstack((LMI12.T,LMI22,LMI24))
#     row3 = cvx.hstack((LMI14.T,LMI24.T,LMI44))
#     LMI = cvx.vstack((row1,row2,row3))
    
#     row1 = cvx.hstack((LMI11,LMI13))
#     row2 = cvx.hstack((LMI13.T,LMI33))
#     LMI = cvx.vstack((row1,row2))
################################################################################################