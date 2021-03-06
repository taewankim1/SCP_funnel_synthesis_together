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

class Single_PTR:
    def __init__(self,name,horizon,tf,maxIter,Model,Cost,Const,Scaling=None,
                        w_c=1,w_vc=1e4,w_tr=1e-3,tol_vc=1e-10,tol_tr=1e-3,tol_bc=1e-3,
                        flag_policyopt=False,verbosity=True):
        self.name = name
        self.model = Model
        self.const = Const
        self.cost = Cost
        self.N = horizon
        self.tf = tf
        self.delT = tf/horizon
        if Scaling is None :
            self.Scaling = TrajectoryScaling() 
            self.flag_update_scale = True
        else :
            self.Scaling = Scaling
            self.flag_update_scale = False
        
        # cost optimization
        self.verbosity = verbosity
        self.w_c = w_c
        self.w_vc = w_vc
        self.w_tr = w_tr
        # self.tol_fun = 1e-6
        self.tol_tr = tol_tr
        self.tol_vc = tol_vc
        self.tol_bc = tol_bc
        self.maxIter = maxIter
        self.last_head = True
        self.flag_policyopt = flag_policyopt
        self.initialize()

    def initialize(self) :
        
        self.dV = np.zeros((1,2))
        self.x0 = np.zeros(self.model.ix)
        self.x = np.zeros((self.N+1,self.model.ix))
        self.u = np.ones((self.N+1,self.model.iu))
        self.xbar = np.zeros((self.N+1,self.model.ix))
        self.ubar = np.ones((self.N+1,self.model.iu))
        self.vc = np.ones((self.N,self.model.ix)) * 1e-1
        self.tr = np.ones((self.N+1))

        self.xnew = np.zeros((self.N+1,self.model.ix))
        self.unew = np.zeros((self.N+1,self.model.iu))
        self.vcnew = np.zeros((self.N,self.model.ix))
        self.Alpha = np.power(10,np.linspace(0,-3,11))

        self.A = np.zeros((self.N,self.model.ix,self.model.ix))
        self.B = np.zeros((self.N,self.model.ix,self.model.iu))  
        self.Bm = np.zeros((self.N,self.model.ix,self.model.iu))  
        self.Bp = np.zeros((self.N,self.model.ix,self.model.iu))  
        self.s = np.zeros((self.N,self.model.ix))
        self.z = np.zeros((self.N,self.model.ix))

        self.c = 0
        self.cvc = 0
        self.ctr = 0
        self.cnew = 0
        self.cvcnew = 0
        self.ctrnew = 0

        self.cx = np.zeros((self.N+1,self.model.ix))
        self.cu = np.zeros((self.N,self.model.iu))
        self.cxx = np.zeros((self.N+1,self.model.ix,self.model.ix))
        self.cxu = np.zeros((self.N,self.model.ix,self.model.iu))
        self.cuu = np.zeros((self.N,self.model.iu,self.model.iu))

    def get_model(self) :
        return self.A,self.B,self.s,self.z,self.vc

    def forward_multiple(self,x,u,iteration=0) :
        N = self.N
        ix = self.model.ix
        iu = self.model.iu

        def dfdt(t,x,um,up) :
            u = um
            return np.squeeze(self.model.forward(x,u))

        xnew = np.zeros((N,ix))

        for i in range(N) :
            # if iteration < 10 :
            #     sol = solve_ivp(dfdt,(0,self.delT),xnew[i],args=(u[i],u[i+1]))
            # else :
            sol = solve_ivp(dfdt,(0,self.delT),x[i],args=(u[i],u[i+1]),method='RK45',rtol=1e-6,atol=1e-10)
            xnew[i] = sol.y[:,-1]

        return xnew,u

    def forward_full(self,x0,u,iteration) :
        N = self.N
        ix = self.model.ix
        iu = self.model.iu

        def dfdt(t,x,um,up) :
            u = um
            return np.squeeze(self.model.forward(x,u))

        xnew = np.zeros((N+1,ix))
        xnew[0] = x0

        for i in range(N) :
            # if iteration < 10 :
            #     sol = solve_ivp(dfdt,(0,self.delT),xnew[i],args=(u[i],u[i+1]))
            # else :
            sol = solve_ivp(dfdt,(0,self.delT),xnew[i],args=(u[i],u[i+1]),method='RK45',rtol=1e-6,atol=1e-10)
            xnew[i+1] = sol.y[:,-1]

        return xnew,u

    def cvxopt(self):
        # TODO - we can get rid of most of loops here
        def get_neighbor_points(xc,margin) :
            xp = []
            xp.append(xc+np.array([margin,margin]))
            xp.append(xc+np.array([-margin,margin]))
            xp.append(xc+np.array([margin,-margin]))
            xp.append(xc+np.array([-margin,-margin]))
            return xp


        # state & input & horizon size
        ix = self.model.ix
        iu = self.model.iu
        N = self.N

        if self.flag_update_scale is True :
            self.Scaling.update_scaling_from_traj(self.x,self.u)
        Sx,iSx,sx,Su,iSu,su = self.Scaling.get_scaling()

        x_cvx = cvx.Variable((N+1,ix))
        u_cvx = cvx.Variable((N+1,iu))
        vc = cvx.Variable((N,ix))

        # initial & final boundary condition
        constraints = []
        if self.xi_neighbor is None :
            constraints.append(Sx@x_cvx[0] + sx == self.xi)
        else :
            for xp in self.xi_neighbor :
                constraints.append(cvx.quad_form(xp-Sx@x_cvx[0]-sx,np.linalg.inv(self.Q[0])) <= 1)

        if self.xf_margin is None :
            constraints.append(Sx@x_cvx[-1] + sx == self.xf)
        else :
            for i in range(ix) :
                a = np.zeros(ix)
                a[i] = 1
                h_Q = np.sqrt(a.T@self.Q[-1]@a) 
                constraints.append(h_Q + a.T@(Sx@x_cvx[-1] + sx) <= self.xf[i]+self.xf_margin[i])
                constraints.append(-h_Q + a.T@(Sx@x_cvx[-1] + sx) >= self.xf[i]-self.xf_margin[i])

        # state and input contraints
        for i in range(0,N) : 
            constraints += self.const.forward(Sx@x_cvx[i]+sx,Su@u_cvx[i]+su,self.x[i],self.u[i],self.Q[i],self.K[i])
        constraints += self.const.forward(Sx@x_cvx[N]+sx,Su@u_cvx[N]+su,
            self.x[N],self.u[N],
            self.Q[N],self.K[0]*0)

        # model constraints
        for i in range(0,N) :
            constraints.append(Sx@x_cvx[i+1]+sx == self.A[i]@(Sx@x_cvx[i]+sx)+self.B[i]@(Su@u_cvx[i]+su)
                +self.tf*self.s[i]
                +self.z[i]
                +vc[i])

        # cost
        objective = []
        objective_vc = []
        objective_tr = []
        for i in range(0,N+1) :
            if i < N :
                objective_vc.append(self.w_vc * cvx.norm(vc[i],1))
            objective.append(self.w_c * self.cost.estimate_cost_cvx(Sx@x_cvx[i]+
                sx,Su@u_cvx[i]+su,i))
            # trust region with reference trajectory
            objective_tr.append( self.w_tr * (cvx.quad_form(x_cvx[i] -
                    iSx@(self.x0[i]-sx),np.eye(ix)) +
                    cvx.quad_form(u_cvx[i]-iSu@(self.u0[i]-su),np.eye(iu))) )

        l = cvx.sum(objective)
        l_vc = cvx.sum(objective_vc)
        l_tr = cvx.sum(objective_tr)

        l_all = l + l_vc + l_tr
        prob = cvx.Problem(cvx.Minimize(l_all), constraints)

        error = False
        # prob.solve(verbose=False,solver=cvx.MOSEK)
        # prob.solve(verbose=False,solver=cvx.CPLEX)
        # prob.solve(verbose=False,solver=cvx.GUROBI)
        prob.solve(verbose=False,solver=cvx.ECOS)
        # prob.solve(verbose=False,solver=cvx.SCS)

        if prob.status == cvx.OPTIMAL_INACCURATE :
            print("WARNING: inaccurate solution")

        try :
            xnew = np.zeros_like(self.x)
            unew = np.zeros_like(self.u)
            for i in range(N+1) :
                xnew[i] = Sx@x_cvx[i].value + sx
                unew[i] = Su@u_cvx[i].value + su
        except ValueError :
            print("FAIL: ValueError")
            error = True
        except TypeError :
            print("FAIL: TypeError")
            error = True
        # print("x_min {:f} x_max {:f} u_min {:f} u _max{:f}".format(np.min(x_cvx.value),
        #                                                         np.max(x_cvx.value),
        #                                                         np.min(u_cvx.value),
        #                                                         np.max(u_cvx.value)))
        return prob.status,l.value,l_vc.value,l_tr.value,xnew,unew,vc.value,error
                   
        
    def run(self,x0,u0,xi,xf,Q=None,K=None,xi_neighbor=None,xf_margin=None):
        # state & input & horizon size
        ix = self.model.ix
        iu = self.model.iu
        N = self.N

        # save initial trajectory and input
        self.x0 = np.copy(x0)
        self.u0 = np.copy(u0)
        
        # initial input
        self.x = x0
        self.u = u0

        if Q is None :
            self.Q = np.tile(np.zeros((ix,ix)),(N+1,1,1)) 
            self.K = np.tile(np.zeros((iu,ix)),(N+1,1,1)) 
        else :
            self.Q = Q
            self.K = K

        self.xi_neighbor = xi_neighbor
        self.xf_margin = xf_margin

        # initial condition
        self.xi = xi

        # final condition
        self.xf = xf
        
        
        # generate initial trajectory
        diverge = False
        stop = False

        self.c = 1e3
        self.cvc = 0
        self.ctr = 0

        # save trajectory
        x_traj = []
        u_traj = []
        T_traj = []

        # iterations starts!!
        flgChange = True
        total_num_iter = 0
        flag_boundary = False
        for iteration in range(self.maxIter) :

            # step1. differentiate dynamics and cost
            if flgChange == True:
                start = time.time()
                self.A,self.B,self.s,self.z,self.x_prop_n = self.model.diff_discrete_zoh(self.x[0:N,:],self.u[0:N,:],self.delT,self.tf)
                self.x_prop = np.squeeze(self.A@np.expand_dims(self.x[0:N,:],2) +
                                self.B@np.expand_dims(self.u[0:N,:],2) + 
                                np.expand_dims(self.tf*self.s+self.z,2))
                # print("prop_n - prop",np.sum(self.x_prop_n-self.x_prop))
                # remove small element
                eps_machine = np.finfo(float).eps
                self.A[np.abs(self.A) < eps_machine] = 0
                self.B[np.abs(self.B) < eps_machine] = 0
                self.Bm[np.abs(self.Bm) < eps_machine] = 0
                self.Bp[np.abs(self.Bp) < eps_machine] = 0

                flgChange = False
                pass
            time_derivs = (time.time() - start)
            # step2. cvxopt
            # try :
            prob_status,l,l_vc,l_tr,self.xnew,self.unew,self.vcnew,error = self.cvxopt()
            if error == True :
                total_num_iter = 1e5
                break

            # step3. forward
            start = time.time()
            self.xfwd,self.ufwd = self.forward_full(self.xnew[0,:],self.unew,iteration)

            expected = self.c + self.cvc + self.ctr - l - l_vc - l_tr
            # check the boundary condtion
            bc_error_norm = np.max(np.linalg.norm(self.xfwd-self.xnew,axis=1))

            if  bc_error_norm >= self.tol_bc :
                flag_boundary = False
            else :
                flag_boundary = True
            time_forward = time.time() - start

            # step4. accept step, draw graphics, print status 
            if self.verbosity == True and self.last_head == True:
                self.last_head = False
                print("iteration   total_cost        cost        ||vc||     ||tr||       reduction   w_tr        bounary")
            # accept changes
            self.x = self.xnew
            self.u = self.unew
            self.vc = self.vcnew
            self.c = l 
            self.cvc = l_vc 
            self.ctr = l_tr
            flgChange = True

            if self.verbosity == True:
                print("%-12d%-18.3f%-12.3f%-12.3g%-12.3g%-12.3g%-12.3f%-1d(%2.3g)" % ( iteration+1,self.c+self.cvc+self.ctr,
                    self.c,self.cvc/self.w_vc,self.ctr/self.w_tr,
                    expected,self.w_tr,flag_boundary,bc_error_norm))

        return self.xfwd,self.ufwd,self.xnew,self.unew,total_num_iter,flag_boundary,l,l_vc,l_tr

    def print_eigenvalue(self,A_) :
        eig,eig_vec = np.linalg.eig(A_)
        print("(discrete) eigenvalue of A",np.max(np.real(eig)))
        if self.model.type_linearization == "numeric_central" :
            A,B = self.model.diff_numeric_central(self.x,self.u)
        elif self.model.type_linearization == "numeric_forward" :
            A,B = self.model.diff_numeric(self.x,self.u)
        elif self.model.type_linearization == "analytic" :
            A,B = self.model.diff(self.x,self.u)
        eig,eig_vec = np.linalg.eig(A)
        print("(continuous) eigenvalue of A",np.max(np.real(eig)))


        
        
        
        
        
        
        
        
        
        
        
        


