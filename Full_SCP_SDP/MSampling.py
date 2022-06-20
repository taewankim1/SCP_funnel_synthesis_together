import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import numpy as np
import cvxpy as cvx
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    # print ("Values are: \n%s" % (x))
from utils_alg import get_sample_eta_w

class Mstep :
    def __init__(self,ix,iu,iq,ip,iw,N) :
        self.ix = ix
        self.iu = iu
        self.iq = iq
        self.ip = ip
        self.iw = iw
        self.N = N

    def initialize(self,xbar,ubar,Qbar,Kbar,A,B,C,D,E,F,G,myModel)  :
        ix,iu,N = self.ix,self.iu,self.N
        ip,iq,iw = self.ip,self.iq,self.iw

        # self.Delta = cvx.Variable((ip*iq,ip*iq),diag=True)
        # self.nu = cvx.Variable(ix)

        # self.LHS = cvx.Parameter(ix)
        # # Ecvx = cvx.Parameter((ix,iq*ip))
        # self.mu = cvx.Parameter(iq*ip)

        # constraints = []
        # constraints.append(self.LHS == E@self.Delta@self.mu + self.nu)
        # cost = cvx.norm(self.Delta,'fro') + 1e4*cvx.norm(self.nu)
        # self.prob = cvx.Problem(cvx.Minimize(cost),constraints)
        # assert self.prob.is_dcp(dpp=True)

        self.xbar = xbar
        self.ubar = ubar
        self.Qbar = Qbar
        self.Kbar = Kbar

        self.A = A
        self.B = B 
        self.C = C 
        self.D = D 
        self.E = E 
        self.F = F 
        self.G = G

        self.myModel = myModel

    def update_lipschitz(self) :
        ix,iu,N = self.ix,self.iu,self.N
        ip,iq,iw = self.ip,self.iq,self.iw

        gamma = np.zeros((N))
        for idx in range(N) :
            num_sample = 100
            eta_sample,w_sample = get_sample_eta_w(self.Qbar[idx],num_sample,ix,iw) 

            Delta_list = []
            gamma_val = []
            max_val = 0
            idx_max = 0
            for idx_s in range(num_sample) :
                Delta = cvx.Variable((ip,iq))
                nu = cvx.Variable(ix)

                es,ws = eta_sample[idx_s],w_sample[idx_s] # samples on surface of ellipse
                xii = self.Kbar[idx]@es

                eta_dot = (self.myModel.forward_noise_1(self.xbar[idx]+es,self.ubar[idx]+xii,ws) 
                        - self.myModel.forward_noise_1(self.xbar[idx],self.ubar[idx],np.zeros(iw))).squeeze()

                LHS = eta_dot - (self.A[idx]+self.B[idx]@self.Kbar[idx])@es-self.F[idx]@ws
                mu = (self.C+self.D@self.Kbar[idx])@es + self.G@ws

                constraints = []
                constraints.append(LHS == self.E@Delta@mu + nu)
                cost = cvx.norm(Delta,2) + 1e4*cvx.norm(nu)
                prob = cvx.Problem(cvx.Minimize(cost),constraints)

                prob.solve()
                if prob.status != "optimal" :
                    print(idx_s,self.prob.status)
                if prob.value >= max_val :
                    idx_max = idx_s
                    max_val = prob.value 
                Delta_list.append(Delta.value)
                gamma_val.append(cvx.norm(Delta,2).value)
            Delta_list = np.array(Delta_list)
            gamma_val = np.array(gamma_val)

            gamma[idx] = np.max(gamma_val)
        #     gamma[idx] = np.abs(Delta_list[idx_max])
            # print("gamma[{:}]".format(idx),gamma[idx],"nu",cvx.norm(nu).value)
        return gamma

    def update_lipschitz_parallel(self) :
        ix,iu,N = self.ix,self.iu,self.N
        ip,iq,iw = self.ip,self.iq,self.iw

        gamma = np.zeros((N))
        for idx in range(N) :
            num_sample = 100
            eta_sample,w_sample = get_sample_eta_w(self.Qbar[idx],num_sample,ix,iw) 

            # Delta = [cvx.Variable((ip,iq)) for j in range(num_sample)]
            Delta = cvx.Variable((num_sample,ip*iq))
            nu = cvx.Variable((num_sample,ix))

            constraints = []
            cost_list = []
            deltanorm_list = []
            for idx_s in range(num_sample) :
                es,ws = eta_sample[idx_s],w_sample[idx_s] # samples on surface of ellipse
                xii = self.Kbar[idx]@es

                eta_dot = (self.myModel.forward_noise_1(self.xbar[idx]+es,self.ubar[idx]+xii,ws) 
                        - self.myModel.forward_noise_1(self.xbar[idx],self.ubar[idx],np.zeros(iw))).squeeze()
                
                LHS = eta_dot - (self.A[idx]+self.B[idx]@self.Kbar[idx])@es-self.F[idx]@ws
                mu = (self.C+self.D@self.Kbar[idx])@es + self.G@ws
                delta = cvx.reshape(Delta[idx_s],(ip,iq))
                constraints.append(LHS == self.E@delta@mu + nu[idx_s])
                cost_list.append(cvx.norm(delta,2))
                deltanorm_list.append(cvx.norm(delta,2))

            # cost = cvx.sum(cost_list) + 1e4*cvx.norm(nu,'fro')
            cost = cvx.norm(Delta,'fro') + 1e4*cvx.norm(nu,'fro')
            # cost = cvx.norm(Delta,2) + 1e4*cvx.norm(nu,'fro')
            prob = cvx.Problem(cvx.Minimize(cost),constraints)
            prob.solve()
            # prob.solve(verbose=False,solver=cvx.GUROBI)
            if prob.status != "optimal" :
                print(idx_s,prob.status)

            # Delta_list = Delta.value
            gamma_all = np.array([deltanorm_list[i].value  for i in range(num_sample)])
            gamma[idx] = np.max(gamma_all)
            # print("gamma[{:}]".format(idx),gamma[idx],"nu",cvx.norm(nu).value)

        return gamma

    def update_lipschitz_parallel2(self) :
        ix,iu,N = self.ix,self.iu,self.N
        ip,iq,iw = self.ip,self.iq,self.iw

        num_sample = 100
        Delta = cvx.Variable((num_sample*N,ip*iq))
        nu = cvx.Variable((num_sample*N,ix))

        gamma = np.zeros((N))
        deltanorm_list = []
        constraints = []
        for idx in range(N) :
            eta_sample,w_sample = get_sample_eta_w(self.Qbar[idx],num_sample,ix,iw) 
            for idx_s in range(num_sample) :
                es,ws = eta_sample[idx_s],w_sample[idx_s] # samples on surface of ellipse
                xii = self.Kbar[idx]@es

                eta_dot = (self.myModel.forward_noise_1(self.xbar[idx]+es,self.ubar[idx]+xii,ws) 
                        - self.myModel.forward_noise_1(self.xbar[idx],self.ubar[idx],np.zeros(iw))).squeeze()
                
                LHS = eta_dot - (self.A[idx]+self.B[idx]@self.Kbar[idx])@es-self.F[idx]@ws
                mu = (self.C+self.D@self.Kbar[idx])@es + self.G@ws
                delta = cvx.reshape(Delta[idx*num_sample+idx_s],(ip,iq))
                constraints.append(LHS == self.E@delta@mu + nu[idx*num_sample+idx_s])
                deltanorm_list.append(cvx.norm(delta,2))

        # cost = cvx.sum(cost_list) + 1e4*cvx.norm(nu,'fro')
        cost = cvx.norm(Delta,'fro') + 1e4*cvx.norm(nu,'fro')
        # cost = cvx.norm(Delta,2) + 1e4*cvx.norm(nu,'fro')
        prob = cvx.Problem(cvx.Minimize(cost),constraints)
        prob.solve()
        if prob.status != "optimal" :
            print(idx_s,prob.status)
            gamma = None
        else :
            for idx in range(N) :
                gamma_all = np.array([deltanorm_list[idx*num_sample + i].value  for i in range(num_sample)])
                gamma[idx] = np.max(gamma_all)
                # print("gamma[{:}]".format(idx),gamma[idx])

        return gamma

    # def update(self) :
    #     ix,iu,N = self.ix,self.iu,self.N
    #     ip,iq,iw = self.ip,self.iq,self.iw

    #     gamma = np.zeros((N,ip*iq))
    #     for idx in range(N) :
    #         num_sample = 100
    #         eta_sample,w_sample = get_sample_eta_w(self.Qbar[idx],num_sample,ix,iw) 

    #         Delta_list = []
    #         gamma_val = []
    #         max_val = 0
    #         idx_max = 0
    #         for idx_s in range(num_sample) :
    #             es,ws = eta_sample[idx_s],w_sample[idx_s] # samples on surface of ellipse
    #             xii = self.Kbar[idx]@es

    #             eta_dot = (self.myModel.forward_noise_1(self.xbar[idx]+es,self.ubar[idx]+xii,ws) 
    #                     - self.myModel.forward_noise_1(self.xbar[idx],self.ubar[idx],np.zeros(iw))).squeeze()
                
    #             self.LHS.value = eta_dot - (self.A[idx]+self.B[idx]@self.Kbar[idx])@es-self.F@ws
    #             self.mu.value = (self.C+self.D@self.Kbar[idx])@es + self.G@ws

    #             self.prob.solve()
    #             if self.prob.status != "optimal" :
    #                 print(idx_s,self.prob.status)
    #             if self.prob.value >= max_val :
    #                 idx_max = idx_s
    #                 max_val = self.prob.value 
    #             Delta_list.append(cvx.diag(self.Delta).value)
    #             gamma_val.append(self.prob.value)
    #         Delta_list = np.array(Delta_list)

    #         gamma[idx] = np.max(np.abs(Delta_list),0)
    #     #     gamma[idx] = np.abs(Delta_list[idx_max])
    #         print("gamma[{:}]".format(idx),gamma[idx],"nu",cvx.norm(self.nu).value)
    #     return gamma

    # def update_parallel(self) :
    #     ix,iu,N = self.ix,self.iu,self.N
    #     ip,iq,iw = self.ip,self.iq,self.iw

    #     gamma = np.zeros((N,ip*iq))
    #     for idx in range(N) :
    #         num_sample = 200
    #         eta_sample,w_sample = get_sample_eta_w(self.Qbar[idx],num_sample,ix,iw) 

    #         Delta = cvx.Variable((num_sample,ip*iq))
    #         nu = cvx.Variable((num_sample,ix))

    #         constraints = []
    #         for idx_s in range(num_sample) :
    #             es,ws = eta_sample[idx_s],w_sample[idx_s] # samples on surface of ellipse
    #             xii = self.Kbar[idx]@es

    #             eta_dot = (self.myModel.forward_noise_1(self.xbar[idx]+es,self.ubar[idx]+xii,ws) 
    #                     - self.myModel.forward_noise_1(self.xbar[idx],self.ubar[idx],np.zeros(iw))).squeeze()
                
    #             LHS = eta_dot - (self.A[idx]+self.B[idx]@self.Kbar[idx])@es-self.F@ws
    #             mu = (self.C+self.D@self.Kbar[idx])@es + self.G@ws
    #             constraints.append(LHS == self.E@cvx.diag(Delta[idx_s])@mu + nu[idx_s])

    #         cost = cvx.norm(Delta,'fro') + 1e4*cvx.norm(nu,'fro')
    #         prob = cvx.Problem(cvx.Minimize(cost),constraints)
    #         prob.solve()
    #         if prob.status != "optimal" :
    #             print(idx_s,prob.status)

    #         Delta_list = Delta.value

    #         gamma[idx] = np.max(np.abs(Delta_list),0)
    #         # print("gamma[{:}]".format(idx),gamma[idx],"nu",cvx.norm(nu).value)

    #     return gamma