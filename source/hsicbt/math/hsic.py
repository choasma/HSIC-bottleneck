import torch
import numpy as np
from torch.autograd import Variable, grad

M = 512
H = torch.eye(M) - (1./M) * torch.ones([M,M])
E = 1E-5
K_Ii = (1./(E*M))
F = .1
nK_I = E*M*torch.eye(M)
kS = int(F*M)
print(kS)
kK_I = E*kS*torch.eye(kS)
kH = torch.eye(kS) - (1./kS) * torch.ones([kS,kS])
ws_I = K_Ii*torch.eye(M)
ws_I_ = 0.00001*torch.eye(4)



def sigma_estimation(X, Y):
    """ sigma from median distance
    """
    D = distmat(torch.cat([X,Y]))
    D = D.detach().cpu().numpy()
    Itri = np.tril_indices(D.shape[0], -1)
    Tri = D[Itri]
    med = np.median(Tri)
    if med <= 0:
        med=np.mean(Tri)
    if med<1E-2:
        med=1E-2
    return med

def distmat(X):
    """ pairwise distance matrix where each element Dij is ||Xi - Xj||^2
    Args:
        X(torch.tensor): design matrix sized (n, d) with n instances 
                         and dimension d of each element
    Returns:
        (torch.tensor): squared distance matrix
    """
    r = torch.sum(X*X, 1)
    r = r.view([-1, 1])
    a = torch.mm(X, torch.transpose(X,0,1))
    D = r.expand_as(a) - 2*a +  torch.transpose(r,0,1).expand_as(a)
    D = torch.abs(D)
    return D

def kernelmat(X, sigma):
    """ kernel matrix baker
    Args:
        X(torch.tensor): squred distance matrix from the function distmat sized (n, n)
    Returns:
        (torch.tensor): kenerl matrix sized (n, n)
    """
    m = int(X.size()[0])
    dim = int(X.size()[1]) * 1.0
    Dxx = distmat(X)
    if sigma:
        variance = 2.*sigma*sigma*X.size()[1]
        Kx = torch.exp( -Dxx / variance).type(torch.FloatTensor)   # kernel matrices
    else:
        try:
            sx = sigma_estimation(X,X)
            Kx = torch.exp( -Dxx / (2.*sx*sx)).type(torch.FloatTensor)
        except RuntimeError as e:
            raise RuntimeError("Unstable sigma {} with maximum/minimum input ({},{})".format(
                sx, torch.max(X), torch.min(X)))
    return Kx

def distcorr(X, sigma=1.0):
    X = distmat(X)
    X = torch.exp( -X / (2.*sigma*sigma))
    return torch.mean(X)

def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)

def mmd(x, y, sigma=None, use_cuda=True, to_numpy=False):
    m = int(x.size()[0])
    H = torch.eye(m) - (1./m) * torch.ones([m,m])
    # H = Variable(H)
    Dxx = distmat(x)
    Dyy = distmat(y)

    if sigma:
        Kx  = torch.exp( -Dxx / (2.*sigma*sigma))   # kernel matrices
        Ky  = torch.exp( -Dyy / (2.*sigma*sigma))
        sxy = sigma
    else:
        sx = sigma_estimation(x,x)
        sy = sigma_estimation(y,y)
        sxy = sigma_estimation(x,y)
        Kx = torch.exp( -Dxx / (2.*sx*sx))
        Ky = torch.exp( -Dyy / (2.*sy*sy))
    # Kxc = torch.mm(Kx,H)            # centered kernel matrices
    # Kyc = torch.mm(Ky,H)
    Dxy = distmat(torch.cat([x,y]))
    Dxy = Dxy[:x.size()[0], x.size()[0]:]
    Kxy = torch.exp( -Dxy / (1.*sxy*sxy))

    mmdval = torch.mean(Kx) + torch.mean(Ky) - 2*torch.mean(Kxy)

    return mmdval

def mmd_pxpy_pxy(x,y,sigma=None,use_cuda=True, to_numpy=False):
    """
    """
    if use_cuda:
        x = x.cuda()
        y = y.cuda()
    m = int(x.size()[0])

    Dxx = distmat(x)
    Dyy = distmat(y)
    if sigma:
        Kx  = torch.exp( -Dxx / (2.*sigma*sigma))   # kernel matrices
        Ky  = torch.exp( -Dyy / (2.*sigma*sigma))
    else:
        sx = sigma_estimation(x,x)
        sy = sigma_estimation(y,y)
        sxy = sigma_estimation(x,y)
        Kx = torch.exp( -Dxx / (2.*sx*sx))
        Ky = torch.exp( -Dyy / (2.*sy*sy))
    A = torch.mean(Kx*Ky)
    B = torch.mean(torch.mean(Kx,dim=0)*torch.mean(Ky, dim=0))
    C = torch.mean(Kx)*torch.mean(Ky)
    mmd_pxpy_pxy_val = A - 2*B + C
    return mmd_pxpy_pxy_val

def hsic_regular(X, Y, sigma):
    """ the empirical HSIC proposed by Gretton05
    Args:
        X(torch.tensor): Design matrix sampled from Px, with size (n,dx)
        Y(torch.tensor): Design matrix sampled from Py, with size (n,dy)
    Returns:
        (torch.tensor): a scalar HSIC value
    """
    Kxc = kernelmat(X, sigma)
    Kyc = kernelmat(Y, sigma)
    KtK = torch.mul(Kxc, Kyc.t())
    Pxy = torch.mean(KtK)
    return Pxy

def hsic_normalized(x, y, sigma=None, use_cuda=True, to_numpy=True):
    """ Normalized HSIC
    Args:
        X(torch.tensor): Design matrix sampled from Px, with size (n,dx)
        Y(torch.tensor): Design matrix sampled from Py, with size (n,dy)
    Returns:
        (torch.tensor): a scalar normalized HSIC value
    """
    m = int(x.size()[0])
    Pxy = hsic_regular(x, y, sigma)
    Px = torch.sqrt(hsic_regular(x, x, sigma))
    Py = torch.sqrt(hsic_regular(y, y, sigma))
    thehsic = Pxy/(Px*Py)
    return thehsic

def hsic_normalized_cca(X, Y, sigma):
    """ Canonical correlation analysis HSIC
    Args:
        X(torch.tensor): Design matrix sampled from Px, with size (n,dx)
        Y(torch.tensor): Design matrix sampled from Py, with size (n,dy)
    Returns:
        (torch.tensor): a scalar ncca-HSIC value
    """
    m = int(X.size()[0])
    # H = torch.eye(m) - (1./m) * torch.ones([m,m])
    # E = 1E-5
    # nK_I = E*m*torch.eye(m)
    Kxc = kernelmat(X, sigma=sigma).mm(H)
    Kyc = kernelmat(Y, sigma=sigma).mm(H)
    epsilon = 1E-5
    Kxc_i = torch.inverse(Kxc + nK_I)
    Kyc_i = torch.inverse(Kyc + nK_I)
    Rx = (Kxc.mm(Kxc_i))
    Ry = (Kyc.mm(Kyc_i))
    Pxy = torch.sum(torch.mul(Rx, Ry.t()))
    return Pxy

def woodbury_specialized(A_i,U,C_i,V):
    second = torch.inverse(C_i + A_i*V.mm(U) + ws_I_)
    second = U.mm(second).mm(V)
    return ws_I - second

def hsic_normalized_cca_sampling_nystrom(x, y, sigma, n=1, use_cuda=True, to_numpy=True):
    # torch.svd gpu problem
    # https://discuss.pytorch.org/t/torch-svd-is-slow-in-gpu-compared-to-cpu/10770/7
    m = int(x.size()[0])
    # x = x/10.
    Kxc = kernelmat(x, sigma=sigma).mm(H)
    Kyc = kernelmat(y, sigma=sigma).mm(H)
    # X
    Qx = Kxc[:,:n]
    Kxqq = Kxc[:n,:n] # TODO: need to implement rank-k approx
    Kxc_i = woodbury_specialized(K_Ii, Qx, Kxqq, Qx.t())
    # Y
    Qy = Kyc[:,:n]
    Kyqq = Kyc[:n,:n] # TODO: need to implement rank-k approx
    Kyc_i = woodbury_specialized(K_Ii, Qy, Kyqq, Qy.t())

    Rx = (Kxc.mm(Kxc_i))
    Ry = (Kyc.mm(Kyc_i))
    Pxy = torch.sum(torch.mul(Rx, Ry.t()))
    t = time.time()
    return Pxy

def hsic_normalized_cca_sampling_kurt_v1(x, y, sigma, use_cuda=True, to_numpy=True, fraction=0, indices=None):
    """
    run_hsicbt -cfg config/hsicsolve-beta.yaml -tt hsictrain -dc mnist -lr 0.01 -s 6 -ld 500 -dt 5
    """
    Kxc = kernelmat(x, sigma=sigma)
    Kyc = kernelmat(y, sigma=sigma)
    m = Kxc.size()[0]

    idx = torch.randperm(m)
    Kxc = Kxc[idx[:kS],:].mm(H)
    Kyc = Kyc[idx[:kS],:].mm(H)
    Kxc = Kxc.mm(Kxc.t())
    Kyc = Kyc.mm(Kyc.t())
    Kxc_i = torch.inverse(Kxc + kK_I)
    Kyc_i = torch.inverse(Kyc + kK_I)
    Rx = (Kxc.mm(Kxc_i))
    Ry = (Kyc.mm(Kyc_i))
    R = torch.mul(Rx, Ry.t())
    Pxy = torch.sum(R)
    return Pxy



def hsic_normalized_cca_sampling_kurt_v2(x, y, sigma):
    """
    run_hsicbt -cfg config/hsicsolve-beta.yaml -tt hsictrain -dc mnist -lr 0.01 -s 6 -ld 500 -dt 5
    """
    K = kernelmat(x, sigma=sigma)
    L = kernelmat(y, sigma=sigma)
    # now K is [[K11, K12], [K21, K22]]
    m = K.size()[0]
    idx = torch.randperm(m)
    K = K[idx[:kS],:].mm(H) # kS: number of rows to be sampled
    L = L[idx[:kS],:].mm(H) #
    # now K is [[K11, K12]]

    # tr(K_{11}L_{11})
    # K11 = K[:,:kS]
    # L11 = L[:,:kS]
    # K11 = K11.mm(K11.t())
    # L11 = L11.mm(L11.t())
    # K11_i = torch.inverse(K11 + kK_I)
    # L11_i = torch.inverse(L11 + kK_I)
    # Rk11 = (K11.mm(K11_i))
    # Rl11 = (L11.mm(L11_i))
    # R11 = torch.mul(Rk11, Rl11.t())
    # Pxy11 = torch.sum(R11)
    # tr(K_{12}L_{12}^T)
    K12 = K[:,kS:]
    L12 = L[:,kS:]
    K12 = K12.mm(K12.t())
    L12 = L12.mm(L12.t())
    K12_i = torch.inverse(K12 + kK_I)
    L12_i = torch.inverse(L12 + kK_I)
    Rk12 = (K12.mm(K12_i))
    Rl12 = (L12.mm(L12_i))
    R12 = torch.mul(Rk12, Rl12.t())
    Pxy12 = torch.sum(R12)
    return Pxy12
