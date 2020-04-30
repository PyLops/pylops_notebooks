import numpy as npimport matplotlib.pyplot as plt


def AngleGather(d, nfft_k, nalpha, dt, ds, ir, cp, nfft_f=2**10, ifin=10, plotflag=False):    """Angle gathers creation

    Create angle gathers from a local reflectivity response
    
    Parameters
    ----------
    d : :obj:`numpy.ndarray`
        Local reflectivity response of size :math:`[n_t \times n_r \times n_s]`
        with symmetric time axis
    nfft_k : :obj:`int`
        Number of samples in wavenumber axis
    nalpha : :obj:`int`
        Number of angles
    dt : :obj:`float`
        Time sampling
    ds : :obj:`float`
        Spatial sampling along source axis
    dt : :obj:`float`
        Time sampling
    ir : :obj:`int`
        Index of selected receiver
    cp : :obj:`float`
        Local velocity along source line to be used for offset-to-angle conversion 
    nfft_k : :obj:`int`, optional
        Number of samples in frequency axis
    ifin : :obj:`int`, optional
        Index of first frequency
    plotflag : :obj:`bool`, optional
        Plotflag
    
    Returns
    -------
    R : :obj:`numpy.ndarray`
        Angle gather at time zero
    alpha : :obj:`numpy.ndarray`
        Angle axis
    R_alpha : :obj:`numpy.ndarray`
        Offset-to-angle converted receiver gather

    """
    nt = d.shape[0]

    # Define frequency and wavenumber axes 
    f=np.arange(0,nfft_f/2+1)/(nfft_f*dt)
    k=np.arange((-nfft_k/2+1),nfft_k/2+1)/(nfft_k*ds)
    # Extract single receiver gather
    d_tmp = np.fft.fftshift(np.squeeze(d[:, ir, :]),0)
    d = np.zeros((nfft_f, nfft_k))
    d[-nfft_f//2-1:, :] = d_tmp[-nfft_f//2-1:, :]
    d[:nfft_f//2, :] = d_tmp[:nfft_f//2, :]
   
    # Convert from t-x to f-x domain
    D_r = np.fft.fft(d,nfft_f,0)
    D_r = D_r[:nfft_f//2+1,:]
    D_r = np.hstack((D_r[:, ir:], D_r[:, :ir]))
    D_fk = np.fft.fftshift(np.fft.fft(D_r,nfft_k,1),1)
 
    # Convert from f-kx to f-angle
    alpha = np.linspace(-90,90,nalpha)
    sinalpha = np.sin(alpha*np.pi/180)
    Alpha_sampled = np.zeros((nfft_f//2+1,nfft_k))
    D_alpha = np.zeros((nfft_f//2+1,nalpha), dtype=np.complex)
    for iif in np.arange(ifin,nfft_f//2+1):
        sinalpha_sampled = cp*k/f[iif]
        Alpha_sampled[iif,:] = sinalpha_sampled
        D_alpha[iif,:] = np.interp(sinalpha,sinalpha_sampled,np.real(D_fk[iif,:])) \
                         + 1j*np.interp(sinalpha,sinalpha_sampled,np.imag(D_fk[iif,:]))
    D_alpha[np.isnan(D_alpha)] = 0
    # Create angle gather
    R_alpha = np.fft.ifftshift(np.fft.ifft(D_alpha,nfft_f,0),0)
    R = np.sum(D_alpha[0:nfft_f//2-1,:],axis=0)
    if plotflag:
    	fig, axs = plt.subplots(1, 3, figsize=(15, 10))
    	axs[0].imshow(d, cmap='gray')
    	axs[0].set_title('Gather - TX')
    	axs[0].axis('tight')
    	axs[1].imshow(np.real(D_fk), cmap='jet', extent=(k[0], k[-1], f[-1], f[0]))
    	axs[1].set_title('Gather - real FK')
    	axs[1].axis('tight')
    	axs[2].imshow(np.imag(D_fk), cmap='jet', extent=(k[0], k[-1], f[-1], f[0]))
    	axs[2].set_title('Gather - imag FK')
    	axs[2].axis('tight')

    	fig, axs = plt.subplots(1, 4, figsize=(15, 10))
    	axs[0].imshow(np.real(Alpha_sampled), cmap='jet', vmin=-3, vmax=3, extent=(k[0], k[-1], f[-1], f[0]))
    	axs[0].set_title('Alpha conversion - FK')
    	axs[0].axis('tight')
    	axs[1].imshow(np.abs(D_alpha), cmap='jet', extent=(k[0], k[-1], f[-1], f[0]))
    	axs[1].set_title('Gather - FAngle')
    	axs[1].axis('tight')
    	axs[2].imshow(np.real(D_alpha), cmap='jet', extent=(k[0], k[-1], f[-1], f[0]))
    	axs[2].set_title('Gather - realFAngle')
    	axs[2].axis('tight')
    	axs[3].imshow(np.imag(D_alpha), cmap='jet', extent=(k[0], k[-1], f[-1], f[0]))
    	axs[3].set_title('Gather - imagFAngle')
    	axs[3].axis('tight')

    	fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    	axs[0,0].imshow(np.real(R_alpha), cmap='gray')
    	axs[0,0].axhline(nfft_f//2)
    	axs[0,0].axis('tight')
    	axs[0,1].imshow(np.imag(R_alpha), cmap='gray')
    	axs[0,1].axhline(nfft_f//2)
    	axs[0,1].axis('tight')
    	axs[1,0].plot(alpha,np.real(R))
    	axs[1,0].plot(alpha,np.real(R_alpha[nfft_f//2]) * nfft_f, '-r')
    	axs[1,1].plot(alpha,np.imag(R))
    	axs[1,1].plot(alpha,np.imag(R_alpha[nfft_f//2]) * nfft_f, '-r')

    return R, alpha, R_alpha
