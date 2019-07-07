import numpy as np

def steepest_descent(G, d, niter=10, m0=None):
	"""Steepest descent  for minimizing ||d - Gm||^2 with square G
	"""
	n = d.size
	if m0 is None:
		m = np.zeros_like(d)
	else:
		m = m0.copy()
	mh = np.zeros((niter + 1, n))
	mh[0] = m0.copy()
	
	r = d - np.dot(G, m)
	for i in range(niter):
		a = np.dot(r, r) / np.dot(r, np.dot(G, r))
		m = m + a*r
		mh[i + 1] = m.copy()
		r = d - np.dot(G, m)
	return mh
	
	
def conjgrad(G, d, niter=10, m0=None):
	"""Conjugate-gradient for minimizing ||d - Gm||^2 with square G
	"""
	n = d.size
	if m0 is None:
		m = np.zeros_like(d)
	else:
		m = m0.copy()
	mh = np.zeros((niter + 1, n))
	mh[0] = m0.copy()
	
	r = d - G.dot(m)
	d = r.copy()
	k = r.dot(r)
	for i in range(niter):
		Gd = G.dot(d)
		dGd = d.dot(Gd)
		a = k / dGd
		m = m + a*d
		mh[i + 1] = m.copy()
		r -= a*Gd
		kold = k
		k = r.dot(r)
		b = k / kold
		d = r + b*d
	return mh
	