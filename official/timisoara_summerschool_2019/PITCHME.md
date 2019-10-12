---
@title[Title]

@snap[west span-95]
## Large-scale inverse problems in geoscience
#### @css[black](Matteo Ravasi)
##### @css[black](Data Assimilation Summer School, Timisoara - 01/08/2019)

---
@title[Quiz]
## @css[blue](Quiz)
<br>
`\[
J = || \mathbf{d} - \mathbf{G} \mathbf{m}||^2 + \lambda^2 ||\mathbf{D} \mathbf{m}||^2
\]`
<br>
How do you minimize this cost function *without* creating explicit matrices?


---
@title[Goals]
## @css[blue](Goals)
<br>
@ul

- Basic of linear algebra (hopefully already...)
- From textbook linear algebra to real-life linear algebra
- Linear algebra in software: **PyLops** <br><br>
- Python is **Slow** if you write is as C, it is not if you leverage its advanced libraries (numpy, scipy, tensorflow, pytorch, pycuda, dask...)
@ulend

---
@title[Course outline]
## @css[blue](Course outline)

@ul

- Motivation: inverse problems in geoscience, how large?
- Basics of inverse problems (@gitlink[EX1](official/timisoara_summerschool_2019/Visual_optimization.ipynb))
- Linear operators: philosophy and implementation (@gitlink[EX2](official/timisoara_summerschool_2019/Linear_Operators.ipynb))
- Least squares solvers (@gitlink[EX3](official/timisoara_summerschool_2019/Solvers.ipynb))
- Sparse solvers (@gitlink[EX4](official/timisoara_summerschool_2019/Solvers.ipynb))
- Geophysical applications (@gitlink[EX5](official/timisoara_summerschool_2019/SeismicRedatuming.ipynb), @gitlink[EX6](developement/SeismicInversion-Volve.ipynb))
- Beyond single-machine inverse problems: GPUs, distributed...

@ulend


---?image=official/timisoara_summerschool_2019/assets/images/lifecycle.png&size=70% 70%&color=#ffffff
@title[Lifecycle]
+++?image=official/timisoara_summerschool_2019/assets/images/lifecycle_highlight.png&size=70% 70%&color=#ffffff
@title[Lifecycle]

---
@title[Motivation]
## @css[blue](Large... how large?)
<br>
- Seismic processing:
`\[ n_S = n_R = 10^3, n_t = n_{fft} = 2 \cdot 10^3 \quad (dt = 4 ms, t_{max} = 8 s)
\]`
`\[ \rightarrow \mathbf{G}: n_S \cdot n_R \cdot n_{fft}^2 = 4 \cdot 10^{12} * 32 bit = 128 TB
\]`
<br>
- Seismic inversion:
`\[ n_x = n_y = 10^3, n_z = 800 \quad (dz = 5 m, z_{max} = 4000 m)
\]`
`\[ \rightarrow \mathbf{G}: n_x \cdot n_y \cdot n_z^2 = 6.4 \cdot 10^{11} * 32 bit \sim 20 TB
\]`

---
@title[Inverse problems 1]
## @css[blue](Inverse problems)
Fundamental theory developed in the '60/'70

<br>
@ul

- Mathematicians: @size[25px](general theory, shared with other applications e.g., ML, data assimilation)
- Seismologists: @size[25px](inference of subsurface properties/behaviours from large set of data)
- Atmosferic scientists: @size[25px](more concerned with data assimilation through time)

@ulend

+++
@title[Inverse problems 2]
## @css[blue](Inverse problems)
References:
<br><br>

@size[30px](A. Tarantola, Inverse Problem Theory;)

@size[30px](S. Kay, Fundamentals of statistical signal processing;)

@size[30px](J. Claerbout, Basic Earth Imaging;)

@size[30px](...)


+++
@title[Inverse problems 3]

#### Forward model
<br>
`\[ \mathbf{d} = \mathbf{g} (\mathbf{m}) \quad / \quad \mathbf{d} = \mathbf{G} \mathbf{m}
\]`

@ul

- **d**: observed data. @size[30px](Quantity that we can physicially measure - ) @size[25px](seismic, production, temperature, precipitation, MRI scan...)
- **m**: model. @size[30px](Quantity that we are interested to know and we think affects the data - ) @size[25px](rock properties, pressure, saturation, human organs...)
- **G**: modelling operator. @size[30px](Set of equations that we think can explain the data by *nonlinear/linear combination* of model parameters - ) @size[25px](PDEs, seismic convolution model, camera blurring...)

@ulend


+++
@title[Inverse problems 4]

#### Forward model
<br>
`\[ \mathbf{d} = \mathbf{g} (\mathbf{m}) \quad / \quad \mathbf{d} = \mathbf{G} \mathbf{m}
\]`

@size[30px](Unique and *easy* to compute, perhaps time consuming...)

<br>

#### Inverse model
<br>
`\[ \mathbf{m} = \mathbf{g}^{-1} (\mathbf{d}) \quad / \quad \mathbf{m} = \mathbf{G}^{-1} \mathbf{d}
\]`

@size[30px](Nonunique and hard to *compute*, both expensive and unstable...)


+++
@title[Inverse problems 5]

@ul

- **square** (N = M): @size[26px](rarely the case in real life. Solution:) `\[ \mathbf{G}^{-1} \rightarrow \mathbf{m}_{est} = \mathbf{G}^{-1} \mathbf{d}  \]` <br>
- **overdetermined** (N &#62; M): @size[26px](most common case, robust to noise as more data points than model parameters. Least-squares solution:) `\[ J = || \mathbf{d} - \mathbf{G} \mathbf{m}||^2 \rightarrow \mathbf{m}_{est} = (\mathbf{G^H G})^{-1} \mathbf{G^H} \mathbf{d} \]`
- **underdetermined** (N &#60; M): @size[26px](not ideal, but sometimes only option - e.g., MRI scan. Least-squares solution:) `\[ J = || \mathbf{m}||^2 \quad s.t \quad  \mathbf{d} = \mathbf{G} \mathbf{m} \rightarrow \mathbf{m}_{est} =  \mathbf{G^H} (\mathbf{G G^H})^{-1}\mathbf{d} \]` @size[26px](Sparse solution:) `\[ J = || \mathbf{m}|| \quad s.t \quad  \mathbf{d} = \mathbf{G} \mathbf{m} \]`
@ulend

+++
@title[Inverse problems 6]
#### Inversion in practice

@size[30px](Now that we know the analytical expressions, how do we find)

`\[ \mathbf{G}^{-1} \quad (\mathbf{G^H G})^{-1} \quad (\mathbf{G G^H})^{-1}\]`

@size[30px](Solutions studied in the fields of *Numerical analysis and Optimization*.)
<br><br>
@size[30px](Two families of solvers:)

@ul
- **direct**: @size[26px](LU, QR, SVD...)
- **iterative**: @size[26px](gradient based methods - CG, LSQR, GMRES...)
@ulen

+++
@title[Inverse problems 7]
#### Inversion in practice

![Gradient_methods](official/timisoara_summerschool_2019/assets/images/gradient_methods.png)
Let's practice @gitlink[EX1](official/timisoara_summerschool_2019/Visual_optimization.ipynb).

+++
@title[Inverse problems 8]
#### Why iterative?

<br>

**G** is too large to be inverted (or solved explicitely)

**G** is too large to be stored in memory
<br><br>
*Take on message:*
`\[ \mathbf{m}_{est} = \sum_{i=0}^{N_{iter}} f(\mathbf{G}, \mathbf{d}, \mathbf{m}_0) \qquad \mathbf{G} \mathbf{m}, \mathbf{G}^H \mathbf{d}, (\mathbf{m}^H \mathbf{m} , \mathbf{d}^H \mathbf{d})\]`


---
@title[Linear operators 1]
## @css[blue](Linear Operators)
A piece of computer code that can perform *forward* and *adjoint* operations without
the need to store an *explicit matrix*:
`\[  \mathbf{G} \mathbf{m}, \mathbf{G}^H \mathbf{d} \]`

<br>

@size[25px](Very powerful, sometimes underutilized concept...
but how do we make sure that forward and adjoint are correctly implemented? --> **Dot-Test**

+++
@title[Linear operators 2]
#### Ex: Diagonal
<br>
`\[ \mathbf{D} = \begin{bmatrix}
            d_{1} & 0     & ... & 0 \\
            0     & d_{2} & ... & 0 \\
            ...                     \\
            0     & 0 & ... & d_{N} \\
        \end{bmatrix},
        \mathbf{D}^H = \begin{bmatrix}
            d_{1} & 0     & ... & 0 \\
            0     & d_{2} & ... & 0 \\
            ...                     \\
            0     & 0 & ... & d_{N} \\
        \end{bmatrix} \]`

@snap[south span-50 text-09]
```python
# forward
def _matvec(x)
    return self.diag * x
# adjoint
def _matvec(x)
    return self.diag * x
```
@[1-3]
@[4-7]
<br><br>
@snapend


+++
@title[Linear operators 3]
#### Ex: Diagonal

<br>

**Dot-Test**: a correct implementation of forward and adjoint for
a linear operator should verify the the following *equality*
within a numerical tolerance:

<br>

`\[ (\mathbf{G} \cdot  \mathbf{u})^H \mathbf{v} = \mathbf{u}^H (\mathbf{G}^H \cdot \mathbf{v})
\]`

+++
@title[Linear operators 4]
#### Ex: First derivative
<br>
`\[ \mathbf{D} = \begin{bmatrix}
            -1      & 1  & ... & 0   & 0 \\
            -0.5    & 0  & 0.5 & ... & 0 \\
            ...                           \\
            0       & 0  & ... & -1  & 1 \\
        \end{bmatrix}
\]`

@snap[south span-73 text-08]
```python
def _matvec(x):
x, y = x.squeeze(), np.zeros(self.N, self.dtype)
y[1:-1] = (0.5 * x[2:] - 0.5 * x[0:-2]) / self.sampling
# edges
y[0] = (x[1] - x[0]) / self.sampling
y[-1] = (x[-1] - x[-2]) / self.sampling
```
<br><br>
@snapend


+++
@title[Linear operators 5]
#### Ex: First derivative
<br>
`\[ \mathbf{D}^H = \begin{bmatrix}
            -1    & -0.5  & ...   & 0    & 0 \\
            1     & 0     &  -0.5 & ...  & 0 \\
            ...                             \\
            0     & 0     & ...   & -0.5 & 1 \\
        \end{bmatrix}
\]`

@snap[south span-73 text-08]
```python
def _rmatvec(x):
x, y = x.squeeze(), np.zeros(self.N, self.dtype)
y[0:-2] -= (0.5 * x[1:-1]) / self.sampling
y[2:] += (0.5 * x[1:-1]) / self.sampling
# edges
y[0] -= x[0] / self.sampling
y[1] += x[0] / self.sampling
y[-2] -= x[-1] / self.sampling
y[-1] += x[-1] / self.sampling
```
<br>
@snapend


+++
@title[Linear operators 6]
<br><br><br><br>
Let's practice @gitlink[EX2](official/timisoara_summerschool_2019/Linear_Operators.ipynb).
<br>

---
@title[Solvers 1]
## @css[blue](Solvers)

<br>
@ul
- **Least-squares**: regularized, preconditioned, bayesian
- **L1**: sparsity promoting, blockiness promoting
@ulend

@snap[south span-60 text-08]
![Cost_functions](official/timisoara_summerschool_2019/assets/images/cost_functions.png)
@snapend


+++
@title[Solvers 2]
#### Least-squares - Regularized inversion
Add information to the inverse problem --> mitigate *ill-posedness*

<br>
`\[ J = || \mathbf{d} - \mathbf{G} \mathbf{m}||^2_{\mathbf{W}_d} + \sum_i{\epsilon_{R_i}^2 ||\mathbf{d}_{R_i} - \mathbf{R}_i \mathbf{m}||^2_{\mathbf{W}_{R_i}}}
\]`

`\[
\begin{bmatrix}
            \mathbf{W}^{1/2}_d \mathbf{G}    \\
            \epsilon_{R_1} \mathbf{W}^{1/2}_{R_1} \mathbf{R}_1 \\
            ...   \\
            \epsilon_{R_N} \mathbf{W}^{1/2}_{R_N} \mathbf{R}_N
        \end{bmatrix} \mathbf{m} =
        \begin{bmatrix}
            \mathbf{W}^{1/2}_d \mathbf{d}    \\
            \epsilon_{R_1} \mathbf{W}^{1/2}_{R_1} \mathbf{d}_{R_1} \\
            ...   \\
            \epsilon_{R_N} \mathbf{W}^{1/2}_{R_N} \mathbf{d}_{R_N}
        \end{bmatrix}
\]`


+++
@title[Solvers 3]
#### Least-squares - Regularized inversion
Add information to the inverse problem --> mitigate *ill-posedness*

@snap[midpoint span-70 text-10]
<br><br>
```python
def RegularizedInversion(G, Reg, d, dreg, epsR):
# operator
Gtot = VStack([G, epsR * Reg])
# data
dtot = np.hstack((d, epsR * dreg))
# solver
minv = lsqr(Gtot, dtot)[0]
```
<br>
@snapend


+++
@title[Solvers 4]
#### Least-squares - Bayesian inversion
Add prior information to the inverse problem --> mitigate *ill-posedness*

<br>
`\[ J = || \mathbf{d} - \mathbf{G} \mathbf{m}||^2_{\mathbf{C}_d^{-1}} + ||\mathbf{m}_{0} - \mathbf{m}||^2_{\mathbf{C}_m^{-1}}
\]`
<br>
`\[
\begin{bmatrix}
            \mathbf{C}^{-1/2}_d \mathbf{G}    \\
            \mathbf{C}^{-1/2}_{m} \\
        \end{bmatrix} \mathbf{m} =
        \begin{bmatrix}
            \mathbf{C}^{-1/2}_d \mathbf{d}    \\
            \mathbf{C}^{-1/2}_{m} \mathbf{m}_0
        \end{bmatrix} \rightarrow
\]`

`\[
\mathbf{m} = (\mathbf{G}^H \mathbf{C}^{-1}_d \mathbf{G} + \mathbf{C}^{-1}_m)^{-1}
(\mathbf{G}^H \mathbf{C}^{-1}_d \mathbf{d} + \mathbf{C}^{-1}_m \mathbf{m_0})
\]`


+++
@title[Solvers 5]
#### Least-squares - Bayesian inversion
Add prior information to the inverse problem --> mitigate *ill-posedness*

<br>
`\[ J = || \mathbf{d} - \mathbf{G} \mathbf{m}||^2_{\mathbf{C}_d^{-1}} + ||\mathbf{m}_{0} - \mathbf{m}||^2_{\mathbf{C}_m^{-1}}
\]`
<br>

`\[
\mathbf{m} = \mathbf{m_0} + \mathbf{C}_m \mathbf{R}^H (\mathbf{R} \mathbf{C}_m \mathbf{R}^H +
\mathbf{C}_d)^{-1} (\mathbf{d} - \mathbf{R} \mathbf{m_0})
\]`


+++
@title[Solvers 6]
#### Least-squares - Bayesian inversion
Add prior information to the inverse problem --> mitigate *ill-posedness*

@snap[midpoint span-70 text-10]
<br><br>
```python
def BayesianInversion(G, d, Cm, Cd):
# operator
Gbayes = G * Cm * G.H + Cd
# data
dbayes = d - G * m0
# solver
minv = m0 + Cm * G.H * lsqr(Gbayes, dbayes)[0]
```
<br>
@snapend


+++
@title[Solvers 7]
#### Least-squares - Preconditioned inversion
Limit the range of plausible models --> mitigate *ill-posedness*

<br>
`\[ J = || \mathbf{d} - \mathbf{G} \mathbf{P} \mathbf{p}||^2
\]`
<br>
`\[
\mathbf{m} = \mathbf{P} \mathbf{p}
\]`


+++
@title[Solvers 8]
#### Least-squares - Preconditioned inversion
Limit the range of plausible models --> mitigate *ill-posedness*

@snap[midpoint span-70 text-10]
<br><br>
```python
def PreconditionedInversion(G, P, d):
# operator
Gtot = G * P
# solver
minv = lsqr(Gtot, d)[0]
```
<br><br>
@snapend


+++
@title[Solvers 8]
#### Sparsity
Introduce L1 norms to cost function

<br>
@ul
- **Data misfit term**: outliers `\[ J = || \mathbf{d} - \mathbf{G} \mathbf{m}||^2 \]`
- **Model**: sparse model `\[ J = || \mathbf{d} - \mathbf{G} \mathbf{m}||^2 + \lambda^2 ||\mathbf{m}|| \]`
- **Projected Model**: e.g. blocky model (projection = first derivative) `\[ J = || \mathbf{d} - \mathbf{G} \mathbf{m}||^2 + \lambda^2 ||\mathbf{D} \mathbf{m}|| \]`
@ulend
<br>

+++
@title[Solvers 9]
<br><br><br><br>
Let's practice @gitlink[EX3-EX4](official/timisoara_summerschool_2019/Solvers.ipynb).
<br>


---
@title[Geophysical applications - MDC 1]
#### Seismic redatuming

@snap[midpoint span-80]
![MDC](official/timisoara_summerschool_2019/assets/images/mdc.png)
@snapend

+++
@title[Geophysical applications - MDC 2]
#### Seismic redatuming

Integral relation:
`\[
g^-(t, x_s, x_v) = \mathscr{F}^{-1} \Big( \int_S g^+(f, x_s, x_r)
        \mathscr{F}(R(t, x_r, x_v)) dr \Big)
\]`

<br>
Discretized relation:
`\[
\mathbf{G^-}= \mathbf{\hat{G}^+} \mathbf{R}
\]`

where:
`\[
\mathbf{\hat{G}^+}= \mathbf{F}^H  \mathbf{G^+} \mathbf{F}
\]`
<br>
Let's practice @gitlink[EX5](official/timisoara_summerschool_2019/SeismicRedatuming.ipynb).


---
@title[Geophysical applications - Seismic inversion 1]
#### Seismic inversion

@snap[midpoint span-80]
![SeismicInversion](official/timisoara_summerschool_2019/assets/images/seismic_inversion.png)
@snapend

+++
@title[Geophysical applications - Seismic inversion 1]
#### Seismic inversion

Integral relation:
`\[
d(t) =  w(t) * \frac{d(ln(AI(t))}{dt}
\]`

<br>
Discretized relation:
`\[
\mathbf{d}=  \mathbf{W} \mathbf{D} \mathbf{ai}
\]`

where **D** is a derivative operator and **W** is a convolution operator.
<br><br>
Let's practice @gitlink[EX6](developement/SeismicInversion-Volve.ipynb).

---
@title[Implementation - PyLops]
#### Some words about implementation...

@ul

- Solving large-scale inverse problems can be daunting --> *Divide and conquer* paradigm
- Focus on fast operators as well as on advanced solvers
- Various paradigms (deterministic, bayesian..) can share same frameworks

@ulend

@snap[south span-60]
![PyLops](official/timisoara_summerschool_2019/assets/images/pylops.png)
@snapend

---
@title[Implementation - GPUs]
#### Beyond laptop usage - GPUs

@ul
- Computational cost of PyLops: forward and adjoint passes (dot products...)
- Several operators are convolution filters in disguise --> leverage ML libraries
- @gitlink[Seismic inversion example](development-cuda/SeismicInversion.ipynb): **d** = **W** **D** **m**, **W**: convolution with w, **D**: derivative = convolution with [-1, 1]
@ulend

@ul
- *TensorFlow*, **PyTorch**, Cupy, PyCuda...
@ulend

---
@title[Implementation - Distributed]
#### Beyond laptop usage - Distributed computing

@ul
- Data and model can outreach RAM size --> distribute
- Several operators can be easily parallelized
- Solvers can be partially parallelized
- MDC example: *G+* and *G-* can reach 100++ GB for 3D seismic acquisition, **G+** is a batched matrix-matrix multiplication
@ulend

@ul
- Joblib, mpi4py, **Dask**...
@ulend
