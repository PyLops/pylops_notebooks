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
J = || \mathbf{d} - \mathbf{G} \mathbf{m}||_2 + \lambda^2 ||\mathbf{D} \mathbf{m}||_2
\]`

How do you implement **D** without an explicit matrix?


---
@title[Course outline]
## @css[blue](Course outline)

@ul

- Motivation: inverse problems in geoscience, how large?
- Basics of inverse problems (@gitlink[EX1](official/timisoara_summerschool_2019/Visual_optimization.ipynb))
- Linear operators: philosophy and implementation (@gitlink[EX2](official/timisoara_summerschool_2019/Linear_Operators.ipynb))
- Least squares solvers (@gitlink[EX3](official/timisoara_summerschool_2019/Solvers.ipynb))
- Sparse solvers (@gitlink[EX4](official/timisoara_summerschool_2019/Solvers.ipynb))
- Geophysical applications (@gitlink[EX5](developement/WaveEquationProcessing_new_and_comparison.ipynb), @gitlink[EX6](developement/SeismicInversion-Volve.ipynb))
- Beyond single-machine inverse problems: GPUs, distributed...

@ulend


---?image=official/timisoara_summerschool_2019/assets/images/lifecycle.png&size=70% 70%&color=#ffffff

---
@title[Motivation]
## @css[blue](Large... how large?)
<br>
- Seismic processing
`\[ n_S = n_R = 10^3, n_t = n_{fft} = 2 \cdot 10^3 \quad (dt = 4 ms, t_{max} = 8 s)
\]`
`\[ \rightarrow \mathbf{G}: n_S \cdot n_R \cdot n_{fft}^2 = 4 \cdot 10^{12} * 32 bit = 128 TB
\]`
<br>
- Seismic inversion:
`\[ n_x = n_y = 10^3, n_z = 800 \quad (dz = 5 m, t_{max} = 4000 m)
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

- **d**: observed data. @size[30px](Quantity that we can physicially measure) @size[25px]((seismic, production, temperature, precipitation, MRI scan...))
- **m**: model. @size[30px](Quantity that we are interested to know and we think affects the data) @size[25px]((rock properties, pressure, saturation, human organs...)
- **G**: modelling operator. @size[30px](Set of equations that we think can explain the data by *nonlinear/linear combination* of model parameters) @size[25px]((PDEs, seismic convolution model, camera blurring...))

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

- **square** (N = M): @size[26px](rarely the case in real life. Solution:) `\[ \mathbf{G}^{-1} \rightarrow \mathbf{m}_{est} = \mathbf{G}^{-1} \mathbf{d}  \]`
- **overdetermined** (N &#62; M): @size[26px](most common case, robust to noise as more data points than model parameters. Least-squares solution:) `\[ J = || \mathbf{d} - \mathbf{G} \mathbf{m}||_2 \rightarrow \mathbf{m}_{est} = (\mathbf{G^H G})^{-1} \mathbf{G^H} \mathbf{d} \]`
- **underdetermined** (N &#60; M): @size[26px](not ideal, but sometimes only option - e.g., MRI scan. Least-squares solution:) `\[ J = || \mathbf{m}||_2 \quad s.t \quad  \mathbf{d} = \mathbf{G} \mathbf{m} \rightarrow \mathbf{m}_{est} =  \mathbf{G^H} (\mathbf{G G^H})^{-1}\mathbf{d} \]` @size[26px](Sparse solution:) `\[ J = || \mathbf{m}||_1 \quad s.t \quad  \mathbf{d} = \mathbf{G} \mathbf{m} \]`
@ulen

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

![Costfunction](official/timisoara_summerschool_2019/assets/images/gradient_methods.png)

+++
@title[Inverse problems 8]
#### Why iterative?

<br>

**G** is too large to be inverted (or solved explicitely)

**G** is too large to be stored in memory
<br><br>
*Take on message:*
`\[ \mathbf{m}_{est} = \sum_{i=0}^{N_{iter}} f(\mathbf{G}, \mathbf{d}, \mathbf{m}_0) \qquad \mathbf{G} \mathbf{m}, \mathbf{G}^H \mathbf{d}, (\mathbf{m}^T \mathbf{m} , \mathbf{d}^T \mathbf{d})\]`


---
@title[Linear operators 1]
## @css[blue](Linear Operators)
A piece of computer code that can perform *forward* and *adjoint* operations without
the need to store an *explicit matrix*:
`\[  \mathbf{G} \mathbf{m}, \mathbf{G}^H \mathbf{d} \]`

<br>

@size[25px](Very powerful, sometimes underutilized concept...
but how do we make sure that forward and adjoint are correctly implemented? --> **DOT TEST**

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
        \mathbf{D}^T = \begin{bmatrix}
            d_{1} & 0     & ... & 0 \\
            0     & d_{2} & ... & 0 \\
            ...                     \\
            0     & 0 & ... & d_{N} \\
        \end{bmatrix} \]`

@snap[south span-60 text-08]
@code[python zoom-13 code-max code-shadow](official/timisoara_summerschool_2019/assets/codes/diagonal.py)
@[1-3]
@[4-7]
<br>
@snapend


+++
@title[Linear operators 3]
#### Ex: Diagonal

<br>

**DOT TEST**: a correct implementation of forward and adjoint for
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
            -1 & 1     & ... &  0 &0 \\
            -0.5     & 0 &  0.5 & ... & 0 \\
            ...                     \\
            0     & 0 & ... & -1 & 1 \\
        \end{bmatrix}
\]`

@snap[south span-73 text-08]
@code[python zoom-13 code-max code-shadow](official/timisoara_summerschool_2019/assets/codes/firstderivative_forward.py)
<br>
@snapend


+++
@title[Linear operators 5]
#### Ex: First derivative
<br>
`\[ \mathbf{D}^T = \begin{bmatrix}
            -1 & -0.5     & ... &  0 & 0 \\
            1     & 0 &  0.5 & ... & 0 \\
            ...                     \\
            0     & 0 & ... & -0.5 & 1 \\
        \end{bmatrix}
\]`

@snap[south span-60 text-08]
@code[python zoom-13 code-max code-shadow](official/timisoara_summerschool_2019/assets/codes/firstderivative_adjoint.py)
<br>
@snapend

---
@title[Solvers 1]
## @css[blue](Solvers)

<br>

@ul

- **Least-squares**: regularized, preconditioned
- **L1**: sparsity promoting, blockiness promoting

@ulend

+++
@title[Solvers 2]
#### Least-squares - Regularized inversion
Add information to the inverse problem --> mitigate *ill-posedness*

<br>
`\[ J = || \mathbf{d} - \mathbf{G} \mathbf{m}||^2_{\mathbf{W}_d} + \sum_i{\epsilon_{R_i}^2 ||\mathbf{d}_{R_i} - \mathbf{R}_i \mathbf{m}||^2_{\mathbf{W}_{R_i}}}
\]`
<br>
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
@title[Solvers 2]
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
        \end{bmatrix}
\]`


+++
@title[Solvers 2]
#### Least-squares - Preconditioned inversion
Limit the range of plausible models --> mitigate *ill-posedness*

<br>
`\[ J = || \mathbf{d} - \mathbf{G} \mathbf{P} \mathbf{p}||^2
\]`
<br>
`\[
\mathbf{m} = \mathbf{P} \mathbf{p}
\]`
