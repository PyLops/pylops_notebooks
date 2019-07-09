---?color=#ffffff
@title[Title]

@snap[west span-95]
## @css[blue](Large scale inverse problems in geoscience)
#### @css[black](Matteo Ravasi)
##### @css[black](Data Assimilation Summer School, Timisoara - 01/08/2019)

---?color=#ffffff
@title[Quiz]
## @css[blue](Quiz)
<br>
`\[
J = || \mathbf{d} - \mathbf{G} \mathbf{m}||_2 + \lambda^2 ||\mathbf{D} \mathbf{m}||_2
\]`

How do you implement **D** without an explicit matrix?


---?color=#ffffff
@title[Course outline]
## @css[blue](Course outline)

@ul

- Motivation: inverse problems in geoscience, how large?
- Basics of inverse problems
- Linear operators: philosophy and implementation (@gitlink[EX1](official/timisoara_summerschool_2019/quiz_solution.ipynb), @gitlink[EX2](official/timisoara_summerschool_2019/blurring.ipynb))
- Least squares solvers (@gitlink[EX3](official/timisoara_summerschool_2019/blurring.ipynb))
- Sparse solvers (@gitlink[EX4](official/timisoara_summerschool_2019/blurring.ipynb))
- Geophysical applications (@gitlink[EX5](developement/WaveEquationProcessing_new_and_comparison.ipynb), @gitlink[EX6](developement/SeismicInversion-Volve.ipynb))
- Beyond single-machine inverse problems: GPUs, distributed...

@ulend


---?image=official/timisoara_summerschool_2019/assets/images/lifecycle.png&size=70% 70%&color=#ffffff

---?color=#ffffff
@title[Motivation]
## @css[blue](Large... how large?)
<br>
- Seismic processing
`\[ n_S = n_R = 10^3, n_t = n_{fft} = 2 \cdot 10^3, \quad (dt = 4 ms, t_{max} = 8 s)
\]`
`\[ \rightarrow \mathbf{G}: n_S \cdot n_R \cdot n_{fft}^2 = 4 \cdot 10^{12} * 32 bit = 128 TB
\]`
<br>
- Seismic inversion:
`\[ n_x = n_y = 10^3, n_z = 800, \quad (dz = 5 m, t_{max} = 4000 m)
\]`
`\[ \rightarrow \mathbf{G}: n_x \cdot n_y \cdot n_z^2 = 6.4 \cdot 10^{11} * 32 bit \sim 20 TB
\]`

---?color=#ffffff
@title[Inverse problems 1]
## @css[blue](Inverse problems)
Fundamental theory developed in the '60/'70

<br>
@ul

- Mathematicians: @size[25px](general theory, shared with other applications e.g., ML, data assimilation)
- Seismologists: @size[25px](inference of subsurface properties/behaviours from large set of data)
- Atmosferic scientists: @size[25px](more concerned with data assimilation through time)

@ulend

+++?color=#ffffff
@title[Inverse problems 2]
## @css[blue](Inverse problems)
References:
<br><br>

@size[30px](A. Tarantola, Inverse Problem Theory;)

@size[30px](S. Kay, Fundamentals of statistical signal processing;)

@size[30px](J. Claerbout, Basic Earth Imaging;)

@size[30px](...)


+++?color=#ffffff
@title[Inverse problems 3]

#### Forward model
<br>
`\[ \mathbf{d} = \mathbf{g} (\mathbf{m}) \quad / \quad \mathbf{d}_{[N \times 1]} = \mathbf{G}_{[N \times M]} \mathbf{m}_{[M \times 1]}
\]`

@ul

- **d**: observed data. @size[30px](Quantity that we can physicially measure) @size[25px]((seismic, production, temperature, precipitation, MRI scan...))
- **m**: model. @size[30px](Quantity that we are interested to know and we think affects the data) @size[25px]((rock properties, pressure, saturation, human organs...)
- **G**: modelling operator. @size[30px](Set of equations that we think can explain the data by *nonlinear/linear combination* of model parameters) @size[25px]((PDEs, seismic convolution model, camera blurring...))

@ulend


+++?color=#ffffff
@title[Inverse problems 4]

#### Forward model
<br>
`\[ \mathbf{d} = \mathbf{g} (\mathbf{m}) \quad / \quad \mathbf{d}_{[N \times 1]} = \mathbf{G}_{[N \times M]} \mathbf{m}_{[M \times 1]}
\]`

@size[30px](Unique and *easy* to compute, perhaps time consuming...)

<br>

#### Inverse model
<br>
`\[ \mathbf{m} = \mathbf{g}^{-1} (\mathbf{d}) \quad / \quad \mathbf{m} = \mathbf{G}^{-1} \mathbf{d}
\]`

@size[30px](Nonunique and hard to *compute*, both expensive and unstable...)


+++?color=#ffffff
@title[Inverse problems 5]

@ul

- **square** (N = M): @size[26px](rarely the case in real life. Solution:) `\[ \mathbf{G}^{-1} \rightarrow \mathbf{m}_{est} = \mathbf{G}^{-1} \mathbf{d}  \]`
- **overdetermined** (N &#62; M): @size[26px](most common case, robust to noise as more data points than model parameters. Least-squares solution:) `\[ J = || \mathbf{d} - \mathbf{G} \mathbf{m}||_2 \rightarrow \mathbf{m}_{est} = (\mathbf{G^H G})^{-1} \mathbf{G^H} \mathbf{d} \]`
- **underdetermined** (N &#60; M): @size[26px](not ideal, but sometimes only option - e.g., MRI scan. Least-squares solution:) `\[ J = || \mathbf{m}||_2 \quad s.t \quad  \mathbf{d} = \mathbf{G} \mathbf{m} \rightarrow \mathbf{m}_{est} =  \mathbf{G^H} (\mathbf{G G^H})^{-1}\mathbf{d} \]` @size[26px](Sparse solution:) `\[ J = || \mathbf{m}||_1 \quad s.t \quad  \mathbf{d} = \mathbf{G} \mathbf{m} \]`
@ulen

+++?color=#ffffff
@title[Inverse problems 6]
#### Inversion in practice

@size[30px](Now that we know the analytical expressions, how do we find)

`\[ \mathbf{G}^{-1} \quad \mathbf{G^H G}^{-1} \quad \mathbf{G G^H}^{-1}\]`

@size[30px](Solutions studied in the fields of *Numerical analysis and Optimization*.)
<br><br>
@size[30px](Two families of solvers:)

@ul
- **direct**: @size[26px](LU, QR, SVD...)
- **iterative**: @size[26px](gradient based methods - CG, LSQR, GMRES...)
@ulen

+++?color=#ffffff
@title[Inverse problems 7]
#### Inversion in practice

![Costfunction](official/timisoara_summerschool_2019/assets/images/gradient_methods.png)