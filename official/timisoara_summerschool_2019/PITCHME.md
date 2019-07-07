---?color=#ffffff
@title[Title]

## @css[black](Large scale inverse problems in geoscience)
#### @css[black](Matteo Ravasi)
##### @css[black](Data Assimilation Summer School, Timisoara - 01/08/2019)

---?color=#ffffff
@title[Quiz]
## @css[black](Quiz)

`\[
J = || \mathbf{P} - \mathbf{G} \mathbf{m}||^2 + \lambda^2 ||\mathbf{D} \mathbf{m}||^2
\]`

How do you implement **D** without an explicit matrix?


---?color=#ffffff
@title[Course outline]
## @css[black](Course outline)

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
## @css[black](Large... how large?)

Seismic processing:
`\[ n_S = n_R = 10^3, n_t = n_{fft} = 2 \cdot 10^3 (dt = 4 ms, t_{max} = 8 s)
\]`
`\[ \rightarrow \mathbf{G}: n_S \cdot n_R \cdot \cdot n_{fft}^2 = 4 \cdot 10^{12} * 32 bit = 128 TB
\]`

Seismic inversion:
`\[ n_x = n_y = 10^3, n_z = 800, \quad (dz = 5 m, t_{max} = 4000 m)
\]`
`\[ \rightarrow \mathbf{G}: n_x \cdot n_y \cdot n_z^2 = 6.4 \cdot 10^{11} * 32 bit \sim 20 TB
\]`