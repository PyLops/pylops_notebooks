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

---?image=official/timisoara_summerschool_2019/assets/images/lifecycle.png&size=70% 70%&color=#ffffff


---?color=#ffffff
@title[Quiz]
## @css[black](Course outline)

@ul

- Motivation: inverse problems in geoscience, how large?
- Basics of nverse problems
- Linear operators: philosophy and implementation (@gitlink[EX1](official/timisoara_summerschool_2019/quiz_solution.ipynb), @gitlink[EX2](official/timisoara_summerschool_2019/blurring.ipynb))
- Least squares solvers (@gitlink[EX3](official/timisoara_summerschool_2019/blurring.ipynb))
- Sparse solvers (@gitlink[EX4](official/timisoara_summerschool_2019/blurring.ipynb))
- Geophysical applications (@gitlink[EX5](developement/WaveEquationProcessing_new_and_comparison.ipynb), @gitlink[EX6](developement/SeismicInversion-Volve.ipynb))
- Beyond single-machine inverse problems: GPUs, distributed...

@ulend