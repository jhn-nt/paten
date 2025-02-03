# PROSEVA Trial Emulation
## Assessing the protective effect of pronation in ARDS patients from observational retrospective data
G. Angelotti, L. Azzimonti, I. Bose, A. Colombo, T. Crupi, F. Faraci, M. Lubian, G. Maroni, A. Pagnamenta, L. Ruinelli, N. Stomeo, R. Švihrová

### Installation 
`pip install "git+https://github.com/jhn-nt/paten.git"`

```python
from paten.etl import dataset, intervention_proxy__uniform,intervention_proxy__capped_cumulative

# estimating proxies of pronation as total pronations hours/ total imv
df=dataset(intervention_proxy__uniform)


# estimating proxies of pronation as cumulative pronation hours until 24 hours.
df=dataset(intervention_proxy__capped_cumulative)
```