# PROSEVA Trial Emulation
## Assessing the protective effect of pronation in ARDS patients from observational retrospective data
G. Angelotti, L. Azzimonti, I. Bose, A. Colombo, T. Crupi, F. Faraci, M. Lubian, G. Maroni, A. Pagnamenta, L. Ruinelli, N. Stomeo, R. Švihrová



### Requirements
1. Access to [AmsterdamUMCdb](https://github.com/AmsterdamUMC/AmsterdamUMCdb). Learn more [here](https://amsterdammedicaldatascience.nl/).
2. Project ID of a Google Project, make sure to have the necessary IAM permissions to run queries on Big Query.
3. Ensure that the [Google SDK](https://cloud.google.com/sdk?hl=it) is properly installed in your environment with proper authentication.

Important Note: The google account enabled to access the AmsterdamUMCdb must the be same as the one associated with the Google Project.

### Installation 
Run:   
`pip install "git+https://github.com/jhn-nt/paten.git"`

Afterwards run:
`python3 -m paten.install`

### Tutorial
To investigate the dataset:
```python
from paten.etl import dataset, intervention_proxy__uniform,intervention_proxy__capped_cumulative, intervention_proxy__duty_cycle

# estimating proxies of pronation as total cumulative pronations hours/ IMV duration.
df=dataset(intervention_proxy__uniform)

# estimating proxies of pronation as cumulative pronation hours until 24 hours.
df=dataset(intervention_proxy__capped_cumulative)

# estimating proxies of pronation hours as a duty cycle over a pseudo 24 hours cycle.
df=dataset(intervention_proxy__duty_cycle)
```

To replicate the entire study: 
```bash
python3 -m paten.pjob --dir savedir --seed 0 
```
This will take several ours.



### Limitations
Our code was tested on a Linux environment only, milage may vary.