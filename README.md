# Variogram calculation
```python
import pandas as pd
import json
from sgs import (NormalScoreTransform, VariogramParams, experimental_variogram,
	         spherical, exponential, gaussian_model, MODELS, fit_variogram,
		 man_variogram)


data = pd.read_csv('example.csv')
x_col, y_col, value_col = 'X', 'Y', 'Bed'
x, y = data[x_col], data[y_col]
values = data[value_col].to_numpy(dtype=float)

# Normal transform values -- essential for sGs
nst = NormalScoreTransform(values)  
g   = nst.forward(values)

lags, gamma, counts = experimental_variogram(x, y, g, n_lags=51)
x_ = np.linspace(0, lags.max() * 1.1, 201)

# Plot experimental and model variograms
plt.plot(lags, gamma, '.k', label='data')
for model_name in MODELS.keys():
    model = MODELS[model_name]
    pars  = fit_variogram(lags, gamma, model_name)
    pred  = model(x_, *pars.to_list()[1:])
    plt.plot(x_, pred, '-', label=model_name)

plt.legend()
plt.xlabel('Lags/[m]', fontsize='large')
plt.ylabel('Variogram/[-]', fontsize='large')
plt.show()

# Save best choice of parameters to a json file
best_model = 'spherical'  # choice: ('spherical'|'exponential'|'gaussian_model')
vp = fit_variogram(lags, gamma, best_model)

with open('params.json', 'w') as f:
    json.dump(vp.to_dict(), f)
```

# Run sGs calculation

```bash
FILE=</path/to/data/>
python sgs.py $FILE -g 5 -j 8 -p <prefix> -c <column_containing_data> -n 2000
```
