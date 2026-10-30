# Variogram calculation

Calculate and view experimental and model variograms
```python
import pandas as pd
import json
from sgs import (NormalScoreTransform, experimental_variogram, spherical,
		 exponential, gaussian_model, MODELS, fit_variogram, _get_xy)


data = pd.read_csv('example.csv', comment='#')
x_col, y_col, value_col = 'X', 'Y', 'Bed'
x, y, x_label, y_label  = _get_xy(data, x_col, y_col)
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
```

Save best choice of parameters to a json file
```python
best_model = 'spherical'  # choice: ('spherical'|'exponential'|'gaussian_model')
vp = fit_variogram(lags, gamma, best_model)

with open('params.json', 'w') as f:
    json.dump(vp.to_dict(), f)
```

# Run sGs calculation

Here, we will use the ```variogram_fit``` option to use the fitted variogram parameters.  
```bash
FILE=example.csv  # </path/to/data/>
PARS=params.json  # </path/to/params/file/>
X="X"		  # "Horizontal" coordinate (e.g. longitude)
Y="Y"		  # "Vertical" coordinate (e.g. latitude)
C="Bed"		  # Column containing values to interpolate
N=200		  # Run N realisations
J=-1		  # -1: Use all available processors
python sgs.py $FILE -g 5 -j 8 -x $X -y $Y -c $C -n 2000 -f $PARS -j $J
```
