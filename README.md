# Wahl-O-Mat Analysis

This script analyses www.wahl-o-mat.de German political party data.

It generates a correlation matrix and a principal component analysis map, where also clusters are marked.
This project is inspired by Reddit users /u/d_loose/ and /u/askLubich/.
Pull request and other improvements are welcome.

## Bundestagswahl 2021

<span>
    <img src="https://raw.githubusercontent.com/microraptor/wahlomat_analysis/main/bundestagswahl2021_c_matrix.svg" height="256" alt="Correlation Matrix">
    &emsp;&emsp;
    <img src="https://raw.githubusercontent.com/microraptor/wahlomat_analysis/main/bundestagswahl2021_pca_map.svg" height="256" alt="Principal Component Analysis">
    &emsp;&emsp;
    <img src="https://raw.githubusercontent.com/microraptor/wahlomat_analysis/main/bundestagswahl2021_pca_influences.svg" height="256" alt="Principal Component Influences">
</span>

Click on the plots to see them in full size.

## Explanation

For an overview of how to interpret the analysis check out [askLubich's repo](https://github.com/askLubich/Wahl-O-Mat-EU-2019) or his [German explanation](https://www.reddit.com/r/de/comments/bqubdv/wahlomat_analyse_zur_euparlamentswahl_2019_oc/eo7zmaq/).

Interestingly, the first principal component (PC1), which is plotted on the X-axis, usually corresponds mostly to the common left and right wing classification. PC2 on the Y-axis can often be partly interpreted as how authoritarian a party is.

## Running the Script

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/microraptor/wahlomat_analysis/HEAD?filepath=wahlomat_analysis.ipynb)

You can open the notebook in your browser by clicking [the badge above](https://mybinder.org/v2/gh/microraptor/wahlomat_analysis/HEAD?filepath=wahlomat_analysis.ipynb). Then you can click the run button repeatedly to run each cell. It takes a moment for the plots to pop up. Similarly, you can also run the whole thing on Google's cloud: [Google Colaboratory](https://colab.research.google.com/)

The .py and .ipynb files in this repo do the same thing and have almost identical code.

## Dependencies for Local Execution

See [requirement.txt](requirement.txt) or [environment.yml](environment.yml).

### Installing the Dependencies

This script requires to have Python 3.8+.

```sh
pip install -r requirements.txt
# On Ubuntu:
pip3 install -r requirements.txt
```

Alternatively, with [Anaconda](https://www.anaconda.com/products/individual#Downloads)
(Replace `base` with your environment of choice):

```sh
conda env update -n base --file environment.yml
```
