# wahlomat_analysis

This script analyses www.wahl-o-mat.de political party data.

It generates a correlation matrix and a principal component analysis map, where also clusters are marked.

The code is inspired by /u/d_loose/ and /u/askLubich/. See:
https://www.reddit.com/r/de/comments/liad93/ich_habe_aus_dem_aktuellen_wahlomat/gn31jpv/

For more information about the analysis check out askLubich's repo:
https://github.com/askLubich/Wahl-O-Mat-EU-2019

Pull request and other improvements are welcome.

## Running

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/microraptor/wahlomat_analysis/HEAD?filepath=wahlomat_analysis.ipynb)

You can open the notebook in your browser by clicking the badge above.

The .py and .ipynb files do the same thing and have almost identical code.

## Bundestagswahl 2021

### Correlation Matrix

![Correlation Matrix](bundestagswahl2021_c_matrix.svg)

### Principal Component Analysis

![Principal Component Analysis](bundestagswahl2021_pca_map.svg)

#### Principal Component Influences

![Principal Component Influences](bundestagswahl2021_pca_influences.svg)
