# Wahl-O-Mat Analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/microraptor/wahlomat_analysis/blob/main/wahlomat_analysis.ipynb)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/microraptor/wahlomat_analysis/HEAD?filepath=wahlomat_analysis.ipynb)

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-green.svg)](http://makeapullrequest.com)

This script analyzes www.wahl-o-mat.de German political party data.

It generates a correlation heatmap and a principal component analysis map.
On both charts clusters are indicated using two seperate calculations.
This project is inspired by Reddit users /u/d_loose/ and /u/askLubich/.

Please be aware it is highly questionable to pinpoint the political stance of a party solely on the very limited dataset of 30+ yes/no/abstain questions. Besides political opinions having more nuance than a simple yes/no and the parties actual behaviour differing from their answers given, the results of this analysis are also heavily influenced on what topics the questions are about and how the very short questions are phrased and then understood by the parties. Additionally, the analysis algorithms here are somewhat arbitrarily chosen. Nonetheless, the interesting thing to see here is that with every Wahl-O-Mat dataset PC1 of the 3 principal component analysis always seems to reflect somewhat the classical left-right wing spectrum.

## Results

Click on the charts below to see them in full size.

If you want to see analysis results for Bundesland elections, just run the Notebook in Binder or Google Colab and change the ELECTION constant in the beginning of the script to the corresponding Wahl-O-Mat URL name.

Please be careful when interpreting changes between different analysis results of Wahl-O-Mat datasets. Different political positions might be explained solely by a change of topic focus of the questions in the dataset and not by changed opinions of the parties. Since the PCA is computed separately between the results below, the components (especially PC2) might express completely different political meanings and should not be directly compared between results. Also the analysis only takes into account relative differences between the parties and overall political shifts of all parties are not shown.

### Bundestagswahl 2021

<span>
    <img src="https://raw.githubusercontent.com/microraptor/wahlomat_analysis/main/results/bundestagswahl2021_corr_heatmap.svg" height="256" alt="Correlation Heatmap">
    &emsp;&emsp;
    <img src="https://raw.githubusercontent.com/microraptor/wahlomat_analysis/main/results/bundestagswahl2021_pca_map.svg" height="256" alt="Principal Component Analysis">
    &emsp;&emsp;
    <img src="https://raw.githubusercontent.com/microraptor/wahlomat_analysis/main/results/bundestagswahl2021_pca_influences.svg" height="256" alt="Principal Component Influences">
</span>

### Europawahl 2024

<span>
    <img src="https://raw.githubusercontent.com/microraptor/wahlomat_analysis/main/results/europawahl2024_corr_heatmap.svg" height="256" alt="Correlation Heatmap">
    &emsp;&emsp;
    <img src="https://raw.githubusercontent.com/microraptor/wahlomat_analysis/main/results/europawahl2024_pca_map.svg" height="256" alt="Principal Component Analysis">
    &emsp;&emsp;
    <img src="https://raw.githubusercontent.com/microraptor/wahlomat_analysis/main/results/europawahl2024_pca_influences.svg" height="256" alt="Principal Component Influences">
</span>

### Bundestagswahl 2025

<span>
    <img src="https://raw.githubusercontent.com/microraptor/wahlomat_analysis/main/results/bundestagswahl2025_corr_heatmap.svg" height="256" alt="Correlation Heatmap">
    &emsp;&emsp;
    <img src="https://raw.githubusercontent.com/microraptor/wahlomat_analysis/main/results/bundestagswahl2025_pca_map.svg" height="256" alt="Principal Component Analysis">
    &emsp;&emsp;
    <img src="https://raw.githubusercontent.com/microraptor/wahlomat_analysis/main/results/bundestagswahl2025_pca_influences.svg" height="256" alt="Principal Component Influences">
</span>

## Explanation

For an overview of how to interpret the analysis check out [askLubich's repo](https://github.com/askLubich/Wahl-O-Mat-EU-2019) or his [German explanation](https://www.reddit.com/r/de/comments/bqubdv/wahlomat_analyse_zur_euparlamentswahl_2019_oc/eo7zmaq/).You might be also interested in the German [PCA discussions here](https://www.reddit.com/r/de/comments/1ijw36w/politische_%C3%A4hnlichkeit_von_parteien_nach/).

The script creates three charts: a correlation heatmap, a principal component analysis (PCA) map and a bar chart showing the PCA-component influences of each question.

Because first the principal component (PC1) explains the most variance, the x-axis on the PCA map is the most important one to visualize political differences. Interestingly, PC1 seems to always correspond somewhat to the commonly used defintion of the left-right wing spectrum. Differences on the y-axis (PC2) can be interpreted as political differences, which cannot be expressed as left-right issues. Therefore, the PC1 bar of the bar chart corresponds to how strong of a left or right opinion it is, if that question is answered with yes.

The Pearson correlation coefficient is used to calculate the coefficient values of the heatmap. For the PCA map the [default Scikit-learn PCA method](https://scikit-learn.org/stable/modules/decomposition.html) with 3 principal components is used (the third component is not shown on the chart). On the axes of the PCA map it is indicated what ratio of variance is explained by that component (axis).

There are also [cluster](https://scikit-learn.org/stable/modules/clustering.html) groups indicated by colored dots on the PCA map, which is calculated independently of the PCA. Therefore, it is possible to have outliers of the clusters, which can't be explained by the principle components and look out of place on the 2D chart. There is also a hierarchical-cluster dendrogram shown on the side of the correlation heatmap. The clustering methods of the two charts (dendrogram and colored dots) are separately calculated. There are variables in the beginning of the script to set which clustering algorithms are used. For the results shown above I used Affinity Propagation as the clustering algorithm for the colored dots on the PCA map. It determines the amount of cluster groups by itself.

## Running the Script

You can open the notebook in your browser [with binder here](https://mybinder.org/v2/gh/microraptor/wahlomat_analysis/HEAD?filepath=wahlomat_analysis.py). Then you can click the run button repeatedly to run each cell. It takes a moment for the charts to pop up. Similarly, you can also run the whole thing on Google's cloud: [Google Colaboratory](https://colab.research.google.com/github/microraptor/wahlomat_analysis/blob/main/wahlomat_analysis.ipynb)

The .py file has cell markers (`# %%`) and can be run cell-by-cell in [various IDEs](https://jupytext.readthedocs.io/en/latest/formats.html#the-percent-format) just like a Jupyter Notebook. If you want an actual .ipynb file, you can download it from the [binder](https://mybinder.org/v2/gh/microraptor/wahlomat_analysis/HEAD?filepath=wahlomat_analysis.py) or convert the .py locally: `jupytext --to notebook wahlomat_analysis.py`

## Dependencies for Local Execution

Most calculations are done using [Scikit-Learn](https://scikit-learn.org/) and [Pandas](https://pandas.pydata.org/). For plotting [Seaborn](https://seaborn.pydata.org/) and [Matplotlib](https://matplotlib.org/) are used.

The script was initially written for Python 3.8+. It probably also works on earlier versions, although the type hints might have to be removed.

### Installing the Dependencies

Commands for a terminal within the project directory (on some systems it might be `pip3` instead):

```sh
pip install -r requirements.txt
```

Alternatively, you can create a new virtual environment inside the project directory and install the dependencies there instead:

On Windows:

```sh
py -m venv env
.\env\Scripts\activate
pip install -r requirements.txt
```

Other OS (on some systems it might be `python3` instead):

```sh
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

If you prefer to use [Anaconda](https://www.anaconda.com/products/individual#Downloads) instead of pip:

Installing on the base environment:

```sh
conda config --append channels conda-forge
conda env update --name base --file requirements.txt
```

Alternatively, installing in a new environment:

```sh
conda create --name wahlomat-analysis --file requirements.txt --channel defaults --channel conda-forge
conda activate wahlomat-analysis
```

If you want to remove the environment:

```sh
conda remove --name wahlomat-analysis --all
```
