#!/usr/bin/env python
"""
Scrapes and analyzes www.wahl-o-mat.de German political party data.

It generates a correlation heatmap and a principal component analysis map.
On both charts clusters are indicated using two seperate calculations.

For more information visit https://github.com/microraptor/wahlomat_analysis

There is a section to configure the settings after the imports.
"""


# %% IPython config

# Comment out, if running as a script
# %config InlineBackend.figure_format = 'svg'


# %% Import packages

# Standard library
import re
from urllib import request as urlreq

# External libraries
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.patches import Rectangle
from sklearn import cluster
from sklearn.decomposition import PCA

# Only required if running Google Colab and downloading images
# from google.colab import files


# %% Settings

# For more information see README.md

# Set which election should be analysed
ELECTION: str = "bundestagswahl2025"  # Part of the URL: www.wahl-o-mat.de/ELECTION/...
ELECTION_NAME: str = "Bundestagswahl 2025"  # Only used in the titles of charts

# Invert x-axis (PC1) of the PCA map
# This can be used to set the axis according to the classical left–right politcal spectrum
INVERT_PC1_AXIS: bool = False

# Set which cluster method should be used for the PCA map (Default: AffinityPropagation)
CLUSTER_METHOD: str = "AffinityPropagation"  # See sklearn.cluster for options
N_CLUSTERS: int = 6  # Number of clusters; only relevant for some cluster methods

# Set which hierarchical-cluster method should be used for the dendrogram on the heatmap
# This also sorts the party list of the heatmap (Defaults: "average" and "euclidean")
H_CLUSTER_METHOD = "average"  # See scipy.cluster.hierarchy.linkage for options
H_CLUSTER_METRIC = "euclidean"  # See scipy.spatial.distance.pdist for options

# Emphasize party names, if they appear
# Currently all parties which got > 2% in the Europawahl 2024
# Use only lowercase (casefold)
EMPHASIZED_PARTIES = (
    "die linke",
    "linke",
    "die grünen",
    "grüne",
    "spd",
    "fdp",
    "cdu",
    "csu",
    "union",
    "cdu/csu",
    "cdu / csu",
    "afd",
    "bsw",
    "volt",
    "freie wähler",
)

# Set seaborn theme and config
sns.set_theme(
    context="paper",
    style="darkgrid",
    palette="deep",
    rc={"savefig.dpi": 300, "figure.dpi": 300},
)


# %% Scrape data

# Download the raw JS data
url_request: str = (
    f"https://www.wahl-o-mat.de/{ELECTION}/app/definitionen/module_definition.js"
)
with urlreq.urlopen(url_request) as response:  # noqa: S310 # nosec: B310
    raw_data_js: str = response.read().decode()

# Extract the data points with regex
titles: list = re.findall(
    r"^WOMT_aThesen\[\d+\]\[\d+\]\[0] = \'(.+?)\';$", raw_data_js, re.MULTILINE
)
questions: list = re.findall(
    r"^WOMT_aThesen\[\d+\]\[\d+\]\[1] = \'(.+?)\';$", raw_data_js, re.MULTILINE
)
party_names_full: list = re.findall(
    r"^WOMT_aParteien\[\d+\]\[\d+\]\[0] ?= ?\'(.+?)\';$", raw_data_js, re.MULTILINE
)
party_names: list = re.findall(
    r"^WOMT_aParteien\[\d+\]\[\d+\]\[1] ?= ?\'(.+?)\';$", raw_data_js, re.MULTILINE
)
raw_answers: list = re.findall(
    r"^WOMT_aThesenParteien\[(\d+)]\[(\d+)] ?= ?\'(.+?)\';$",
    raw_data_js,
    re.MULTILINE,
)


# %% Transform and clean datasets

# Create dataframes
question_df: pd.DataFrame = pd.DataFrame(
    zip(titles, questions), columns=["title", "question"]
)
party_df: pd.DataFrame = pd.DataFrame(
    zip(party_names_full, party_names), columns=["full_name", "party_name"]
)
answer_df: pd.DataFrame = pd.DataFrame(
    pd.DataFrame(raw_answers, columns=["question", "party_id", "answer"]).astype("int")
)

# Exclude party entries with seemingly invalid answers
bad_parties: pd.Index = party_df.loc[
    answer_df.groupby("party_id")["answer"].std() == 0
].index
for party_id in bad_parties:
    answer_df = answer_df[answer_df["party_id"] != party_id]

# Shorten all party names over 16 characters long
party_df["party_name"] = party_df["party_name"].apply(
    lambda name: name if not (isinstance(name, str) and len(name) > 16) else name[:13] + "..."
)

# Modify answer dataframe to have party names as rows and questions as columns
answer_df = answer_df.join(party_df, on="party_id")
answer_df = pd.pivot_table(
    answer_df, values="answer", index="question", columns="party_name"
)


# %% Process data

# Calculate correlations with the Pearson correlation coefficient
answer_corr: pd.DataFrame = answer_df.corr(method="pearson")

# Calculate PCA components
pca: PCA = PCA(n_components=3)
party_pca: pd.DataFrame = pd.DataFrame(
    pca.fit_transform(answer_df.T),
    columns=["pca_x", "pca_y", "pca_z"],
    index=answer_df.T.index,
)

# Calculate clusters
cluster_params: dict = {}
if CLUSTER_METHOD in [  # Methods that use number of clusters as parameter
    "KMeans",
    "MiniBatchKMeans",
    "SpectralClustering",
    "SpectralBiclustering",
    "SpectralCoclustering",
    "Birch",
    "FeatureAgglomeration",
]:
    cluster_params["n_clusters"] = N_CLUSTERS
if CLUSTER_METHOD in [  # Methods that have a random state parameter
    "AffinityPropagation",
    "KMeans",
    "MiniBatchKMeans",
    "SpectralClustering",
    "SpectralBiclustering",
    "SpectralCoclustering",
]:
    cluster_params["random_state"] = 0  # Make calculations deterministic
party_pca["cluster"] = getattr(cluster, CLUSTER_METHOD)(**cluster_params).fit_predict(
    answer_df.T
)

# Define data for influence explanations
pca_influences: pd.DataFrame = question_df.join(
    pd.DataFrame(pca.components_.T, columns=["pca_x", "pca_y", "pca_z"])
).join(answer_df.sum(axis="columns").rename("answers_sum"))
pca_xvr, pca_yvr, pca_zvr = pca.explained_variance_ratio_


# %% Draw correlation heatmap

# Create and customize plot
plt.clf()
diag_mask: np.ndarray = np.zeros_like(answer_corr, dtype=bool)
np.fill_diagonal(diag_mask, val=True)
c_matrix: sns.matrix.ClusterGrid = sns.clustermap(
    data=answer_corr,
    cmap="RdBu_r",  # "RdYlGn" is also good but not as colorblind-friendly
    center=0,
    cbar_pos=None,
    annot=answer_corr * 100,
    fmt=".0f",
    annot_kws={"fontsize": 8},
    mask=diag_mask,
    linewidths=0.8,
    figsize=(12, 12),
    method=H_CLUSTER_METHOD,
    metric=H_CLUSTER_METRIC,
)
c_matrix.fig.suptitle(f"Übereinstimmungen der Parteien\n{ELECTION_NAME}", y=0.90)
c_matrix.ax_col_dendrogram.remove()
c_matrix.ax_row_dendrogram.set(title="Cluster-Hierarchie")
c_matrix.ax_heatmap.set(
    title="Korrelationskoeffizienten in %", xlabel=None, ylabel=None
)
c_matrix.ax_heatmap.tick_params(axis="both", direction="out", bottom=True, right=True)

# Emphasize specific parties
labels_row: list = c_matrix.ax_heatmap.get_yticklabels()
labels_col: list = c_matrix.ax_heatmap.get_xticklabels()
for party_label in labels_row:
    if party_label.get_text().casefold() in EMPHASIZED_PARTIES:
        party_label.set_color("darkgreen")
        party_label.set_fontweight("bold")
        pos_x, pos_y = party_label.get_position()
        c_matrix.ax_heatmap.add_patch(
            Rectangle(
                xy=(pos_x - 1.0, pos_y - 0.5),
                width=len(labels_col),
                height=1,
                fill=False,
                edgecolor="black",
                lw=2,
                clip_on=False,
            )
        )
for party_label in labels_col:
    if party_label.get_text().casefold() in EMPHASIZED_PARTIES:
        party_label.set_color("darkgreen")
        party_label.set_fontweight("bold")
        pos_x, pos_y = party_label.get_position()
        c_matrix.ax_heatmap.add_patch(
            Rectangle(
                xy=(pos_x - 0.5, pos_y),
                width=1,
                height=len(labels_row),
                fill=False,
                edgecolor="black",
                lw=2,
                clip_on=False,
            )
        )
    # Rotate all x-axis labels and adjust positioning
    party_label.set_rotation(55)
    party_label.set_rotation_mode("anchor")
    party_label.set_ha("right")

# Save as a file or show plot
plt.savefig(f"{ELECTION}_corr_heatmap.svg", bbox_inches="tight")
# files.download(f"{ELECTION}_corr_heatmap.svg")  # Google Colab only
# plt.show()


# %% Draw PCA map

# Create and customize plot
plt.clf()
plt.figure(figsize=(10, 10))
plt.suptitle(f"Hauptkomponentenanalyse (PCA) der Parteien\n{ELECTION_NAME}", y=0.94)
pca_map: plt.Axes = sns.scatterplot(
    data=party_pca, x="pca_x", y="pca_y", hue="cluster", palette="bright", legend="full"
)
pca_map.set(
    xlabel=f"Komponente X (PC1)\n{pca_xvr:.1%} Varianzanteil",
    ylabel=f"Komponente Y (PC2)\n{pca_yvr:.1%} Varianzanteil",
    xticks=[0],
    yticks=[0],
    xticklabels=[],
    yticklabels=[],
)
pca_map.legend(
    title="Clusters",
    handles=pca_map.get_legend_handles_labels()[0],
    labels=[""] * party_pca["cluster"].nunique(),  # noqa: WPS435
    facecolor="white",
    markerscale=1.5,
    ncol=2,
    handletextpad=0,
    columnspacing=0.2,
    shadow=True,
    borderaxespad=1,
)

# Invert x-axis (PC1), if configured
if INVERT_PC1_AXIS:
    pca_map.invert_xaxis()

# Define grid
pca_map.xaxis.set_minor_locator(ticker.AutoLocator())
pca_map.yaxis.set_minor_locator(ticker.AutoLocator())
pca_map.grid(True, which="major", linewidth=1.2)
pca_map.grid(True, which="minor", linewidth=0.3)

# Add labels to the dots
map_label_x_offset: float = 0.05
if INVERT_PC1_AXIS:
    map_label_x_offset *= -1
for party_name in party_pca.index:
    # pylint: disable=invalid-name
    color: str = "black"
    fontweight: str = "normal"
    if party_name.casefold() in EMPHASIZED_PARTIES:
        color = "darkblue"
        fontweight = "bold"
    pca_map.text(
        x=party_pca.loc[party_name, "pca_x"] + map_label_x_offset,
        y=party_pca.loc[party_name, "pca_y"] + 0.05,
        s=party_name,
        color=color,
        fontweight=fontweight,
        fontsize="small",
    )

# Add rugplot (ticks along the x and y axes)
rugplt = sns.rugplot(
    ax=pca_map,
    data=party_pca,
    x="pca_x",
    y="pca_y",
    hue="cluster",
    palette="bright",
    legend=False,
    clip_on=False,
    height=-0.01,
)

# Save as a file or show plot
plt.savefig(f"{ELECTION}_pca_map.svg", bbox_inches="tight")
# files.download(f"{ELECTION}_pca_map.svg")  # Google Colab only
# plt.show()


# %% Draw PCA influence barplot

# Scale data and adjust dataframe for plotting function
infl_prep = pca_influences.copy()
infl_prep["answers_sum"] *= (
    infl_prep[["pca_x", "pca_y"]].abs().max().max()  # max() reduces only one dimension
    / infl_prep["answers_sum"].abs().max()
)
infl_prep = infl_prep.melt(
    id_vars=["title"],
    value_vars=["pca_x", "pca_y", "answers_sum"],
    var_name="component",
    value_name="influence",
)

# Invert all pca_x influence values, if the x-axis is inverted on PCA map
if INVERT_PC1_AXIS:
    infl_prep.loc[infl_prep["component"] == "pca_x", "influence"] *= -1

# Create and customize plot
plt.clf()
plt.figure(figsize=(5, 18))
plt.suptitle(f"Einfluss der Fragen\n{ELECTION_NAME}", y=0.97)
inf_barplot: plt.Axes = sns.barplot(
    data=infl_prep,
    x="influence",
    y="title",
    hue="component",
    orient="h",
)
inf_barplot.set(
    xlabel=None,
    ylabel=None,
    xticks=[0],
    xticklabels=[
        r"$\longleftarrow$ $-$ / Nein                   $+$ / Ja $\longrightarrow$"
    ],
)
inf_barplot.legend(
    title=None,
    handles=inf_barplot.get_legend_handles_labels()[0],
    labels=[
        "Komponente X (PC1)",
        "Komponente Y (PC2)",
        "Antworten aller Parteien kumuliert",
    ],
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    facecolor="white",
    shadow=True,
)
inf_barplot.tick_params(axis="x", labelbottom=False, labeltop=True, length=0)

# Set minor y-ticks to the same as major ticks, shift them and use them for the grid
inf_barplot.set_yticks([pos - 0.5 for pos in inf_barplot.get_yticks()], minor=True)
inf_barplot.grid(False, axis="x")
inf_barplot.grid(True, which="minor", axis="y", linewidth=1)

# Save as a file or show plot
plt.savefig(f"{ELECTION}_pca_influences.svg", bbox_inches="tight")
# files.download(f"{ELECTION}_pca_influences.svg")  # Google Colab only
# plt.show()

# %%
