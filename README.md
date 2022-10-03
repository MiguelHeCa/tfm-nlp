# tfm-nlp


## Data

Download Enron data

```
wget https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz
```

## Project structure configuration

Execute `projec-structure.sh` to create or ensure that the project has the correct structure.

## Extraction

The `extract.sh` file downloads enron emails and extracts them into their respective environment.
If files are already existing, it does nothing. This is to prevent unnecessary downloading since the files are heavy.


## Transformation

### Ignoring directories

According to Klimt & Yang (2004), there are two folders that are not necessary:

* `discussion_threads`: Do not appear to be used directly by the users, but rather computer generated.
* `all_documents`: Contanied  large number of duplicate emails, which where already present the rest of the folders (TODO: Check if there is time).

## Preprocessing

Consult Notebook `Preprocessing.ipynb` in the notebooks folder.

## Doc2vec and TF-IDF transformations

Consult Notebooks `Doc2Vec.ipynb` and `TF-IDF.ipynb` in the notebooks folder.

## Distance Matrices Calculation

Consult scripts `distance_matrices*.py` for the obtention of euclidean, cosine and WMD distance matrices for both doc2vec and TF-IDF vectors.

## Clustering

Consult scripts `test_clustering*.py` for performing of methods KMeans, DBSCAN and HDBSCAN, and their respective scores:

* Silhouette Coefficients
* Calinski-Harabasz
* David-Bouldin
* Entropy

There is also the notebook `cuMLHDBSCAN.ipynb` where it can be seen a draft of the logic behind the scripts.

## Results

There are a couple of notebooks that are still in works:

* `Results.ipynb`
* `LabeledEmails.ipynb`

## Notes

The rest of the notebooks and scripts are left for legacy purposes. Will be removed for the final delivery.
