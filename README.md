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


