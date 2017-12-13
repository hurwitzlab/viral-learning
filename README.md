# viral-learning
Deep learning demonstration code.

# Repository Setup
This repository contains Jupyter notebooks, which are not friendly to `git`. One
way to help `git` handle these files is to set up `jq` to automatically
strip out non-code elements from notebooks.

The following recipe comes from http://timstaley.co.uk/posts/making-git-and-jupyter-notebooks-play-nice/.

First install [jq](https://stedolan.github.io/jq/download/).

Next add the following to `~/.gitconfig`. (Note: it may be necessary to specify
the full path to `jq` for tools such as SourceTree.)

```
[core]
attributesfile = ~/.gitattributes_global

[filter "nbstrip_full"]
clean = "jq --indent 1 \
        '(.cells[] | select(has(\"outputs\")) | .outputs) = []  \
        | (.cells[] | select(has(\"execution_count\")) | .execution_count) = null  \
        | .metadata = {\"language_info\": {\"name\": \"python\", \"pygments_lexer\": \"ipython3\"}} \
        | .cells[].metadata = {} \
        '"
smudge = cat
required = true
```

Finally add the following to `~/.gitattributes_global`:

```
*.ipynb filter=nbstrip_full
```

# Demonstration data
The `first_try` notebook needs two data files:

  + 'virus.fasta' https://www.ncbi.nlm.nih.gov/nuccore/NC_031261.1?report=fasta
  + 'bacterium.fna.gz' ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/195/955/GCF_000195955.2_ASM19595v2/GCF_000195955.2_ASM19595v2_genomic.fna.gz

# Install
The `first_try` notebook requires Jupyter, Keras, and TensorFlow. These can be installed
in a virtual environment as follows:

```
$ python3.6 -m venv ~/venv/vl
$ ~/venv/vl/bin/activate
(vl) $ pip install jupyter keras tensorflow
```

# Run
Start the Jupyter notebook server and run the `first_try` notebook.

```
(vl) $ jupyter notebook
```
