# viral-learning
Deep learning demonstration code.

# Repository Setup
This repository contains Jupyter notebooks, which are not friendly to `git`. One
way to help `git` handle these files is to set up `jq` to automatically
strip out non-code elements from notebooks on commit.

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

# Install
The required packages can be installed in a virtual environment by the usual method:
```
$ python3.6 -m venv ~/venv/vl
$ ~/venv/vl/bin/activate
(vl) $ pip install -r keras tensorflow pandas scikit-learn
```

Installation on Ocelote is similar but uses the Anaconda distribution:
```
$ conda create -n vl keras tensorflow pandas scikit-learn
```

An installation for GPU nodes on Ocelote can be created like this:
```
$ qsub -I -N gpu-vl -m bea -M <your-email@address> -W group_list=bhurwitz -q standard -l select=1:ncpus=28:ngpus=1:mem=168gb -l cput=01:00:00 -l walltime=01:00:00
$ conda create -n vlgpu pandas scikit-learn
$ source activate vlgpu
$ conda install -c anaconda tensorflow-gpu
$ conda install keras
```

# Run
Start the Jupyter notebook server and run the `kmer_nn_np_generator.ipynb` notebook.

```
(vl) $ jupyter notebook
```

