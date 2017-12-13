# viral-learning
Deep learning demonstration code.

# Repository Setup
This repository contains Jupyter notebooks, which are not friendly to `git`. The
best way to help `git` handle these files is to set up `jq` to automatically
strip out non-code elements from notebooks.

First install [jq](https://stedolan.github.io/jq/download/).

Next add the following to `~/.gitconfig`

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

Finally add the following to `./gitattributes_global`:

```
*.ipynb filter=nbstrip_full
```
