# Using Gibbs Sampling to do LDA

This repository contains code for:

* crawling data for LDA
* Gibbs sampling to do LDA

## Data Preparation

Dependencies:

* `urllib`
* `pyquery`

Usage (needs network connection):

```bash
python get_data.py
```

Then the program will produce a `data.txt`, with the titles of each paper as lines.

## Gibbs Sampling

Dependencies:

* `numpy`
* `matplotlib`
* `pickle`

Usage:

```bash
$ python gibbs.py -h
usage: gibbs.py [-h] [--data DATA] [--K K] [--step STEP]

Uses gibbs samping to solve LDA model.

optional arguments:
  -h, --help   show this help message and exit
  --data DATA  The position of data file.
  --K K        The number of topics.
  --step STEP  Max number of steps.
```

After solving, the program produces a picture of log-likelihood changes and prints out the top 10 words for each topic.
