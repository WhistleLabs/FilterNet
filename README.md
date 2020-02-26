
# FilterNet: A many-to-many deep learning architecture for time series classification

This repository contains code to reproduce the results and figures in the paper: 
*FilterNet: A many-to-many deep learning architecture for time series classification*.

## Setup
The easiest way to run this software is via the Anaconda Python distribution.

## Running tests
In the root dir of this repo:

```
pytest tests
```

## Reproducing Results

1. Run the scripts in the `scripts/` directory. These are very long-running scripts that 
   reproduce each experimental condition many times. You might want to set, e.g., `NUM_REPEATS=1` 
   if you don't need this level of reproducibility.
   
2. Run the notebooks to re-produce the figures. You might need to edit a few paths to specific
   models to match the filenames on your system, especially if you changed the 
   `NAME` or `NUM_REPEATS` parameters.
     
------
Copyright (C) 2020 Pet Insight  Project - All Rights Reserved
