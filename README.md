# Insightface face detection and recognition model that just works out of the box.

This is a minimalistic inference-focused repack of [Insightface](https://github.com/deepinsight/insightface).

For examples please refer to `InsightfaceExample.ipynb`.

# What it does

Provides you whith a one-liner to set up SoTA face detection and recognition model for inference.

# Supported features

- extract face embedding for a single-face image. 
- use any model from [Insightface ModelZoo](https://github.com/deepinsight/insightface/wiki/Model-Zoo) 

# Fixes and improvements over orignal version

- Everything that is not required for inference is removed
- Model instantiation is a single-line call
- Original version crashed with unintelligible `segmentation fault` if output of the face detector was wrong or empty. This version throws exception or returns `None`.
- Jupyter Notebook example


# TODO:

Simplify and refactor original code parts for readability and simplicity.
