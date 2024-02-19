# AI-Generated-Text-Detection
A machine learning model that detects whether an essay was written by a student or an LLM


![picture](images/img.png)

The notebook `notebooks/training.ipynb`, showcase KerasCore and KerasNLP's multi-backend capabilities for detecting fake text.
Thanks to KerasCore (soon to be Keras 3.0), the notebook seamlessly executes on TensorFlow, Jax, and PyTorch platforms with minimal adjustments. 
It also supports single/multi GPU and TPU training. 
As datasets grow, TPUs become invaluable for training larger models.