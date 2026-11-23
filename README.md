# AI Handbook — Chapter 16

Python examples for Chapter 16 of the AI Handbook, demonstrating fundamental concepts behind attention mechanisms in modern AI models.

## Examples

| File | Topic |
|---|---|
| `16.1.py` | Computing attention scores between a query and keys |
| `16.1.py` | Computing attention scores between a query and keys |
| `16.2.py` | Applying scaled scores and stable softmax weights |
| `16.3.py` | Using attention weights to produce an output from value vectors |
| `16.4.py` | Creating a lower-triangular causal-attention mask |
| `16.5.py` | Combining outputs from multiple attention heads |
| `16.6.py` | Computing self-attention and contextual vectors |

## Requirements

- Python 3
- NumPy

Install NumPy when it is not already available:

```bash
python -m pip install numpy
```

## Usage

Run an example from the repository root:

```bash
python 16.1.py
python 16.2.py
python 16.3.py
python 16.4.py
python 16.5.py
python 16.6.py
```

Each script prints the resulting scores, weights, masks, or vectors to standard output.
