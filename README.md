# GPT From Scratch

A comprehensive implementation of the Transformer architecture from scratch, building understanding through first principles and detailed explanations.

## What's Included

- **Complete Transformer Implementation**: From positional encodings to multi-head attention
- **Educational Focus**: Detailed explanations of why each component exists and how it works
- **From-Scratch Approach**: Built using PyTorch primitives without relying on high-level transformer libraries
- **Architecture Coverage**: 
  - Positional Embeddings (sinusoidal)
  - Scaled Dot-Product Attention
  - Multi-Head Attention
  - Feed-Forward Networks
  - Encoder and Decoder Blocks
  - Causal Masking
  - Full Encoder-Decoder Transformer
  - Full Decoder-Only Transformer and Core Differences

## Implementation Details

### Core Components

1. **PositionalEncoding**: Fixed sinusoidal embeddings that provide position information
2. **ScaledDotProductAttention**: Core attention mechanism with proper scaling
3. **MultiHeadAttention**: Parallel attention heads with learned projections
4. **FeedForward**: Position-wise non-linear transformations
5. **EncoderBlock**: Self-attention + FFN with residual connections
6. **DecoderBlock**: Masked self-attention + cross-attention + FFN
7. **Transformer**: Complete encoder-decoder architecture

### Key Design Choices

- **Post-LayerNorm**: Following the original Transformer paper
- **Residual Connections**: Enable training of deep models
- **Causal Masking**: Ensures autoregressive behavior in decoder
- **Learned Projections**: Separate Q, K, V projections for flexibility

## File Structure

```
GPT_From_Scratch/
├── GPT_From_Scratch (2).ipynb    # Main implementation notebook
├── index.html                    # Static HTML export of notebook
└── README.md                     # This file
```

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch
- Jupyter Notebook

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd GPT_From_Scratch

# Install dependencies
pip install torch torchvision torchaudio
pip install jupyter notebook
```

### Running the Code

```bash
# Start Jupyter notebook
jupyter notebook

# Open and run "GPT_From_Scratch (2).ipynb"
```

## Educational Value

This implementation is designed for learning and understanding. Each component includes:

- **Detailed Comments**: Explaining the "why" behind design choices
- **Mathematical Context**: References to original papers and theory
- **Architectural Rationale**: Why certain components are necessary
- **Historical Context**: Evolution from RNNs to Transformers

## Contributing

This repository is primarily educational. Contributions that improve clarity, fix bugs, or enhance the educational value are welcome.

## License

This project is provided for educational purposes. Please refer to the original papers for proper attribution when using these concepts.
