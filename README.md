# GPT From Scratch

A comprehensive implementation of the Transformer architecture from scratch, building understanding through first principles and detailed explanations.

## Overview

This repository contains a complete implementation of an Encoder-Decoder Transformer model, as introduced in the seminal paper "Attention Is All You Need" (Vaswani et al., 2017). The implementation is designed to be educational, with extensive documentation explaining the reasoning behind each architectural component.

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

## Key Concepts Explained

### Why Encoder-Decoder?
The original Transformer architecture was designed for sequence-to-sequence tasks like translation. The encoder processes the entire input sequence to build contextual representations, while the decoder generates output sequences autoregressively while attending to both its past outputs and the encoder's representations.

### Why Multi-Head Attention?
Multi-head attention allows the model to attend to different types of relationships simultaneously. Each head can specialize in different aspects (syntactic, semantic, long-range dependencies), providing richer representations than single-head attention.

### Why Positional Encodings?
Attention mechanisms are permutation-invariant - they have no inherent notion of sequence order. Positional encodings inject explicit order information, allowing the model to distinguish between sequences with the same tokens but different orderings.

### Why Scaled Dot-Product Attention?
The scaling factor (1/√d_k) prevents dot products from growing too large, which would cause softmax to become overly peaked and lead to vanishing gradients. This ensures stable training and balanced attention distributions.

## Architecture

```
Input Sequence
    ↓
Token Embeddings + Positional Encodings
    ↓
┌─────────────────┐    ┌─────────────────┐
│   Encoder Stack │    │  Decoder Stack  │
│                 │    │                 │
│  ┌─────────────┐│    │ ┌─────────────┐│
│  │ Encoder     ││    │ │ Decoder     ││
│  │ Block       ││◀───┤ │ Block       ││
│  │             ││    │ │             ││
│  │ - Self-Attn ││    │ │ - Self-Attn ││
│  │ - FFN       ││    │ │ - Cross-Attn││
│  │ - Norm      ││    │ │ - FFN       ││
│  └─────────────┘│    │ │ - Norm      ││
│       ...        │    │ └─────────────┘│
└─────────────────┘    │       ...        │
    ↓                    └─────────────────┘
Contextual                    ↓
Representations         Output Projections
                            ↓
                        Output Sequence
```

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

## Key Insights

### Why Transformers Won Out

While encoder-decoder models were strong around 2018-2020, decoder-only models eventually dominated because:

- **Better Scaling**: Simpler architecture with fewer attention paths
- **Training-Inference Alignment**: Perfect consistency between training and generation
- **Prompt-Based Learning**: Can absorb conditional tasks via prompting
- **General-Purpose Design**: More flexible for diverse applications
- **Efficient Caching**: KV caching during generation

### The Power of Attention

The attention mechanism enables:
- **Long-Range Dependencies**: Direct connections between any two tokens
- **Parallel Processing**: No sequential dependencies like RNNs
- **Interpretability**: Attention weights provide insights into reasoning
- **Flexibility**: Same mechanism works for different tasks

## References

1. **Vaswani et al. (2017)** - "Attention Is All You Need"
2. **Ba et al. (2016)** - "Layer Normalization"
3. **He et al. (2016)** - "Deep Residual Learning for Image Recognition"
4. **Bahdanau et al. (2015)** - "Neural Machine Translation by Jointly Learning to Align and Translate"

## Contributing

This repository is primarily educational. Contributions that improve clarity, fix bugs, or enhance the educational value are welcome.

## License

This project is provided for educational purposes. Please refer to the original papers for proper attribution when using these concepts.

---

**Note**: This implementation focuses on understanding the core Transformer architecture. For production use, consider optimized libraries like `transformers` (Hugging Face) which provide pre-trained models and efficient implementations.