# run the following:
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/autoregressive/inference/wm_compression.py     --checkpoint_dir checkpoints     --ar_model_dir Cosmos-Predict1-4B     --input_type tokens --input_tokens_dir /workspace/data/val_v2.0/     --top_p 0.8     --temperature 1.0    --video_save_name autoregressive-4b1-tokens --only_eval
```

### Generated examples using 8x8x8 tokenizer provided in the v2.0 validation set. Using 1-4B autoregressive model.

> First 9 frames are the input frames, next 8 frames are the generated frames. Concatenated with the video is the ground truth video with 17 decoded frames.

> Note: the autoregressive model is trained using 8x16x16 tokenizer and not the 8x8x8 tokenizer which is why we have poor results. I'm working on the post-training right now.

<table>
  <tr>
    <td><img src="assets/tokenized/sample_1_image.gif" width="150" /></td>
    <td><img src="assets/tokenized/sample_11_image.gif" width="150" /></td>
    <td><img src="assets/tokenized/sample_13_image.gif" width="150" /></td>
    <td><img src="assets/tokenized/sample_24_image.gif" width="150" /></td>
  </tr>
  <tr>
    <td><img src="assets/tokenized/sample_40_image.gif" width="150" /></td>
    <td><img src="assets/tokenized/sample_72_image.gif" width="150" /></td>
    <td><img src="assets/tokenized/sample_162_image.gif" width="150" /></td>
    <td><img src="assets/tokenized/sample_248_image.gif" width="150" /></td>
  </tr>
  <tr>
    <td><img src="assets/tokenized/sample_257_image.gif" width="150" /></td>
    <td><img src="assets/tokenized/sample_375_image.gif" width="150" /></td>
    <td><img src="assets/tokenized/sample_443_image.gif" width="150" /></td>
    <td><img src="assets/tokenized/sample_445_image.gif" width="150" /></td>
  </tr>
  <tr>
    <td><img src="assets/tokenized/sample_446_image.gif" width="150" /></td>
    <td><img src="assets/tokenized/sample_455_image.gif" width="150" /></td>
    <td><img src="assets/tokenized/sample_48_image.gif" width="150" /></td>
    <td><img src="assets/tokenized/sample_9_image.gif" width="150" /></td>
  </tr>
</table>

### Generated examples using 8x16x16 tokenizer. The raw data is encoded in this process. Using 1-4B autoregressive model.

> First 9 frames are the input frames, next 8 frames are the generated frames.

<table>
  <tr>
    <td><img src="assets/raw/sample_17.gif" width="150" /></td>
    <td><img src="assets/raw/sample_2.gif" width="150" /></td>
    <td><img src="assets/raw/sample_20.gif" width="150" /></td>
    <td><img src="assets/raw/sample_23.gif" width="150" /></td>
  </tr>
  <tr>
    <td><img src="assets/raw/sample_27.gif" width="150" /></td>
    <td><img src="assets/raw/sample_33.gif" width="150" /></td>
    <td><img src="assets/raw/sample_36.gif" width="150" /></td>
    <td><img src="assets/raw/sample_9.gif" width="150" /></td>
  </tr>
</table>

---

### Video Tokenization in Cosmos

The Cosmos model uses a discrete video tokenizer with 8×8×8 compression:
- **Temporal compression**: 8× (every 8 frames → 1 latent frame)
- **Spatial compression**: 8× in height and 8× in width



Given this compression ratio, a sequence of 17 raw video frames results in a latent representation with dimensions:

$$T_{latent} \times H_{latent} \times W_{latent} = \lceil 17/8 \rceil \times H/8 \times W/8 = 3 \times 32 \times 32$$

This creates a total of 3×32×32 = 3,072 tokens per video clip. For 256×256 input frames, each latent frame is represented as a 32×32 grid of tokens.

### 2. Inference with Pre-trained Autoregressive Models

#### 2.1 Context and Generation Windows

Our inference operates in a fully autoregressive scenario with:
- **Prompt Frames**: First 9 frames used as conditioning context
- **Predicted Frames**: Next 8 frames generated autoregressively


#### 2.2 Token Organization Mathematically

For a 17-frame sequence tokenized to 3 latent frames:

$$\text{Tokens} = [\underbrace{t_1, t_2, ..., t_{2048}}_{\text{9 prompt frames (2 latent frames)}}, \underbrace{t_{2049}, ..., t_{3072}}_{\text{8 predicted frames (1 latent frame)}}]$$

In the latent grid layout (3×32×32):
- The first 2 frames in the temporal dimension (2×32×32 = 2048 tokens) serve as context
- The last frame (1×32×32 = 1024 tokens) is predicted autoregressively

This is because the groups are formed such that the first frame is downsampled in a temporally caausal manner to the first latent frame. The paper doesn't explicitly mention this, but my assumption is that this helps in retreiving more information from the first frame as context.

The tokenizer compresses the temporal dimension from $(1 + T)$ raw frames to $(1 + T')$ latent frames, where:

$$T' = \lceil T/8 \rceil$$

So, the 17-frame sequence is compressed to 3 latent frames

#### 2.1 Causal Temporal Design

The tokenizer uses a temporally causal design so that each stage processes only current and past frames, independent of future frames. So for the 8x8x8 tokenized dataset given, it was safe to extract the first 2048 tokens representing the first 9 frames as context and then autoregressively generate the next 1024 tokens representing the next 8 frames since the encoding for the first 9 frames is not polluted by future frames.

### Tokenization Process

The tokenization begins with a 2-level wavelet transform that processes inputs in a group-wise manner:

$$\{x_0, x_{1:4}, x_{5:8}, \ldots, x_{(T-3):T}\} \rightarrow \{g_0, g_1, g_2, \ldots, g_{T/4}\}$$

This transformation downsamples inputs by a factor of 4 along all dimensions (x, y, t) and then subsequent encoder stages process frames in a temporally causal manner:

$$\{g_0, g_{0:1}, g_{0:2}, \ldots\} \rightarrow \{\xi_0, \xi_1, \xi_2, \ldots\}$$


#### 2.3 Cross-Entropy Loss Calculation

The evaluation metric I use is cross-entropy loss averaged on all the 8 generated frames, calculated as follows:

```python
# Extract the relevant logits for frames being predicted
future_logits = generation_logits[0, :ground_truth_future_tokens.size(0), :]  # [seq_len, vocab_size]
        
# Calculate cross entropy between model predictions and ground truth
ce_loss = F.cross_entropy(future_logits, ground_truth_future_tokens).item()
```

I'm not sure if this is the temporally teacher forced loss that the challenge expects, but I calculated the loss as follows:
- The model receives all ground truth tokens from the prompt frames
- The model autoregressively predicts all tokens in the future frames
- I compare predictions against ground truth for those future tokens only

### Run:

Our current implementation uses the pre-trained 1-4B autoregressive model to evaluate performance on the 8×8×8 tokenized dataset:

```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/autoregressive/inference/wm_compression.py \
    --checkpoint_dir checkpoints \
    --ar_model_dir Cosmos-Predict1-4B \
    --input_type tokens --input_tokens_dir /workspace/data/val_v2.0/ \
    --top_p 0.8 \
    --temperature 1.0 \
    --video_save_name autoregressive-4b1-tokens --only_eval
```

However, as noted in the examples above, the COSMOS 1-4B model was originally trained using a Discrete Video Tokenizer 8×16×16-720p tokenizer, not the 8×8×8 tokenizer used in the validation set. This is why we have suboptimal results.

### Current WIP:

- [x] Part 1: Initial Evaluation
  - Run inference using pretrained 1-4B autoregressive model
  - Evaluate performance on 8x8x8 tokenized dataset
  - Analyze baseline results

- [ ] Part 2: Model Post-Training
  - Post-train model on train_v2.0 8x8x8 tokenized dataset
  - Use temporally teacher-forced loss objective
  - Evaluate updated model performance

- [ ] Part 3: Action Integration
  - Integrate Pi0 FAST tokenizer for encoding action sequences
  - Implement cross-attention between transformer model features and action embeddings from FAST encoder
  - Evaluate final model with action conditioning

