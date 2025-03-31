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

## Explanation:

## Additional Challenge Details

1. We've provided `magvit2.ckpt` in the dataset download, which are the weights for a [MAGVIT2](https://github.com/TencentARC/Open-MAGVIT2) encoder/decoder. The encoder allows you to tokenize external data to try to improve the metric.
2. The loss metric is nonstandard compared to LLMs due to the vocabulary size of the image tokens, which was changed as of v1.0 release (Jul 8, 2024). Instead of computing cross entropy loss on logits with 2^18 classes, we compute cross entropy losses on 2x 2^9 class predictions and sum them up. The rationale for this is that the large vocabulary size (2^18) makes it very memory-intensive to store a logit tensor of size `(B, 2^18, T, 16, 16)`. Therefore, the compression challenge considers families of models with a factorized pmfs of the form p(x1, x2) = p(x1)p(x2). For sampling and evaluation challenge, a factorized pmf is a necessary criteria.
3. For the compression challenge, we are making the deliberate choice to evaluate held-out data on the same factorized distribution p(x1, x2) = p(x1)p(x2) that we train on. Although unfactorized models of the form p(x1, x2) = f(x1, x2) ought to achieve lower cross entropy on test data by exploiting the off-block-diagonal terms of Cov(x1, x2), we want to encourage solutions that achieve lower losses while holding the factorization fixed.
4. For the compression challenge, submissions may only use the *prior* actions to the current prompt frame. Submissions can predict subsequent actions autoregressively to improve performance, but these actions will not be provided with the prompt.
5. Naive nearest-neighbor retrieval + seeking ahead to next frames from the training set will achieve reasonably good losses and sampling results on the dev-validation set, because there are similar sequences in the training set. However, we explicitly forbid these kinds of solutions (and the private test set penalizes these kinds of solutions).
6. We will not be able to award prizes to individuals in U.S. sanctioned countries. We reserve the right to not award a prize if it violates the spirit of the challenge.


### Metric Details
There are different scenarios for evaluation, which vary in the degree of ground truth context the model receives.
In decreasing order of context, these scenarios are:
- **Fully Autoregressive**: the model receives a predetermined number of ground truth frames and autoregressively predicts all remaining frames.
- **Temporally Teacher-forced**: the model receives all ground truth frames before the current frame and autoregressively predicts all tokens in the current frame.
- **Fully Teacher-forced**: the model receives all ground truth tokens before the current token, 
including tokens in the current frame. Only applicable for causal LMs.

As an example, consider predicting the final token of a video, corresponding to the lower right patch of frame 15. 
The context the model receives in each scenario is:
- Fully Autoregressive: the first $t$x16x16 tokens are ground truth tokens corresponding to the first $t$ prompt frames, 
and all remaining tokens are autoregressively generated, where $0 < t < 15$ is the predetermined number of prompt frames.
- Temporally Teacher-forced: the first 15x16x16 tokens are ground truth tokens corresponding to the first 15 frames, 
and all remaining tokens are autoregressively generated.
- Fully Teacher-forced: all previous (16x16x16 - 1) tokens are ground truth tokens.

The compression challenge uses the "temporally teacher-forced" scenario.
## Leaderboard

These are evaluation results on `data/val_v1.1`.
<table>
  <thead>
    <tr>
      <th>User</th>
      <th>Temporally Teacher-forced<br>CE Loss</th>
      <th>Temporally Teacher-forced<br>Token Accuracy</th>
      <th>Temporally Teacher-forced<br>LPIPS</th>
      <th>Generation Time* <br>(secs/frame)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://huggingface.co/1x-technologies/GENIE_138M">1x-technologies/GENIE_138M</a><br>(<code>--maskgit_steps 2</code>)</td>
      <td>8.79</td>
      <td>0.0320</td>
      <td>0.207</td>
      <td>0.075</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/1x-technologies/GENIE_35M">1x-technologies/GENIE_35M</a><br>(<code>--maskgit_steps 2</code>)</td>
      <td>8.99</td>
      <td>0.0301</td>
      <td>0.217</td>
      <td>0.030</td>
    </tr>
  </tbody>
</table>
*Generation time is the time to generate latents on a RTX 4090 GPU, and excludes the time to decode latents to images.


## Help us Improve the Challenge!

Beyond the World Model Challenge, we also want to make the challenges and datasets more useful for *your* research questions. Want more data interacting with humans? More safety-critical tasks like carrying cups of hot coffee without spilling? More dextrous tool use? Robots working with other robots? Robots dressing themselves in the mirror? Think of 1X as the operations team for getting you high quality humanoid data in extremely diverse scenarios.

Email challenge@1x.tech with your requests (and why you think the data is important) and we will try to include it in a future data release. You can also discuss your data questions with the community on [Discord](https://discord.gg/UMnzbTkw). 

We also welcome donors to help us increase the bounty.

## Run the following:
```
$ CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/autoregressive/inference/wm_compression.py     --checkpoint_dir checkpoints     --ar_model_dir Cosmos-Predict1-4B     --input_type tokens --input_tokens_dir /workspace/data/val_v2.0/     --top_p 0.8     --temperature 1.0    --video_save_name
```

## Citation

If you use this software or dataset in your work, please cite it using the "Cite this repository" button on Github.

## Changelog

- v1.1 - Release compression challenge criteria; removed pauses and discontinuous videos from dataset; higher image crop.
- v1.0 - More efficient MAGVIT2 tokenizer with 16x16 (C=2^18) mapping to 256x256 images, providing raw action data.
- v0.0.1 - Initial challenge release with 20x20 (C=1000) image tokenizer mapping to 160x160 images.


## Dataset Metadata
The following table is necessary for this dataset to be indexed by search
engines such as <a href="https://g.co/datasetsearch">Google Dataset Search</a>.
<div itemscope itemtype="http://schema.org/Dataset">
<table>
  <tr>
    <th>property</th>
    <th>value</th>
  </tr>
  <tr>
    <td>name</td>
    <td><code itemprop="name">1X World Model Challenge</code></td>
  </tr>
  <tr>
    <td>url</td>
    <td><code itemprop="url">https://github.com/1x-technologies/1xgpt</code></td>
  </tr>
  <tr>
    <td>description</td>
    <td><code itemprop="description">A dataset of over 100 hours of compressed image tokens + raw actions across a fleet of EVE robots.</code></td>
  </tr>
  <tr>
    <td>provider</td>
    <td>
      <div itemscope itemtype="http://schema.org/Organization" itemprop="provider">
        <table>
          <tr>
            <th>property</th>
            <th>value</th>
          </tr>
          <tr>
            <td>name</td>
            <td><code itemprop="name">1X Technologies</code></td>
          </tr>
        </table>
      </div>
    </td>
  </tr>
  <tr>
    <td>license</td>
    <td>
      <div itemscope itemtype="http://schema.org/CreativeWork" itemprop="license">
        <table>
          <tr>
            <th>property</th>
            <th>value</th>
          </tr>
          <tr>
            <td>name</td>
            <td><code itemprop="name">Apache 2.0</code></td>
          </tr>
        </table>
      </div>
    </td>
  </tr>
</table>
</div>
