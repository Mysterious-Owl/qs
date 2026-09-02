# Deep Learning Concepts & Architectures Cheat Sheet

Companion to [`ML_Algorithms_Cheatsheet.md`](ML_Algorithms_Cheatsheet.md) (classical ML) and [`Transformers_Cheatsheet.md`](Transformers_Cheatsheet.md) (transformers in depth). Interview-oriented: theory + the equation + the gotcha interviewers actually probe.

See also [`../Example_Company/11_DL_Architectures.md`](../Example_Company/11_DL_Architectures.md) for how these come up in the specific Example Company loop.

---

# Part 1 — Core Concepts

## 1. The Neuron & Forward Pass
- **What it does**: Computes a weighted sum of inputs plus a bias, then applies a non-linear activation
- **Equation**: `a = σ(Wx + b)`
- **Why non-linearity matters**: Without an activation function, stacking N linear layers collapses to a single linear transformation (`W₃W₂W₁x` is just some other matrix `W'x`) — the network could never learn anything a single linear layer couldn't. The non-linearity is what makes depth meaningful.
- **Gotcha**: "Why do we need activation functions?" is a near-guaranteed question — the answer is the collapse-to-linear argument above, not "because it works better."

## 2. Backpropagation
- **What it does**: Efficiently computes the gradient of the loss with respect to every parameter, by applying the chain rule backward through the computation graph
- **Equation**: For a layer, `∂L/∂W = ∂L/∂a · ∂a/∂z · ∂z/∂W` — each layer's gradient reuses the gradient already computed for the layer above it
- **Key theory**: The efficiency insight is *reuse* — a naive per-parameter numerical gradient would need one forward pass per parameter; backprop gets all gradients in one forward + one backward pass.
- **Key terms**: Computation graph, chain rule, automatic differentiation
- **Gotcha**: Backprop is only the *gradient computation*; the optimizer (SGD/Adam) is what actually uses that gradient to update weights — conflating the two is a common sloppy answer.

## 3. Gradient Descent & Variants
- **Equation**: `θ ← θ - η·∇L(θ)`, where `η` = learning rate

| Variant | Update frequency | Trade-off |
|---|---|---|
| Batch GD | Once per full dataset | Stable gradient, very slow, memory-heavy |
| Stochastic GD (SGD) | Every single sample | Noisy, fast, noise can help escape shallow local minima |
| Mini-batch GD | Every batch (e.g. 32-256) | **The practical default** — balances stability and speed, and uses GPU parallelism efficiently |

- **Gotcha**: "SGD" in modern practice almost always means mini-batch, not literally one-sample-at-a-time.

## 4. Optimizers
| Optimizer | Core idea | Equation sketch |
|---|---|---|
| SGD | Plain gradient step | `θ ← θ - η·g` |
| SGD + Momentum | Accumulate a velocity term to smooth updates and accelerate through consistent directions | `v ← βv + g`; `θ ← θ - ηv` |
| RMSProp | Per-parameter learning rate scaled by a running average of squared gradients | `θ ← θ - η·g/√(E[g²]+ε)` |
| **Adam** | Momentum + RMSProp combined, with bias correction | Tracks 1st moment `m` (mean) and 2nd moment `v` (uncentered variance), then `θ ← θ - η·m̂/(√v̂+ε)` |
| AdamW | Adam with *decoupled* weight decay (applied directly to weights, not folded into the gradient) | Standard for training transformers |

- **When to choose**: Adam/AdamW is the default for most deep learning; SGD+momentum still often generalizes slightly better for large-scale vision training with a well-tuned LR schedule.
- **Key terms**: Momentum, adaptive learning rate, bias correction, weight decay
- **Gotcha**: Know *why* AdamW exists — in original Adam, L2 regularization added to the loss gets scaled by the adaptive per-parameter learning rate, which weakens it unpredictably; AdamW decouples weight decay from the gradient adaptation so it behaves as intended.

## 5. Activation Functions
| Function | Equation | Range | Notes |
|---|---|---|---|
| Sigmoid | `1/(1+e⁻ˣ)` | (0,1) | Saturates at both ends → vanishing gradients; still used for binary output layers |
| Tanh | `(eˣ-e⁻ˣ)/(eˣ+e⁻ˣ)` | (-1,1) | Zero-centered (better than sigmoid for hidden layers) but still saturates |
| **ReLU** | `max(0,x)` | [0,∞) | Default for CNNs/MLPs — cheap, non-saturating for positive inputs, sparse activations |
| Leaky ReLU | `max(αx,x)`, α≈0.01 | (-∞,∞) | Fixes "dying ReLU" by allowing a small negative slope |
| GELU | `x·Φ(x)` (Φ = Gaussian CDF) | (-∞,∞) | Smooth; **standard in transformers** |
| Softmax | `eˣⁱ/Σeˣʲ` | (0,1), sums to 1 | Multi-class output layer — converts logits to a probability distribution |
| Swish/SiLU | `x·σ(x)` | (-∞,∞) | Smooth, self-gated; used in some modern architectures |

- **Gotcha**: "Dying ReLU" — a neuron whose pre-activation is always negative outputs 0 forever, so its gradient is 0 forever and it never recovers. Leaky ReLU/GELU mitigate this.

## 6. Loss Functions
| Task | Loss | Equation |
|---|---|---|
| Regression | MSE | `(1/n)Σ(y-ŷ)²` |
| Regression (outlier-robust) | MAE / Huber | `|y-ŷ|` / quadratic near zero, linear in the tails |
| Binary classification | Binary cross-entropy | `-[y·log(ŷ) + (1-y)·log(1-ŷ)]` |
| Multi-class | Categorical cross-entropy | `-Σyᵢlog(ŷᵢ)` |
| Imbalanced classification | Focal loss | `-α(1-ŷ)ᵞlog(ŷ)` — down-weights easy examples so training focuses on hard ones |
| Embedding/similarity learning | Contrastive / Triplet / InfoNCE | Pull matching pairs together, push non-matching apart in embedding space |

- **Gotcha**: Cross-entropy is preferred over MSE for classification because MSE's gradient vanishes when a sigmoid output saturates, whereas cross-entropy's gradient stays proportional to the error — so training doesn't stall on confidently-wrong predictions.

## 7. Vanishing & Exploding Gradients
- **Problem**: Gradients are products of many per-layer terms; if those terms are consistently <1 the product shrinks toward 0 (vanishing — early layers stop learning), if consistently >1 it blows up (exploding — unstable training/NaNs)
- **Fixes**: ReLU-family activations (no saturation for positive inputs), **residual/skip connections** (give gradients a direct path backward), careful weight initialization, normalization layers, gradient clipping (for exploding specifically), and LSTM/GRU gating (for RNNs)
- **Gotcha**: This is *the* motivating problem behind residual connections and LSTMs — being able to name the problem and connect it to the architectural fix is what's being tested.

## 8. Weight Initialization
| Scheme | Use with | Idea |
|---|---|---|
| Xavier/Glorot | tanh/sigmoid | Variance `2/(nᵢₙ+nₒᵤₜ)` — keeps activation variance stable across layers |
| He/Kaiming | ReLU family | Variance `2/nᵢₙ` — accounts for ReLU zeroing half the activations |

- **Gotcha**: Initializing all weights to zero means every neuron in a layer computes the same thing and receives the same gradient — they stay identical forever (symmetry never breaks). Random init exists to break symmetry.

## 9. Normalization Layers
| Type | Normalizes across | Used in |
|---|---|---|
| **Batch Norm** | The batch dimension, per feature | CNNs |
| **Layer Norm** | The feature dimension, per sample | **Transformers/RNNs** |
| Instance Norm | Per-sample, per-channel (spatial) | Style transfer |
| Group Norm | Groups of channels per sample | Small-batch vision |
| RMSNorm | Like LayerNorm but without mean-centering | Modern LLMs (cheaper) |

- **Equation (BatchNorm)**: `x̂ = (x-μ_B)/√(σ²_B+ε)`, then scale and shift by learnable `γ`, `β`
- **Why it helps**: Stabilizes/smooths the optimization landscape, allows higher learning rates, and has a mild regularizing effect (from batch-statistic noise)
- **Gotcha (very commonly asked)**: BatchNorm behaves *differently at train vs. inference* — at training it uses batch statistics, at inference it uses running averages accumulated during training. Forgetting to switch to eval mode is a classic production bug. It also degrades with very small batch sizes (noisy statistics), which is exactly why transformers use LayerNorm (independent of batch size) instead.

## 10. Regularization in Deep Learning
| Technique | How it works |
|---|---|
| **Dropout** | Randomly zeroes a fraction `p` of activations during training — prevents co-adaptation of neurons; acts like training an implicit ensemble of subnetworks |
| L2 / weight decay | Penalizes large weights |
| Early stopping | Halt when validation loss stops improving |
| Data augmentation | Synthetically expand training data (crops, flips, noise; paraphrasing for text) |
| Label smoothing | Replace hard 0/1 targets with e.g. 0.1/0.9 — reduces overconfidence |

- **Gotcha**: Dropout is **disabled at inference**. With inverted dropout (the standard implementation) activations are scaled up by `1/(1-p)` during training so no rescaling is needed at test time.

## 11. Learning Rate Scheduling
- **Why**: A high LR early explores fast; a lower LR later allows fine convergence
- **Common schedules**: Step decay, cosine annealing, **linear warmup + cosine decay** (standard for transformers), ReduceLROnPlateau, one-cycle
- **Warmup**: Starting at a tiny LR and ramping up over the first few thousand steps prevents early instability when adaptive-optimizer moment estimates are still unreliable — essentially mandatory for transformer training
- **Gotcha**: Learning rate is generally the single highest-impact hyperparameter in deep learning — if asked "what would you tune first," this is the answer.

## 12. Transfer Learning & Fine-Tuning
- **Idea**: Reuse a model pretrained on a large general dataset, adapt it to your smaller specific task
- **Strategies**: Feature extraction (freeze the backbone, train only a new head) → partial fine-tuning (unfreeze top layers) → full fine-tuning → **parameter-efficient fine-tuning (LoRA/adapters)**
- **When to choose which**: Small dataset + similar domain → freeze most; large dataset or distant domain → fine-tune more layers
- **Key terms**: Backbone, head, frozen layers, catastrophic forgetting, LoRA
- **Gotcha**: Fine-tuning with too high a learning rate destroys the pretrained representations you're trying to leverage — use a much lower LR than you would for training from scratch (often 10-100x lower), or use discriminative LRs (lower for early layers).

---

# Part 2 — Architectures

## 13. MLP / Feedforward Network
- **Structure**: Fully-connected layers with non-linear activations
- **Use**: Tabular data, or as a component inside larger architectures (the FFN block in a transformer is an MLP)
- **Limitation**: No inductive bias for structure — treats input features as an unordered set, so it can't exploit spatial locality (images) or sequence order (text) without learning it from scratch
- **Gotcha**: On plain tabular data, gradient-boosted trees usually beat MLPs — reach for deep nets when the data has structure worth exploiting.

## 14. CNN (Convolutional Neural Network)
- **What it does**: Applies learned filters that slide across spatial input, detecting local patterns regardless of position
- **Key equation**: Output size `= ⌊(W - K + 2P)/S⌋ + 1` (W=input size, K=kernel, P=padding, S=stride) — worth memorizing, it's a common quick question
- **Key components**: Conv layers (learned filters), pooling (spatial downsampling / translation invariance), fully-connected head
- **Key theory / inductive biases**: **Parameter sharing** (the same filter is applied everywhere → far fewer parameters than a fully-connected layer over pixels) and **translation equivariance** (a feature is detected regardless of where it appears). Receptive field grows with depth, so early layers see edges/textures and deeper layers see object parts.
- **Notable architectures**: LeNet → AlexNet → VGG (uniform 3×3 stacks) → **ResNet** (residual connections, enabled very deep nets) → Inception (multi-scale parallel filters) → EfficientNet (compound scaling) → MobileNet (depthwise-separable convs for edge devices)
- **When to choose**: Images, video, spectrograms, and any grid-structured data; also 1D convs for sequences when locality matters more than long-range dependency
- **Key terms**: Kernel/filter, stride, padding, pooling, receptive field, feature map, depthwise-separable convolution
- **Gotcha**: CNNs are translation-*equivariant* by construction (a shifted input gives a correspondingly shifted feature map); the pooling layers are what add approximate translation-*invariance*. They are **not** inherently rotation- or scale-invariant — that comes from data augmentation.

## 15. ResNet & Residual Connections
- **What it does**: Adds a skip connection so a block learns a *residual* `F(x)` on top of its input: output `= F(x) + x`
- **Key theory**: Makes the identity function trivially easy to represent (set `F(x)=0`), so adding depth can't easily make things worse; critically, the `+x` term gives gradients a direct, unattenuated path backward, which is what solves vanishing gradients in very deep networks
- **Why it matters**: This single idea is what made 50-1000+ layer networks trainable, and it's reused in essentially every transformer block
- **Gotcha**: "Why do residual connections help?" — the strongest answer names *both* the optimization/gradient-flow argument and the "easy to learn identity, so depth doesn't hurt" argument.

## 16. RNN (Recurrent Neural Network)
- **What it does**: Processes sequences step by step, maintaining a hidden state that carries information forward
- **Equation**: `hₜ = σ(W_hh·hₜ₋₁ + W_xh·xₜ + b)`
- **Limitations**: Vanishing/exploding gradients over long sequences (the same weight matrix is applied repeatedly); **inherently sequential**, so it can't be parallelized across timesteps during training
- **Gotcha**: That non-parallelizability is precisely the bottleneck transformers eliminated — being able to state this is the key to explaining *why* transformers won.

## 17. LSTM (Long Short-Term Memory)
- **What it does**: An RNN with a gated cell state designed to carry information over long distances
- **Gates**: **Forget** gate (what to discard from cell state), **Input** gate (what new information to write), **Output** gate (what to expose as the hidden state)
- **Equation sketch**: `fₜ, iₜ, oₜ = σ(W·[hₜ₋₁,xₜ])`; `Cₜ = fₜ⊙Cₜ₋₁ + iₜ⊙C̃ₜ`; `hₜ = oₜ⊙tanh(Cₜ)`
- **Key theory**: The cell state's update is *additive* (`fₜ⊙Cₜ₋₁ + ...`) rather than a repeated multiplication, which is what lets gradients flow across many timesteps without vanishing — structurally the same trick as a residual connection
- **When to choose**: Sequential data where a transformer is overkill (small datasets, streaming/online inference, strict latency/memory limits)
- **Key terms**: Cell state, gating, constant error carousel
- **Gotcha**: LSTM's advantage over vanilla RNN comes specifically from the additive cell-state path — "it has gates" is an incomplete answer without saying what the gates buy you.

## 18. GRU (Gated Recurrent Unit)
- **What it does**: A simplified LSTM with two gates (reset, update) and no separate cell state
- **Differs from LSTM**: Fewer parameters, faster to train, often comparable performance; LSTM sometimes edges it out on very long sequences
- **When to choose**: Same situations as LSTM, when you want something lighter/faster
- **Gotcha**: There's no universal winner between LSTM and GRU — the honest interview answer is "comparable, GRU is cheaper, benchmark both on your task."

## 19. Seq2Seq & Encoder-Decoder
- **What it does**: An encoder compresses an input sequence into a representation; a decoder generates an output sequence from it (translation, summarization)
- **Bottleneck problem**: Compressing an entire input sequence into one fixed-size vector loses information for long inputs
- **Fix**: **Attention** — let the decoder look back at *all* encoder hidden states, weighted by relevance, at each generation step
- **Gotcha**: Attention was invented to fix the seq2seq bottleneck *within* RNNs; the transformer's insight was that attention alone (dropping recurrence entirely) works better. Knowing this history makes the "why transformers" answer much stronger.

## 20. Autoencoder
- **What it does**: Learns to compress input to a low-dimensional latent code (encoder) and reconstruct it (decoder), trained on reconstruction loss
- **Equation**: minimize `‖x - decoder(encoder(x))‖²`
- **Variants**: Denoising AE (reconstruct clean input from corrupted), Sparse AE, **VAE** (learns a probabilistic latent distribution, enabling generation by sampling)
- **When to choose**: Non-linear dimensionality reduction, **anomaly detection** (high reconstruction error = anomalous — directly relevant to fraud/compliance use cases), pretraining, denoising
- **Key terms**: Latent space, bottleneck, reconstruction error, VAE/reparameterization trick
- **Gotcha**: A plain autoencoder's latent space isn't structured for generation — sampling a random latent gives garbage. VAEs add a KL-divergence term forcing the latent to approximate a known prior (usually a standard Gaussian), which is what makes sampling meaningful.

## 21. VAE (Variational Autoencoder)
- **What it does**: An autoencoder that learns a *distribution* over the latent space, enabling principled generation
- **Loss**: Reconstruction loss + `KL(q(z|x) ‖ p(z))` — the second term pulls the learned latent distribution toward a standard Gaussian prior
- **Reparameterization trick**: Sample `z = μ + σ⊙ε` where `ε~N(0,1)` — this moves the randomness outside the network so gradients can flow through `μ` and `σ`
- **Gotcha**: The reparameterization trick is the standard "did you actually understand VAEs" question — you can't backprop through a random sampling operation directly, hence the trick.

## 22. GAN (Generative Adversarial Network)
- **What it does**: A generator creates fake samples, a discriminator tries to distinguish real from fake — trained adversarially in a minimax game
- **Objective**: `min_G max_D E[log D(x)] + E[log(1-D(G(z)))]`
- **Failure modes**: **Mode collapse** (generator produces only a few outputs that reliably fool the discriminator), training instability, vanishing generator gradient when the discriminator gets too strong too fast
- **Fixes**: Wasserstein GAN (better-behaved loss with meaningful distance semantics), gradient penalty, spectral normalization
- **When to choose**: Image synthesis, data augmentation — though diffusion models have largely displaced GANs for high-quality generation
- **Gotcha**: GAN training is a two-player game, not standard loss minimization — there's no single loss curve that reliably indicates progress, which is what makes them notoriously hard to train.

## 23. Diffusion Models
- **What it does**: Learns to reverse a gradual noising process — trained to predict and remove noise, then generates by denoising from pure noise
- **Forward process**: Progressively add Gaussian noise over T steps until the image is pure noise; **reverse process**: a network (usually a U-Net or transformer) learns to denoise step by step
- **Differs from GANs**: Stable, likelihood-grounded training (no adversarial game) and better sample diversity/quality; the trade-off is much slower sampling (many denoising steps vs. a GAN's single forward pass)
- **When to choose**: State-of-the-art image/audio/video generation
- **Key terms**: Forward/reverse diffusion, denoising score matching, classifier-free guidance, DDPM/DDIM
- **Gotcha**: Diffusion's main practical downside is inference cost (multiple network passes per sample) — distillation and improved samplers (DDIM, consistency models) exist specifically to cut the step count.

## 24. Graph Neural Networks (GNN)
- **What it does**: Learns node/edge/graph representations by iteratively aggregating information from each node's neighbors ("message passing")
- **Equation sketch**: `h_v^(k+1) = UPDATE(h_v^(k), AGGREGATE({h_u^(k) : u ∈ N(v)}))`
- **Variants**: GCN (normalized-mean neighbor aggregation), GraphSAGE (sampled neighbors — scales to large graphs), GAT (attention-weighted neighbor aggregation)
- **When to choose**: Explicitly relational data — social networks, **fraud rings / shared-identity networks**, molecular structures, recommendation bipartite graphs. Directly relevant to the fraud-ring and duplicate-product graph problems in [`../Example_Company/04_ML_System_Design_Fraud_Detection.md`](../Example_Company/04_ML_System_Design_Fraud_Detection.md) and [`../Example_Company/06_Duplicate_Product_Detection.md`](../Example_Company/06_Duplicate_Product_Detection.md)
- **Key terms**: Message passing, neighborhood aggregation, over-smoothing
- **Gotcha**: **Over-smoothing** — stacking too many GNN layers makes all node representations converge to nearly the same value (each node ends up aggregating the whole graph), so most practical GNNs are shallow (2-4 layers).

## 25. Vision Transformer (ViT)
- **What it does**: Splits an image into fixed-size patches, embeds each patch as a token, and processes them with a standard transformer encoder
- **Differs from CNN**: A CNN has built-in locality/translation inductive biases; ViT has almost none and must learn spatial relationships from data — so ViTs need much more training data (or heavy augmentation/pretraining) to match CNNs, but scale better at very large data/compute
- **When to choose**: Large-scale vision with substantial pretraining available; CNNs remain strong for smaller datasets
- **Gotcha**: The data-efficiency trade-off is the key comparison point — "ViT is better" is wrong without the "given enough data" qualifier.

## 26. CLIP / Contrastive Multimodal Models
- **What it does**: Jointly trains an image encoder and text encoder so matching image-text pairs land close together in a shared embedding space
- **Loss**: Contrastive (InfoNCE) — maximize similarity for the N matching pairs in a batch, minimize it for the N²-N non-matching pairs
- **Why it matters**: Enables zero-shot classification (compare an image embedding against text-prompt embeddings for each candidate class) and powers image-similarity search
- **Direct relevance**: This shared embedding space is exactly the mechanism behind image-based duplicate detection and multimodal compliance/attribute extraction — see [`../Example_Company/06_Duplicate_Product_Detection.md`](../Example_Company/06_Duplicate_Product_Detection.md), [`../Example_Company/07_Content_Compliance_Moderation.md`](../Example_Company/07_Content_Compliance_Moderation.md), [`../Example_Company/08_Attribute_Extraction_Content_Generation.md`](../Example_Company/08_Attribute_Extraction_Content_Generation.md)
- **Key terms**: Contrastive learning, InfoNCE, zero-shot transfer, shared embedding space
- **Gotcha**: Large batch sizes matter a lot for contrastive learning — more in-batch negatives means a stronger training signal.

---

## Quick reference: "which architecture when"

| Data / problem | Architecture |
|---|---|
| Tabular | Gradient-boosted trees usually; MLP if you must go neural |
| Images (moderate data) | CNN (ResNet/EfficientNet) |
| Images (very large data + pretraining) | ViT |
| Sequences, long-range dependencies, lots of data | Transformer |
| Sequences, small data / tight latency / streaming | LSTM or GRU |
| Text + image together | CLIP-style contrastive dual encoder, or a multimodal transformer |
| Relational / network structure | GNN (GraphSAGE for scale, GAT for attention weighting) |
| Anomaly detection without labels | Autoencoder (reconstruction error) |
| Generation (images/audio) | Diffusion (quality) or GAN (fast sampling) |
| Non-linear dimensionality reduction | Autoencoder; UMAP for visualization |
