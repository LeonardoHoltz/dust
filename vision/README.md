# DUST on images: MNIST

DUST's estimator is not specific to text: it jitters the output of any linear layer, scores the jitter
by the loss change, and multiplies the estimated output error by the layer's cached input. What it
does rely on is a loss at every token (each token's jitter is rewarded with that token's loss) and
causal attention (attention internals are credited by later tokens only).

[`model.py`](model.py) builds an image model with both properties from the GPT blocks of
[`../model.py`](../model.py): a 28×28 digit is cut into 49 patches of 4×4 pixels, read in raster
order under causal attention, and every position predicts the digit from the patches seen so far.
The prediction is the last position's. The patch projection takes the token embedding's place,
value embeddings are indexed by patch position, and the head has 10 classes. `dust.DUST` is used
unchanged apart from accepting a non-token input and a head narrower than 256 classes.

## Run

Inside the Docker image (see the main README):

```bash
python vision/mnist.py --method dust --lr 0.1 --momentum 0.8 --epochs 5 --output runs/mnist-dust
python vision/mnist.py --method backprop --lr 0.1 --momentum 0.9 --epochs 5 --output runs/mnist-bp
python vision/gradient_check.py   # cosine between DUST's estimate and the backprop gradient
python vision/test_reassemble.py  # self-check: exact output error -> autograd's weight gradient (CPU)
```

MNIST is downloaded to `data/mnist`. The last 5,000 training images select the checkpoint; the
10,000 test images are scored once.

## Results

4 blocks, width 128, 0.8M parameters, batch 128, 5 epochs (2,145 SGD steps), one RTX 3060.
DUST uses population 256 (`mnist_recipe`); learning rates were picked from short half-epoch sweeps.

| Method   | lr, momentum | Test accuracy | Test loss | Wall clock |
|----------|--------------|---------------|-----------|------------|
| Backprop | 0.1, 0.9     | 97.94%        | 0.071     | 31 s       |
| DUST     | 0.1, 0.8     | 90.85%        | 0.290     | 40 min     |

At initialization DUST's estimate has cosine 0.65–0.95 with the backprop gradient for the MLP and
output projections and 1.0 for the head, but only 0.2–0.5 for the attention keys and 0.1–0.2 for the
position value embeddings, whose estimates are also 2–6× too large. Its learning curve is
noise-limited: in the sweep, learning rates from 0.02 to 0.2 reached the same accuracy after half an
epoch.

## A CNN with one loss per image

[`cnn.py`](cnn.py) applies the same regimen to a plain CNN (conv 5×5 ×16, pool, conv 5×5 ×32, pool,
FC 128, FC 10; 215k parameters) whose only loss is the image's cross-entropy. There are no tokens, so
one scalar reward per image credits every channel at every pixel of the jittered layer. What carries
over: jitter one layer's output at a time, reward each image's jitter with its own loss drop, center
rewards across each chunk of draws, and turn error × cached input into the weight gradient (via
`unfold` for convolutions). Only the layers after the jittered one are rerun, so the head is cheap.

```bash
python vision/cnn.py --method dust --population 1024 --lr 0.0125 --epochs 5 --output runs/cnn-dust-p1024
python vision/cnn.py --method backprop --lr 0.1 --epochs 5 --output runs/cnn-bp
python vision/cnn.py --check --population 256   # cosine with the backprop gradient
```

5 epochs, batch 128, momentum 0.9:

| Method          | lr     | Test accuracy | Test loss | Wall clock |
|-----------------|--------|---------------|-----------|------------|
| Backprop        | 0.1    | 98.76%        | 0.046     | 4 s        |
| DUST, pop. 256  | 0.0125 | 98.09%        | 0.062     | 2 min      |
| DUST, pop. 1024 | 0.0125 | 98.61%        | 0.039     | 7 min      |

The per-image reward makes the conv estimates noisy. At initialization the cosine with the backprop
gradient is 0.26 / 0.24 (conv1 / conv2), 0.73 (FC 128) and 1.0 (head) at population 256, and
0.83 / 0.68 / 0.96 / 1.0 at 4,096. At the trained checkpoint the conv estimates are 6–12× the true
gradient's norm, so the updates are mostly zero-mean noise that momentum averages out. That noise is
why the learning rate must stay low: at lr 0.05, population 256 diverged within 100 steps and 1,024
between steps 600 and 800. Weight sharing is what keeps the conv layers trainable at all: each kernel's gradient
sums the error estimate over every pixel of every image.

`cnn.py --method es` is a conventional evolution strategy for comparison: each draw jitters all
weights at once and is scored by the batch's mean loss. It is implemented and gradient-checked but
was not trained.

## Activation maximization

[`visualize.py`](visualize.py) finds, for chosen neurons, the image that drives each one highest
(gradient ascent on the pixels of a frozen model, from a near-black start shared by all models), and
draws one row per model:

```bash
python vision/visualize.py --arch cnn --neurons logit:0-9 --output runs/viz/cnn-logits \
    --models backprop=runs/cnn-bp/best.pt dust256=runs/cnn-dust-p256/best.pt dust1024=runs/cnn-dust-p1024/best.pt
python vision/visualize.py --arch patch --neurons mlp3:0-7 --output runs/viz/patch-mlp3 \
    --models backprop=runs/mnist-bp/best.pt dust256=runs/mnist-dust/best.pt
```

## Limits

A bidirectional ViT trained only through a class token would leave every patch token without a loss
of its own, so their MLPs and output projections would receive no direct estimate, and the local
attention credit assumes causal attention. Such models need changes to the estimator, not just a
new model file. The CNN shows that a single loss is workable when the perturbed layers share weights
over many positions, at the price of more draws and a lower learning rate.
