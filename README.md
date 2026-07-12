# Perceptron from Scratch

**A from-scratch study of binary and multiclass handwritten-digit classification.**

This repository revisits my December 2022 L3 Physics project at Université Paris Cité. The goal was not to call a machine-learning classifier, but to implement the learning rules with NumPy, inspect their behavior on MNIST, and understand where a linear perceptron stops being enough.

> Educational implementation: `scikit-learn` is used only to download MNIST. Both classifiers, their gradients, and the evaluation logic are implemented in this repository.

![Examples from MNIST](assets/results/mnist-samples.png)

## What I investigated

The project follows two steps:

1. **Binary classification (0 vs 1).** A Rosenblatt perceptron learns one linear decision boundary and updates its weights only after a mistake.
2. **Multiclass classification (0–9).** A one-hidden-layer network uses ReLU activations, a softmax output, and gradient descent implemented directly with NumPy.

The original study also tested learning-rate sensitivity, visualized learned pixel weights, followed confusion matrices during training, and reduced MNIST images from 28×28 to 14×14 with nearest-neighbor sampling.

## Results from the 2022 study

The binary model classified more than 99% of the selected 0/1 samples correctly. Its train and test curves remained close, suggesting good generalization for this linearly separable task.

| Binary perceptron | Multiclass network |
|---|---|
| ![Binary train and test accuracy](assets/binary-train-test-accuracy.png) | ![One layered Multy Class train and test accuracy](assets/plot_accuracy_lrate.png) |


The one-hidden-layer multiclass model plateaued around **70% accuracy**. This is not presented as a competitive MNIST result: it documents the limits of the deliberately small architecture and original full-batch training method.

| Binary perceptron | Multiclass network |
|---|---|
| ![Binary confusion matrix](assets/binary-confusion-matrix.png) | ![Multiclass confusion matrix](assets/multiclass-confusion-matrix.png) |


Visualisation of the weights matrices can also be found and were discussed.
| Binary perceptron | Multiclass network |
|---|---|
| ![Binary Weight Matrix](assets/weights_matrix.png) | ![One layered Multy Class Weight Matrices](assets/weights_matrices_03.png) |


The report contains the complete methodology and discussion:

- [Project report (French PDF)](docs/project-report.pdf)
- [Presentation (French PDF)](docs/presentation.pdf)

## Reproduce the study

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install"
jupyter lab
```

Open [`perceptron_mnist.ipynb`](perceptron_mnist.ipynb). It runs an offline synthetic smoke study by default. Set `RUN_FULL_MNIST = True` in its configuration cell to download MNIST into ignored `data/`, train both models, and produce fresh results. Fixed seeds make each mode repeatable.

## Repository map

```text
notebook                     Narrative experiment and main entry point
sources/                     Tested data, model, metric, and plot functions
assets/                      Selected figures from the 2022 project
docs/                        Original report and presentation
archive/original-sources/    Three selected, unmodified 2022 scripts
```

## What changed in this reconstruction

The original scripts mixed data loading, model code, plotting, and execution in global state. This reconstruction separates those concerns, uses sample-major array shapes consistently, stabilizes softmax numerically, makes randomness explicit. The underlying models remain faithful to the project rather than being tuned into modern high-accuracy MNIST systems.

The debugging journey mattered as much as the final curves: I found shape errors that NumPy accepted silently, a softmax normalization over the wrong axis, exploding exponentials, and dataset-format mismatches. Those failures motivated the explicit contracts and tests now used throughout the package.

## Authorship

Project, report, presentation, implementations, and reconstruction by **Eloi Raad**.
