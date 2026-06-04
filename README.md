# Adaptive Oscillator OCR: Biologically-Inspired Pattern Recognition

A biologically inspired pattern recognition project using **Kuramoto oscillator networks** and adaptive synchronization for optical character recognition (OCR) of handwritten digits.

---

## 🧠 Overview

This project implements a **Kuramoto oscillator-based associative memory network** for handwritten digit recognition. Instead of traditional deep learning approaches, we leverage the biological principles of coupled phase oscillators to classify patterns.

### Key Features

- **Biologically-Inspired**: Uses coupled Kuramoto oscillators to model pattern recognition similar to neural synchronization
- **Phase-Based Encoding**: Represents pixel intensities as phase values in the Kuramoto model
- **Associative Memory**: Learns class-specific coupling matrices for each digit (0-9)
- **Robust Recognition**: Handles noisy inputs effectively through phase oscillation dynamics
- **Lightweight**: No deep neural networks—pure mathematical model based on coupled oscillators

---

## 📚 Theoretical Background

### Kuramoto Model

The Kuramoto model describes the dynamics of N coupled oscillators with phases θᵢ:

```
dθᵢ/dt = ωᵢ + Σⱼ Kᵢⱼ sin(θⱼ - θᵢ)
```

Where:
- **θᵢ**: Phase of oscillator i
- **ωᵢ**: Natural frequency of oscillator i
- **Kᵢⱼ**: Coupling strength between oscillators i and j
- **sin(θⱼ - θᵢ)**: Synchronization term driving oscillators toward phase alignment

### Adaptation to OCR

1. **Image to Phase Mapping**: Each pixel value is encoded as a phase in [0, π]
   - Pixel intensity 0 → Phase 0
   - Pixel intensity 0.5 → Phase π/2
   - Pixel intensity 1.0 → Phase π

2. **Class-Specific Coupling**: For each digit class, we compute correlation matrices:
   ```
   Cᵢⱼ = ⟨cos(φᵢ - φⱼ)⟩ over all training samples of that class
   ```

3. **Global Weight Matrix**: Combined from all digit classes:
   ```
   W = (K/M) × Σ(C_classes)
   ```

4. **Recognition via Synchronization**: 
   - Input digit → Initialize phases → Run Kuramoto dynamics → Compare final state with prototypes

---

## 📁 Project Structure

```
adaptive-oscillator-ocr/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── KuramotoModelTest.ipynb           # Basic Kuramoto model demonstration
├── OCR.ipynb                         # Main OCR implementation with full experiments
└── MoreThan2OscillatorAnalysis.ipynb # Extended analysis with multiple oscillators
```

### Notebook Descriptions

| Notebook | Purpose |
|----------|---------|
| **KuramotoModelTest.ipynb** | Simple 2-oscillator Kuramoto model example with phase synchronization visualization |
| **OCR.ipynb** | Complete implementation including data loading, model training, recognition, and evaluation |
| **MoreThan2OscillatorAnalysis.ipynb** | Extended analysis of oscillator networks with more than 2 oscillators |

---

## 🚀 Getting Started

### Requirements

```bash
numpy>=1.20.0
matplotlib>=3.3.0
scipy>=1.6.0
jupyter>=1.0.0
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/sabin147/adaptive-oscillator-ocr.git
cd adaptive-oscillator-ocr
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset (MNIST-like optdigits):
```bash
# The OCR.ipynb notebook expects optdigits.tra dataset
# Available from UCI ML Repository or included in scikit-learn
```

4. Run the notebooks:
```bash
jupyter notebook OCR.ipynb
```

---

## 📊 Model Architecture

### Step-by-Step Process

#### 1. **Data Normalization**
- Input: 8×8 pixel images (64 pixels per image)
- Normalize pixel values to [0, 1]
- Map to phase values Φ ∈ [0, π]

#### 2. **Training Phase**
```
For each digit class (0-9):
  - Compute correlation matrix C from training samples
  - C_ij = average of cos(φ_i - φ_j)
```

#### 3. **Global Coupling Matrix**
```
W = (K/M) × Σ(C_0 + C_1 + ... + C_9)
K = coupling strength (0.04)
M = number of oscillators (64)
```

#### 4. **Recognition Phase**
```
For input digit:
  1. Initialize: θ₀ = π × (normalized image)
  2. Evolve: Run Kuramoto dynamics for 50 steps
  3. Readout: Blend final state with original input
  4. Classify: Compare with all digit prototypes
  5. Output: Digit class with highest similarity
```

### Mathematical Model

**Kuramoto Update (single step)**:
```
dθᵢ = (dt) × [ωᵢ + Σⱼ Wᵢⱼ sin(θⱼ - θᵢ)]
θᵢ ← mod(θᵢ + dθᵢ, 2π)
```

**Classification Score**:
```
score_digit = average of cos(θ_readout - θ_prototype_digit)
predicted_digit = argmax(scores)
```

---

## 📈 Results

### Accuracy Metrics

| Scenario | Accuracy | Robustness |
|----------|----------|-----------|
| Clean Images | ~91% | Baseline |
| Noisy Images (σ=0.25) | ~87% | Good |
| High Noise (σ=1.0) | ~74% | Acceptable |
| Extreme Noise (σ=1.5) | ~42% | Limited |

### Performance Analysis

1. **Robustness vs. Noise**: Success rate decreases gracefully with increasing phase noise
2. **Computational Cost**: O(M²) complexity per integration step (M = 64 oscillators)
3. **Memory Capacity**: Can store and distinguish up to 40+ digit classes
4. **Error Patterns**: Most confusion occurs between visually similar digits (4↔9, 5↔8)

---

## 🔬 Experimental Features

### Task 1: Robustness Analysis
Tests model resilience to phase noise at different noise levels (σ ∈ [0, 1.5])

```python
success_rates = test_robustness(sigma_range=np.linspace(0, 1.5, 10))
```

### Task 2: Computational Scaling
Analyzes integration time as function of grid size (M = N²)

```python
test_computational_cost()  # Tests M ∈ [64, 256, 576, 1024]
```

### Task 3: Memory Capacity
Evaluates accuracy when storing multiple digit classes

```python
test_memory_capacity(max_c=40)  # Tests storage of C ∈ [2, 40] classes
```

---

## 💡 Key Parameters

| Parameter | Value | Description |
|-----------|-------|------------|
| M | 64 | Number of oscillators (image pixels) |
| K | 0.04 | Coupling strength constant |
| dt | 0.1 | Time step for integration |
| steps | 50 | Number of Kuramoto evolution steps |
| α | 0.5 | Blending factor for readout (0=input only, 1=final state only) |
| noise_level | 0.25 | Gaussian noise standard deviation for robustness testing |

---

## 🎯 How It Works: Example Walkthrough

### Recognizing a Handwritten Digit

```
Input: 8×8 image of digit "3"
  ↓
1. Normalize pixels to [0, 1]
  ↓
2. Encode as phases: θ₀ = π × normalized_image
  ↓
3. Run 50 Kuramoto evolution steps
   - Oscillators gradually synchronize based on coupling
   - Similar pixels tend to align their phases
  ↓
4. Apply readout: θ_final = 0.5 × θ_evolved + 0.5 × θ_initial
  ↓
5. Compute similarity with each digit prototype (0-9)
   score_3 = ⟨cos(θ_final - θ_prototype_3)⟩ = 0.87
   score_4 = ⟨cos(θ_final - θ_prototype_4)⟩ = 0.52
   ...
  ↓
6. Select class with highest score
   Output: "3" ✓
```

---

## 🤖 Biological Inspiration

This approach draws from neuroscience:

- **Phase Oscillators**: Model neural firing patterns and synchronization
- **Kuramoto Model**: Describes synchronization in biological systems (e.g., firefly flashing, neural networks)
- **Associative Memory**: Similar to Hopfield networks but uses phase dynamics instead of binary states
- **Adaptive Coupling**: Each digit class "teaches" the network about relevant phase relationships

---

## 📖 Key References

1. **Kuramoto, Y.** (1984). "Chemical Oscillations, Waves, and Turbulence" - Foundational work on coupled oscillators
2. **Strogatz, S. H.** (2000). "From Kuramoto to Crawford: exploring the onset of synchronization in populations of coupled oscillators" - Comprehensive review
3. **Hopfield, J. J.** (1982). "Neural networks and physical systems with emergent collective computational abilities" - Associative memory framework

---

## 🔧 Usage Examples

### Basic Recognition
```python
# Predict digit from normalized image
pred, theta_final, reconstructed, scores = predict_digit(
    x_input=normalized_image,
    W=coupling_matrix,
    prototype_phases=digit_prototypes,
    steps=50,
    dt=0.1,
    alpha=0.5
)
print(f"Predicted digit: {pred}")
```

### Batch Evaluation
```python
# Evaluate on 300 random samples
accuracy, predictions, true_labels = evaluate_accuracy(
    X_data=normalized_images,
    y_data=digit_labels,
    noisy=False,
    num_tests=300
)
print(f"Accuracy: {accuracy:.2%}")
```

### Adding Noise for Testing
```python
# Add Gaussian noise to image
noisy_image = add_noise(clean_image, noise_level=0.25)
pred = predict_digit(noisy_image, W, prototype_phases)
```

---

## ⚙️ Customization

### Modify Coupling Strength
```python
K = 0.06  # Increase for stronger synchronization
W = (K / M) * np.sum(C_classes, axis=0)
```

### Adjust Evolution Time
```python
steps = 100  # More steps = deeper processing but slower
dt = 0.05    # Smaller dt = more accurate but slower
```

### Change Readout Strategy
```python
alpha = 0.8  # Favor final Kuramoto state over input
# vs
alpha = 0.2  # Favor original input over evolution
```

---

## 📝 Notes & Observations

### Strengths
✅ No training required (pattern memorization only)  
✅ Biologically plausible synchronization mechanism  
✅ Robust to noise through oscillatory dynamics  
✅ Interpretable phase-based representations  
✅ Memory efficient for small datasets  

### Limitations
⚠️ Lower accuracy than deep neural networks (~91% vs 99%+)  
⚠️ Limited scalability (O(M²) complexity)  
⚠️ Struggles with high noise (>1.0σ)  
⚠️ Confusion between visually similar low-resolution digits  
⚠️ Requires careful hyperparameter tuning  

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Implement adaptive coupling matrix learning
- [ ] Extend to multi-digit sequence recognition
- [ ] GPU acceleration for larger networks
- [ ] Support for other datasets (CIFAR-10, Fashion-MNIST)
- [ ] Hybrid approaches combining Kuramoto with neural networks

---

## 📄 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Sabin147** - Biologically-inspired AI researcher

---

## 🔗 Resources

- **UCI ML Repository**: [Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/ml/datasets/optical+recognition+of+handwritten+digits)
- **Kuramoto Model**: [Wikipedia Entry](https://en.wikipedia.org/wiki/Kuramoto_model)
- **Phase Oscillators**: [Scholarpedia](http://www.scholarpedia.org/article/Kuramoto_model)

---

## ⭐ Citation

If you use this project in your research, please cite:

```bibtex
@repository{AdaptiveOscillatorOCR,
  author = {Sabin147},
  title = {Adaptive Oscillator OCR: Biologically-Inspired Pattern Recognition using Kuramoto Networks},
  url = {https://github.com/sabin147/adaptive-oscillator-ocr},
  year = {2026}
}
```

---

## 📧 Questions?

For questions or discussions, feel free to open an issue on GitHub!

**Happy oscillating!** 🌊📐