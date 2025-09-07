# Energy Efficiency in Neural Network Training: An Optimizer Comparison Study

This repository contains research investigating the relationship between optimizer choice and energy consumption in neural network training. The study examines performance and environmental impact across multiple datasets and optimization algorithms.

## Overview

Modern machine learning training consumes significant computational resources, raising concerns about environmental impact. This research systematically evaluates how different optimization algorithms affect both model performance and energy consumption through controlled experiments.

**Key Question**: How do different optimizers balance training performance with energy efficiency?

## Paper

**Title**: An Analysis of Optimizer Choice on Energy Efficiency and Performance in Neural Network Training

**Author**: Tom Almog, University of Waterloo

**Abstract**: As machine learning models grow increasingly complex and computationally demanding, understanding the environmental impact of training decisions becomes critical for sustainable AI development. This paper presents an empirical study investigating the relationship between optimizer choice and energy efficiency in neural network training through 360 controlled experiments across three benchmark datasets using eight popular optimizers with robust statistical validation.

## Key Findings

Our analysis of 360 experiments reveals several important insights:

- **AdamW consistently efficient**: Best balance of performance and low energy consumption across datasets
- **Dataset complexity matters**: Simple vs complex tasks show different optimizer efficiency patterns  
- **SGD excels on complex tasks**: Achieves 20.61% accuracy on CIFAR-100 vs <10% for most others, but at higher energy cost
- **Direct measurement essential**: Training time and energy consumption are not perfectly correlated
- **Statistical significance confirmed**: Differences between optimizers are statistically significant (p < 0.01)

## Experimental Setup

### Datasets Tested
- **MNIST**: 60,000 handwritten digits (421,642 model parameters)
- **CIFAR-10**: 50,000 natural images, 10 classes (3,249,994 parameters)  
- **CIFAR-100**: 50,000 natural images, 100 classes (3,296,164 parameters)

### Optimizers Evaluated
SGD, Adam, AdamW, RMSprop, Adagrad, Adadelta, Adamax, NAdam

### Methodology
- **360 total experiments**: 3 datasets × 8 optimizers × 15 random seeds
- **Robust statistics**: 15 seeds per configuration for statistical validation
- **Comprehensive metrics**: accuracy, training time, CO2 emissions, memory usage
- **Statistical analysis**: Friedman tests with Bonferroni correction for multiple comparisons

## Repository Structure

```
optimizer-energy-study/
├── README.md
├── LICENSE  
├── requirements.txt
├── paper/                          # LaTeX manuscript
│   ├── optimizer_energy_efficiency.tex
│   ├── optimizer_energy_efficiency.pdf
│   └── references.bib
├── src/                           # Experiment code
│   └── experiment_runner.py
├── data/                          # Raw experimental data
│   └── experimental_data/
│       ├── comprehensive_results.csv
│       ├── epoch_details.csv
│       └── emissions/
└── results/                       # Analysis outputs
    └── plots/
        ├── accuracy_vs_emissions.png
        ├── training_duration_boxplots.png
        ├── emissions_rate_heatmap.png
        └── statistical_significance.png
```

## Getting Started

### Prerequisites

```bash
git clone https://github.com/tomalmog/optimizer-energy-study.git
cd optimizer-energy-study
pip install -r requirements.txt
```

### Hardware Requirements
- Modern CPU/GPU with at least 16GB RAM
- Apple M1 Pro used for original experiments (for exact replication)
- GPU acceleration support (MPS/CUDA) recommended

### Running Experiments

```bash
# Run full experimental suite (several hours)
python src/experiment_runner.py

# Results saved to data/experimental_data/
```

The experiment script automatically:
- Downloads required datasets
- Runs training with all optimizer configurations  
- Tracks energy consumption using CodeCarbon
- Saves detailed results for analysis

### Analyzing Results

Pre-computed results are available in `data/experimental_data/`. Visualization code is embedded in the experiment runner script.

## Results Summary

| Dataset | Highest Accuracy | Most Efficient | Fastest |
|---------|------------------|----------------|---------|
| MNIST | Adadelta (98.29%) | NAdam | AdamW |
| CIFAR-10 | AdamaxV (66.53%) | AdamW | Adadelta |
| CIFAR-100 | SGD (20.61%) | AdamW | NAdam |

### Efficiency Rankings
1. **AdamW**: Consistently high efficiency across problem types
2. **NAdam**: Excellent for simpler tasks  
3. **SGD**: High performance on complex problems but energy-intensive

## Technical Details

### Energy Measurement
- **Tool**: CodeCarbon 3.0.4 with macOS powermetrics
- **Metrics**: CPU/GPU power, CO2 emissions, memory usage
- **Carbon intensity**: Ontario, Canada grid factor

### Statistical Methods
- Friedman tests for overall optimizer differences
- Wilcoxon signed-rank tests for pairwise comparisons  
- Bonferroni correction for multiple testing
- Effect size analysis with Cohen's d

## Citation

If you use this work, please cite:

```bibtex
@article{almog2025optimizer,
  title={An Analysis of Optimizer Choice on Energy Efficiency and Performance in Neural Network Training},
  author={Almog, Tom},
  institution={University of Waterloo},
  year={2025}
}
```

## Contributing

Contributions welcome:
- Bug reports and fixes
- Extension to additional optimizers or datasets
- Replication studies on different hardware
- Methodology improvements

## Contact

Tom Almog  
University of Waterloo  
talmog@uwaterloo.ca

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- University of Waterloo for computational resources
- Open-source ML community for development tools
- CodeCarbon team for energy measurement framework

## Practical Recommendations

Based on experimental evidence:

1. **Default choice**: Use AdamW when environmental impact matters
2. **Research settings**: SGD may justify higher emissions for challenging datasets
3. **Simple tasks**: Prioritize efficiency over minor accuracy differences  
4. **Complex tasks**: Weigh performance gains against environmental costs

See paper for detailed analysis and recommendations.
