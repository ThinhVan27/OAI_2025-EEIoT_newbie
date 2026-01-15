# Mushroom Image Classification (OAI Challenge 2025)

## Project Overview

This project implements an image classification system for mushroom species identification using deep learning techniques. It is developed as part of the [OAI Challenge 2025](https://oai.hutech.edu.vn/), focusing on cutting-edge and accurate solution.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- `pip` package manager
- 4GB RAM (minimum)
- GPU support recommended (CUDA 11.0+)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd OAI_2025-EEIoT_newbie
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Train and test model:
```bash
python main.py
```

## Project Structure

```
OAI_2025-EEIoT_newbie/
├── dataset/
│   ├── train/
│   └── test/
├── libs/
│   ├── models.py           # Create classification models
│   ├── predict.py          # Predict on test data
│   ├── setup_data.py       # Data preparation
│   ├── train.py            # Train model
│   └── transform.py        # Custome transform
├── output/
│   └── results.csv         # Predict results
├── main.py                 # Main program
├── OAI.pdf                 # Guide for run
├── Presentation.pdf        # Slide  
├── requirements.txt
└── README.md
```

## License
Educational only
