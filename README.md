# Reproducing and Extending LAQuer for Post-Hoc Attribution

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ltellado/LAQuer-Audit/blob/main/notebooks/tests/laquer-audit-unit-tests.ipynb)

This repository provides the official standalone experimental framework, reproduction scripts, and post-hoc audit extension for:
> **Reproducing and Extending LAQuer for Post-Hoc Attribution**  
> *CAI5307: Natural Language Processing — Final Project*  
> *Author: Leo Tellado (University of South Florida)*

---

## 1. Quick Start & Setup

Clone the repository and run the setup script to install dependencies and pin the official upstream LAQuer baseline repository (`eranhirs/LAQuer` at commit `a4affb2`):

```bash
git clone https://github.com/ltellado/LAQuer-Audit.git
cd LAQuer-Audit
chmod +x setup.sh
./setup.sh