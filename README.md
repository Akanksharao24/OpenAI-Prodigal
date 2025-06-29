

# Amazon Archaeological Discovery Platform

## Overview

The **Amazon Archaeological Discovery Platform** is designed to predict and analyze archaeological sites within the Amazon Basin, integrating advanced machine learning models, satellite data, indigenous group information, and professional computer vision techniques. The platform uses real-world data to predict the locations of potential archaeological sites, providing a deeper understanding of ancient civilizations and their environmental interactions.

---

## Table of Contents

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Requirements](#requirements)
4. [Installation Guide](#installation-guide)
5. [Running Your First Example](#running-your-first-example)
6. [Usage](#usage)
7. [Methodology](#methodology)
8. [Contributing](#contributing)
9. [License](#license)

---

## Features

### 1. **Archaeological Site Prediction**

* Predict archaeological site locations using machine learning algorithms based on historical, environmental, and indigenous data.

### 2. **Satellite Data Integration**

* Incorporate satellite imagery from NASA and ESA to enhance the prediction of archaeological sites.

### 3. **AI-Powered Analysis**

* Use OpenAI GPT-4 for generating detailed reports and analyses of archaeological findings.

### 4. **Geospatial Mapping**

* Visualize archaeological sites and their proximity to geographical features (rivers, elevation) using Folium and GeoPandas.

### 5. **Data Logging and Error Handling**

* Advanced logging system tracks operations, API requests, and analysis outcomes to ensure smooth processing.

---

## Project Structure

```
amazon-archaeological-platform/
├── Research_Platform.py  # Research dashboard
├── Discovery_Platform    # Discovery dashboard
├── requirements.txt                            # Python dependencies
├── README.md                                   # Project overview and instructions
└── LICENSE                                     # MIT License
```

---

## Requirements

* **Python**: Version 3.11
* **Streamlit**: For building the web interface
* **Scikit-learn**: For machine learning algorithms (RandomForest, SVM, etc.)
* **GeoPandas**: For geospatial data processing (optional)
* **Plotly**: For visualizing geospatial data and analyses

---

## Installation Guide

1. Clone the repository:

```bash
git clone https://github.com//OpenAI-prodigal.git
cd OpenAI-prodigal
```

2. Create and activate a Python virtual environment:

```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running Your First Example

1. **Run the Streamlit app:**

```bash
streamlit run Research.py
```

```bash
streamlit run Discovery.py
```

2. The system will ask for API keys and credentials.

3. The platform will analyze the data and provide archaeological site predictions and insights.

---

## Methodology

1. **Data Collection**

   * The platform integrates data from:

     * Verified archaeological site records
     * Environmental and geospatial data (satellite imagery, soil types, river systems)
     * Indigenous territories and traditional earthworks

2. **Machine Learning Models**

   * **Feature Engineering**: Includes data like latitude, longitude, elevation, rainfall, and structure count.
   * **Model Training**: Models are trained on historical site data to predict site locations.

3. **Geospatial Analysis**

   * Uses **GeoPandas** and **Folium** to visualize site locations in relation to geographical features.

4. **AI-Powered Insights**

   * **OpenAI GPT-4** provides in-depth analysis of predictions and generates detailed archaeological reports.

5. **Site Clustering**

   * **DBSCAN** clustering groups nearby archaeological sites, identifying regions of historical significance.

---

## Contributing

We welcome contributions! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute to the platform.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details.

---
