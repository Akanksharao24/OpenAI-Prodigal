# Amazon Archaeological Discovery & Research Platform

## Overview

The **Amazon Archaeological Discovery Platform** provides a state-of-the-art suite for predicting and analyzing archaeological sites in the Amazon Basin, using advanced machine learning models, real archaeological data, and professional computer vision analysis.

The platform integrates data from verified archaeological sites, indigenous group information, environmental factors, and satellite data. It features an advanced prediction model that leverages historical data to provide insights into archaeological locations and their significance.

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

* Machine learning algorithms for predicting archaeological site locations based on real-world data
* Uses historical site data, environmental factors, and indigenous data

### 2. **Satellite Data Integration**

* Integrates NASA and ESA satellite data for site identification
* Uses geospatial data for accurate site prediction

### 3. **AI-Powered Analysis**

* Utilizes OpenAI GPT-4 for in-depth archaeological analysis and interpretation

### 4. **Geospatial Mapping**

* Includes professional mapping tools like Folium and GeoPandas for visualizing and analyzing site locations

### 5. **Data Logging and Error Handling**

* Advanced logging system to track data operations, API requests, and analysis success

---

## Project Structure

```
amazon-archaeological-platform/
├── Amazon_Archaeological_Research_Platform.py  #Research Dashboard
├── Amazon_Archaeological_Discovery_Platform    #Discover Dashboard
├── requirements.txt                            # Python dependencies
├── README.md                                   # Project overview and instructions
└── LICENSE                                     # MIT License
```

---

## Requirements

* **Python**: Version 3.12
* **Streamlit**: For building the platform's web interface
* **Scikit-learn**: For machine learning models (RandomForest, SVM, etc.)
* **GeoPandas**: For geospatial data processing (optional)
* **Plotly**: For visualization of geospatial data and analysis

---

## Installation Guide

1. Clone the repository:

```bash
git clone https://github.com/Prodigal-AI/amazon-archaeological-platform.git
cd amazon-archaeological-platform
```

2. Create and activate a Python 3.12 virtual environment:

```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running Your First Example

1. ** Script**: To test the platform, navigate to `src/amazon_archaeological_platform` and run the `example.py` script.

```bash
streamlit run Amazon_Archaeological_Research_Platform.py
```
```bash
streamlit run Amazon_Archaeological_Discovery_Platform.py
```

2. The script will prompt you for your API keys and relevant credentials.

3. The system will analyze the data and return site predictions and archaeological analysis.

---

## Methodology

1. **Data Collection**: The platform integrates data from multiple sources:

   * Verified archaeological site records
   * Environmental and geospatial data (satellite imagery, river systems, soil types)
   * Indigenous groups' territories and their traditional earthwork practices

2. **Machine Learning Models**:

   * **Feature Engineering**: Features such as latitude, longitude, elevation, rainfall, and structure count are used to train machine learning models like Random Forest and SVM.
   * **Model Training**: Models are trained using historical site data, including confirmed archaeological locations and environmental data.

3. **Geospatial Analysis**:

   * Using **GeoPandas** and **Folium**, the platform visualizes archaeological site locations and their proximity to environmental factors such as rivers and elevation.

4. **AI-Powered Insights**:

   * **OpenAI GPT-4** provides contextual analysis of the archaeological findings and generates detailed reports based on the machine learning predictions.

5. **Site Clustering**:

   * Clustering algorithms like **DBSCAN** are used to group nearby archaeological sites, helping identify large regions with potential historical significance.

---

## Contributing

We welcome contributions! Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to the platform.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details.

---

