# Hunting-for-Rare-Galaxies-A-Deep-Generative-Approach-Using-ResNet50-and-Variational-Autoencoders
A two-stage hybrid machine learning pipeline combining supervised fine-tuning (ResNet50) and unsupervised generative modeling (VAE) to automate morphological anomaly detection and hunt for rare galaxies in massive astronomical datasets.
# Hunting for Rare Galaxies: A Deep Generative Approach Using ResNet50 and Variational Autoencoders

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced, two-stage computer vision pipeline designed to solve the critical data-explosion problem in modern astrophysics: automating the discovery of rare galactic morphologies across massive sky surveys while concurrently mitigating severe class imbalance.

## 🌌 Overview & Core Concept

In modern astronomy, datasets like Galaxy Zoo yield millions of images. Manually discovering unique or anomalous celestial structures is practically impossible. Traditional supervised networks fail at this task because rare structures lack sufficient training data.

This project implements a unique **two-stage hybrid pipeline**:
1. **Supervised Representation Learning:** A pre-trained **ResNet50** backbone is fine-tuned to map high-dimensional cosmic images ($224 \times 224 \times 3$) into a highly optimized, dense **256-dimensional morphological embedding vector**.
2. **Unsupervised Anomaly Hunting:** A **Variational Autoencoder (VAE)** is trained *strictly* on standard, statistically dominant galaxy configurations (*spirals, smooth rounds*). When a rare or uniquely shaped galaxy profile passes through the system, the VAE fails to reconstruct it smoothly. This generates a high **Mean Squared Error (MSE)**, flagging the target as a rare anomaly.












The model workflow consists of the following consecutive stages:
