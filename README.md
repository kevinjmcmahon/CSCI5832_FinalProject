# **SemEval 2026 Task 9: Multilingual Polarization Detection**

**A Final Project for Natural Language Processing (CSCI 5832) at the University of Colorado Boulder**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code and documentation for our submission to SemEval 2026 Task 9: "Detecting Multilingual, Multicultural and Multievent Online Polarization". This project was completed as the final requirement for the Natural Language Processing course at the University of Colorado Boulder. The work within this respository is the experimentation scripts that were originally created in Google Colab to leverage high-perfoermance GPUs.

## **Introduction**

In this project, we explore the detection of online polarization across multiple languages and cultural contexts. The goal is to develop a system that can accurately identify polarized text, classify its type, and identify how that polarization is manifested. Our approach utilizes [mention your high-level approach, e.g., transformer-based models, multilingual embeddings] to tackle this challenging NLP task.

## **Task Description**

SemEval 2026 Task 9 focuses on identifying online polarization. The task is broken down into the following subtasks:

*   **Subtask 1: Polarization Detection:** Identifying whether a text exhibits polarization.
*   **Subtask 2: Polarization Type Classification:** Classifying polarized content into specific types (e.g., political, social).
*   **Subtask 3: Manifestation Identification:** Determining how the polarization is expressed.

This project will focus on each of the subtasks. The dataset provided for this task includes text in over 20 languages. Our models created will be trained using 5 languages that are still to be decided.

### **Data**

The data for this task can be obtained from the [Official SemEval 2026 Task 9 website]([https://semeval.github.io/SemEval2026/](https://www.codabench.org/competitions/10522/)). Once downloaded, place the data in the `/data` directory.

## **Methodology**

We took an experimental appraoch to the subtasks in this project. In the binary classification we tested model performance using different data augmentation techniques to see what lead to the best results. In the multi-class classification (subtask 3) the team created an ensemble of pre-trained models to see if the strengths of each model could be highlighted in the system and lead to better overall results.

### **Results**

From subtask 1: more data, more training leads to better results. Trying to be clever with data augmentations are hit or miss. From subtask 3: can't extract the best attributes from a pre-trained model doesn't work. Performance ultimately averages in ensemble models.

A detailed analysis of our results can be found in our final project report.

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## **Acknowledgments**

*   We would like to thank our instructor, James Martin, and the teaching assistants for their guidance and support.
*   We also thank the organizers of SemEval for creating this valuable shared task.
