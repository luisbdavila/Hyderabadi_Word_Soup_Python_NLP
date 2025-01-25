# Solving the Hyderabadi Word Soup - Natural Language Processing (NLP)

## Project Overview
This project analyzes restaurant reviews in Hyderabad using text mining techniques to provide insights into customer sentiment, classify cuisines, recognize frequently mentioned dishes, and extract emergent topics. The insights aim to assist the Hyderabad Tourism Board in improving the restaurant landscape and guiding health and safety inspections.

## Objectives
The main goals of this project were:
1. **Sentiment Analysis**: Predict restaurant Zomato scores using review polarity and analyze customer satisfaction levels.
2. **Multilabel Classification**: Classify restaurants by cuisine type based on their reviews.
3. **Co-occurrence Analysis and Clustering**: Identify relationships between frequently mentioned dishes and cluster similar cuisines.
4. **Topic Modeling**: Extract specific topics from reviews to gain deeper insights into customer opinions.

## Problem Statement
Despite structured review systems, free text comments often hold valuable insights that predefined scores cannot capture. This project addresses the challenge of extracting actionable insights from unstructured review text, enabling better categorization of restaurants, understanding customer satisfaction, and identifying health and safety concerns.

## Methodology
### Tools and Techniques
- **CRISP-DM Framework**: Ensured a structured and efficient approach.
- **Datasets**:
  - 105 restaurants in Hyderabad (metadata like cost, cuisine, and timings).
  - 10,000 Zomato reviews (text, ratings, and reviewer metadata).
- **Preprocessing**:
  - Cleaned and integrated datasets, removing irrelevant or missing data.
  - Applied advanced text cleaning techniques, such as tokenization, lemmatization, and stopword removal.
- **Modeling**:
  - Sentiment Analysis: Implemented models like Vader, TextBlob, RoBERTa, and DistilBERT.
  - Cuisine Classification: Used Bag-of-Words, TF-IDF, Doc2Vec, and BERT with Logistic Regression and Random Forest classifiers.
  - Co-occurrence Analysis: Built co-occurrence matrices and created clusters using K-means, HDBSCAN, and Optics.
  - Topic Modeling: Applied LSA, LDA, and BERTopic to extract emergent topics from reviews.

### Evaluation
- Sentiment models were evaluated using correlation metrics (Pearson, Spearman).
- Classification models were assessed using F1-scores, with higher weights for minority classes.
- Clustering and topic modeling were validated through silhouette score and coherence scores.

## Key Insights
1. **Sentiment Analysis**:
   - Majority of reviews are positive, but some highlight serious concerns, such as hygiene and food quality.
   - DistilBERT emerged as the most accurate model for sentiment prediction.
2. **Cuisine Classification**:
   - Captured patterns for popular cuisines like North Indian, Chinese, and Continental with F1-scores exceeding 0.6.
   - Struggled with niche cuisines like Pizza and Latin due to vocabulary overlaps.
3. **Co-occurrence Analysis**:
   - Identified clear dish clusters tied to cuisine types, such as:
     - **Bakery**: Cheesecake, cupcakes, brownies.
     - **Seafood**: Fish, prawn, pomfret.
     - **Fast Food**: Burgers, fries, chicken wings.
4. **Topic Modeling**:
   - LSA and LDA extracted broad topics like food taste and service quality.
   - BERTopic identified finer details such as ordering issues, specific complaints, and trending dishes.

## Deliverables
1. **Report**: A comprehensive document summarizing methods, insights, and recommendations.
2. **Notebooks**: Jupyter Notebooks with all preprocessing, modeling, and evaluation steps.
3. **Visualizations**: Word clouds, heatmaps, and network graphs to illustrate findings.

## Conclusion
This project highlights the power of text mining for extracting insights from unstructured data. The results provide actionable recommendations for the Hyderabad Tourism Board, such as focusing on improving hygiene, highlighting positive aspects in marketing, and addressing specific customer complaints. The methodologies and models can be extended to other domains for similar analyses.
