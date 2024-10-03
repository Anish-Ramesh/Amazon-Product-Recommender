# Amazon Product Recommender

## Introduction

In today's vast online marketplaces like **Amazon**, users are often overwhelmed by the sheer volume of products available. Making a decision can be daunting, especially when browsing through a myriad of similar products. Our **Amazon Product Recommender System** simplifies this process by providing personalized, intelligent product recommendations.

This system leverages the latest in **AI-powered natural language processing** to interact with users and recommend products based on their specific needs and preferences. It is designed to be domain-specific, focusing on Amazon products, and offers a seamless conversational experience where users can ask about products and receive tailored suggestions. This not only enhances the user experience but also improves decision-making by offering relevant product information directly from the data scraped from Amazon.

### Why is it Useful?

- **Personalized Recommendations**: Users receive product suggestions that are most relevant to their queries, reducing the time spent on product research.
  
- **Natural Interaction**: The system allows users to interact in a conversational manner, making it easier to explore product details, compare items, and discover new options without needing to browse through endless product listings.

- **Domain-Specific Precision**: By focusing on Amazon products, the system is finely tuned to recommend items within that marketplace, ensuring that recommendations are accurate and specific to user needs.

- **Efficient Information Retrieval**: With product details stored and vectorized from scraped data, the system can quickly retrieve the most relevant information from a large dataset, making the user experience fast and responsive.

- **Enhanced User Decision-Making**: By providing contextually rich information and recommendations, the system helps users make more informed purchasing decisions.

---

## Project Overview

This project is a **Recommendation System** tailored for products on **Amazon**, utilizing **scraped product details** that are stored in PDF format. The project is built using a combination of cutting-edge tools and technologies, offering an advanced **RAG (Retrieval-Augmented Generation)** model powered by **Llama 3 (7B parameter model)** from **Ollama**. The system can interact with the product data and provide intelligent recommendations based on the user's queries.

### Key Features

- **Data Handling**: 
  - Scraped product details from Amazon are stored in PDF format.
  - The PDFs are vectorized and stored in **ChromaDB** for fast retrieval.
  
- **Recommendation System**:
  - Implemented using **LangChain** to connect data retrieval and model inference.
  - **Ollama Server** runs **Llama 3 (7B model)**, providing conversational recommendations from PDF data.
  
- **RAG Implementation**: 
  - Combines **retrieval** of product data from ChromaDB and **generation** of recommendations using Llama 3.
  
- **Backend**: 
  - Developed using **Flask**, handling API requests and serving the recommender system.
  
- **Frontend**:
  - A simple, interactive UI built with **HTML, CSS, and JavaScript**.
  
- **Additional Model**:
  - A **Simple Sentence Transformer** is integrated to enhance natural language understanding.

---

## Architecture

1. **PDF Data Storage**: Scraped Amazon product details are stored in PDF files.
2. **Vectorization**: PDF data is vectorized using embedding techniques and stored in **ChromaDB**.
3. **RAG Model**:
    - **Retrieval**: Relevant product data is retrieved from ChromaDB based on the user's query.
    - **Augmentation**: The retrieved data is passed to the **Llama 3 model** via **LangChain**.
    - **Generation**: Llama 3 generates context-aware responses that recommend products.
4. **Flask Backend**: Manages API endpoints for data processing and model inference.
5. **Frontend UI**: Allows users to input queries and receive product recommendations.

---

## Installation

### Prerequisites

- **Python 3.8+**
- **Ollama Server**
- **ChromaDB**
- **Flask**
- **LangChain**
- **HTML/CSS/JS** (for frontend)
