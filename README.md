# KTP Classification

<br />
<div align="center">
  <a href="">
    <img src="static/KTP Classification.png">
  </a>
</div>

<p></p>

<!-- TABLE OF CONTENTS -->
<details>
  <p>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#background">Background</a></li>
    <li><a href="#business-values">Business Values</a></li>
    <li><a href="#how-to-works">How to Works</a></li>
    <li><a href="#limitations-model">Limitations Model</a></li>
    <li><a href="#next-step">Next Step</a></li>
    <li><a href="#mlops-concepts">MLOPS Concepts</a></li>
    <li><a href="#aiml-canvas">AI/ML Canvas</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
  </ol>
  </p>
</details>


<p></p>

<!-- ABOUT THE PROJECT -->
## About The Project

This project is an AI tool based on CNN that classifies documents, including KTP or non-KTP. This project is based on CNN with the transfer learning method (using efficient-net).


### Built With

These are list any major frameworks/libraries used to make the project.

* [![Pytorch][Pytorch]][Pytorch-url]
* [![Streamlit][Streamlit]][Streamlit-url]
* [![Fastapi][Fastapi]][Fastapi-url]

## Background

Companies in fields such as finance, logistics, etc., definitely have customers and vendors. The company requires customer and vendor documents for administrative purposes. One of the documents is an ID card (like a KTP). However, it is a challenge to ensure that the documents sent by the customer/vendor are correct.

We can use AI tools based on CNN to classify the document entered by the user/customer as correct. This project creates AI tools based on CNN with a transfer learning method to classify documents, including KTP or Non-KTP.

## Business Values
- Can receive the correct data according to company needs.
- Reduce manual work.

## How to Works
The current project is web (so it's easier to demonstrate); to build in production, it's better in API.
You only need to input an image for this web project, and the results will come out.

## Next Step 
The model can be retrained, even with other ID card data such as NPWP, SIM, etc.

## MLOPS Concepts
<p align="center">
  <img src = "static/MLOPS.png">
</p>

- **Versioning**. The project includes automated versioning using semantic release Python.
- **CI/CD**. The project uses the CI/CD concept with GitHub action.
- **Testing**. Testing includes data, model, and code testing.
- **Reproducibility**.
- **Monitoring**. For the next step, the project can monitor forecasting results. We can use [evidentlyAI](https://www.evidentlyai.com/)  for tool monitoring. The monitoring includes data drift, target drift, etc.

## AI/ML Canvas
Link AI/Canvas for this project. [LINK](https://github.com/ALDOPUTRA07/automated_forecasting/blob/main/static/AI_ML%20Canvas%20KTP%20Classification.pdf)
<p align="center">
  <img src = "static/AI_ML Canvas KTP Classification.png">
</p>


<!-- GETTING STARTED -->
## Getting Started
This is a tutorial for running a project locally.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/ALDOPUTRA07/ktp_classification
   ```
2. Change to the project directory
   ```sh
   cd ktp_classification
   ```
3. Setting up programming environment to run the project
   ```sh
   poetry shell
   ```
4. Install the dependencies
   ```sh
   poetry install
   ```
5. Running the project (Without 🐳 Docker)
   ```sh
   poetry run python run.py
   ```

## License
MIT

<p align="right">(<a href="#automed-forecasting">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[Pytorch]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[Pytorch-url]: https://pytorch.org/
[Streamlit]: https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white
[Streamlit-url]: https://streamlit.io/
[Fastapi]: https://img.shields.io/badge/fastapi-109989?style=for-the-badge&logo=FASTAPI&logoColor=white
[Fastapi-url]: https://fastapi.tiangolo.com/