# Generative AI Extensions Workshop

Welcome to the Secure Generative AI Extensions Workshop! This technical workshop is designed to provide participants with a comprehensive understanding of how to develop AI-powered extensions within the SAP Business Technology Platform (BTP). Throughout this two-hour session, you'll gain practical experience with SAP Generative AI Hub and the HANA Cloud Vector Store, using open-source models and embeddings to construct a Retrieve-Augment-Generate (RAG) scenario.

## Workshop Overview

The workshop focuses on basic development to provide a foundation for developments with AI. This session is perfect for colleagues looking to improve their technical skills in SAP's AI capabilities and for those who wish to teach customers in building generative AI extensions.

### Key Takeaways

By the end of this workshop, you will:

- Understand how to use SAP Generative AI Hub and HANA Cloud Vector Store.
- Be able to build a RAG scenario with Azure-OpenAI models and according embeddings.

## Prerequisites

- Basic understanding of Python programming.
- Familiarity with SAP BTP and AI concepts.
- An active SAP BTP account (for hands-on exercises).

## Workshop Limitations

Please note that the workshop is limited to 20 participants to ensure a quality learning experience for all attendees.

## Setup Instructions

Before participating in the workshop, please ensure you have the following setup completed:

1. Clone this repository to your local machine.
2. Install Python 3.12 
3. Install the required Python packages using `pip`:
   ```sh
   pip install -r requirements.txt```

Create a .env file in the root directory of this project - it will be provided in the workshop.

## Access test programs

HANA Vector Store Access Test
To verify your connection to the HANA Cloud Vector Store, run the _step1_hanavs_access.py_ script:

```python step1_hanavs_access.py```
This script will test your connection and provide feedback in the console.

LLM Access Test
To test access to the Large Language Model (LLM), run the step2_llm_access.py script:


```python step2_llm_access.py```
This script will ask a predefined question to the LLM and output the response.

Support
If you encounter any issues or have questions, please open an issue in this repository, and we'll get back to you as soon as possible.

License
This workshop and its materials are provided under the MIT License.

We look forward to seeing you at the workshop and exploring the world of secure generative AI extensions together!
