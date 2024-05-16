# NER False Positive Removal using Language Models

This project demonstrates the use of language models to analyze Named Entity Recognition (NER) responses and remove false positives from the results. It leverages the power of the Mistral-7B language model to understand the context of entities and filter out incorrect predictions.

## Features

- Accept NER responses as input in JSON format
- Integrate the Mistral-7B language model to analyze the context of entities
- Implement a filtering mechanism to identify and remove false positives from the NER responses
- Optimize the program for efficiency and scalability to handle large volumes of NER data

## Prerequisites

To run this project, you need the following dependencies:

- Python 3.x
- PyTorch
- Transformers
- Langchain
- FAISS
- Sentence Transformers
- PyPDF2

You can install the required packages using the following command:

```
pip install -q -U torch tensorflow transformers langchain faiss-cpu sentence_transformers
pip install -q peft==0.4.0 trl==0.4.7 accelerate==0.21.0 bitsandbytes==0.41.3
pip install pypdf PyPDF2
```

## Usage

1. Clone the repository

2. Prepare your NER response data in JSON format and save it in the project directory. The JSON file should have the following structure:

```json
[
  [
    {
      "matches": {
        "false_positives": {
          "entity_page_mapping": [
            {
              "text": "False positive entity 1"
            },
            {
              "text": "False positive entity 2"
            }
          ]
        },
        "false_negative": {
          "entity_page_mapping": [
            {
              "text": "False negative entity 1"
            },
            {
              "text": "False negative entity 2"
            }
          ]
        },
        "true_positive": {
          "entity_page_mapping": [
            {
              "text": "True positive entity 1"
            },
            {
              "text": "True positive entity 2"
            }
          ]
        }
      }
    }
  ]
]
```

3. Open the Jupyter Notebook `NER_LLM.ipynb` and run the cells in sequential order.

4. When prompted, enter the file name of your NER response JSON file.

5. The program will load the NER data, analyze the context of entities using the Mistral-7B language model, and remove false positives from the responses.

6. The filtered NER data and false positive information will be displayed in the notebook output.

## Example

Here's an example of how to use the program:

1. Save your NER response JSON file as `ner_response.json` in the project directory.

2. Open the `NER_LLM.ipynb` notebook and run the cells.

3. When prompted, enter the file name `ner_response.json`.

4. The program will process the NER response and display the filtered NER data and false positive information.

Output:
```
Answer:
Based on the given NER data, here are the filtered NER entities and false positives:

Filtered NER Data:
{
  'ADRESS': ['True positive entity 1', 'True positive entity 2', 'False negative entity 1', 'False negative entity 2'],
  'DATE': ['True positive entity 1', 'True positive entity 2', 'False negative entity 1', 'False negative entity 2'],
  'EMAIL': ['True positive entity 1', 'True positive entity 2', 'False negative entity 1', 'False negative entity 2'],
  'MANUAL_MARKED': ['True positive entity 1', 'True positive entity 2', 'False negative entity 1', 'False negative entity 2'],
  'NAME': ['True positive entity 1', 'True positive entity 2', 'False negative entity 1', 'False negative entity 2'],
  'ORGANIZATION': ['True positive entity 1', 'True positive entity 2', 'False negative entity 1', 'False negative entity 2'],
  'METRICS': ['True positive entity 1', 'True positive entity 2', 'False negative entity 1', 'False negative entity 2']
}

False Positives:
{
  'ADRESS': ['False positive entity 1', 'False positive entity 2'],
  'DATE': ['False positive entity 1', 'False positive entity 2'],
  'EMAIL': ['False positive entity 1', 'False positive entity 2'],
  'MANUAL_MARKED': ['False positive entity 1', 'False positive entity 2'],
  'NAME': ['False positive entity 1', 'False positive entity 2'],
  'ORGANIZATION': ['False positive entity 1', 'False positive entity 2'],
  'METRICS': ['False positive entity 1', 'False positive entity 2']
}
```

## Configuration

- The `get_NER()` function in the notebook accepts the file name of the NER response JSON file as input. Make sure to provide the correct file name when prompted.

- The language model parameters such as temperature, repetition penalty, and maximum new tokens can be adjusted in the `text_pipeline` and `text_pipeline2` definitions to fine-tune the model's behavior.

## Optimization

The program is optimized for efficiency and scalability to handle large volumes of NER data. The following optimizations are implemented:

- The language model pipeline parameters are adjusted to control the randomness and repetition of generated text, ensuring more deterministic and concise outputs.

- The maximum number of new tokens is limited to prevent excessively long outputs, improving processing speed.

- The program leverages the power of the Mistral-7B language model, which is a highly efficient and scalable model for natural language processing tasks.
