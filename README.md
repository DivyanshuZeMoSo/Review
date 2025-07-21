# Prompt Engineering Draft
This repository contains document related to the katas problem mentioned in the WBS sheet based upon the topic mentioned above.

## 5. Prompt Engineering DRAFT

### Problem 1

Your task is to design a system, given a generic question about a company's data, performs two main functions through prompt engineering with a Large Language Model (LLM):

1. Transforms the generic question into a detailed, context-specific question. This involves identifying the relevant company and aspects of the query that need specificity. The output should be a JSON object containing the refined question and the name of the targeted database.
2. Generates an accurate SQL query based on the refined question and the specified database schema. This SQL query should be capable of retrieving the relevant data from the database.

Deliverables:

- A document detailing the prompts used for both stages of the process, including the rationale behind the prompt design and any iterations or refinements made.
- Examples of generic questions, the JSON outputs generated from these questions, and the final SQL queries produced.
- A brief analysis of the system's effectiveness, including any challenges encountered in guiding the LLM to produce accurate and relevant outputs at both stages.
