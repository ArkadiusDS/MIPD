# MIPD Dataset

Official Code and Data repository of our paper **MIPD: Exploring Manipulations and Intentions In a Novel Corpus of Polish Disinformation**

## Overview
The **MIPD Dataset** is a novel collection of **15,356 Polish web articles**, annotated with four key labels: whether the article is disinformation, intention types, manipulation techniques, and thematic categories. The dataset was curated by a team of professional fact-checkers and debunkers using a detailed methodology to identify and categorize disinformation. This dataset is designed for research into disinformation, its patterns, and the methods used to manipulate information in online media.

## Data Content

### 1. Article Classification
Each article in the dataset has been classified as either:

- **Credible**
- **Disinformative**

Articles that were labeled as "hard to say" or containing misinformation were excluded from the published dataset, focusing the data on binary classifications between disinformation and credibility.

### 2. Intention Types
For each disinformative article, experts annotated the creator's intention. The taxonomy includes:

- **Negating Scientific Facts (NSF)**
- **Undermining the Credibility of Public Institutions (UCPI)**
- **Challenging an International Organization (CIO)**
- **Promoting Social Stereotypes/Antagonisms (PSSA)**
- **Weakening International Alliances (WIA)**
- **Changing Electoral Beliefs (CEB)**
- **Undermining the International Position of a Country (UIPC)**
- **Causing Panic (CP)**
- **Raising Morale of a Conflict's Side (RMCS)**

### 3. Manipulation Techniques
The dataset identifies manipulation techniques used in disinformative articles. These include:

- **Cherry Picking (CHP)**: Selectively presenting data to support a specific argument.
- **Quote Mining (QM)**: Distorting someone's statements by using selective excerpts.
- **Anecdote (AN)**: Using personal stories to discredit statistical evidence.
- **Whataboutism (WH)**: Deflecting from the topic by introducing an unrelated argument.
- **Strawman (ST)**: Misrepresenting an argument to make it easier to attack.
- **Leading Questions (LQ)**: Posing questions in a way that suggests a predetermined conclusion.
- **Appeal to Emotion (AE)**: Manipulating the reader's emotions to sway opinion.
- **False Cause (FC)**: Assuming causation from mere correlation.
- **Exaggeration (EG)**: Overstating or understating facts.
- **Reference Error (RE)**: Citing unreliable or false sources.
- **Misleading Clickbait (MC)**: Creating a misleading or sensational headline that contradicts the article's content.

### 4. Thematic Categories
Each article is classified into one of ten thematic categories based on its content:

- **COVID-19 (COVID)**
- **Migrations (MIG)**
- **LGBT+**
- **Climate Crisis (CLIM)**
- **5G**
- **War in Ukraine (WUKR)**
- **Pseudomedicine (PSMED)**
- **Women’s Rights (WOMR)**
- **Paranormal Activities (PA)**
- **News or Other (NEWS)**

These categories help structure the data and allow for targeted analysis of disinformation on different subjects.

## Annotation Methodology
The dataset was annotated by a team of five Polish debunking and/or fact-checking experts, each with at least three years of experience. The process involved several steps to ensure high-quality annotations:

1. **Article Evaluation**: Articles were independently evaluated by two experts.
2. **Consensus Building**: If the evaluations differed, a consensus was reached through discussion. If no consensus was found, the article was labeled as "hard to say" and excluded from the final dataset.

## Data Statistics
Two basic statistics about full dataset:

- **Average Article Length (Words)**: 767 words per article on average.
- **Number of Articles**: A total of 15,356 articles distributed across thematic categories.

## Sources
The articles were gathered from more than **400 publicly available sources**, including government-managed sites, alternative media, and websites containing conspiracy theories and propaganda. This diverse range of sources ensures the dataset reflects a broad spectrum of perspectives and content.

## Usage
The MIPD dataset can be used to, e.g.:

- Study disinformation patterns.
- Analyze the manipulation techniques used in disinformative content.
- Analyze disinformation per different thematic category


## License
The MIPD dataset is made available under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International Public License (CC BY-NC-ND 4.0).
For full details, please refer to the [LICENSE](https://github.com/ArkadiusDS/MIPD/blob/master/LICENSE-CC-BY-NC-ND-4.0.md).
