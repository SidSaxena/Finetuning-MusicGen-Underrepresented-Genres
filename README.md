[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/bBEmmLm3)
# Final Project: Part 2 Audio

---

# Fine-Tuning MusicGen for Under-Represented Genres

### [Link to report - Fine-Tuning MusicGen for Under-Represented Genres](https://docs.google.com/document/d/1NhbUVf7CznpsCQemPcy1Me-4SQZAxHctpjJvE6tpas0/edit?usp=sharing)

**Description:**
This project explores fine-tuning Facebook's MusicGen for Indian classical fusion music generation using techniques inspired by DreamBooth, focusing on style transfer and preserving the characteristic sounds of Indian classical instruments like sitar and tabla within contemporary production contexts.

## Project Overview
We fine-tuned MusicGen-Melody using LoRA (Low-Rank Adaptation) to create models that can generate Indian classical fusion music, particularly in the style of Anoushka Shankar. Through multiple iterations, we developed models with increasingly refined understanding of the fusion genre, enhancing their ability to generate authentic-sounding compositions that blend traditional Indian classical elements with contemporary production techniques.

## Available Models on Hugging Face 🤗

| Model | Description | Link |
|-------|-------------|------|
| **The 1975 - artist fine tuned** | Fine-tuned for The 1975 band style | [🔗 MadJ99/musicgen-melody-the1975](https://huggingface.co/MadJ99/musicgen-melody-the1975) |
| **Indian Classical Fusion 1** | Basic Indian classical fusion model | [🔗 MadJ99/musicgen-melody-as-ch3](https://huggingface.co/MadJ99/musicgen-melody-as-ch3) |
| **Indian Classical Fusion 2** | Expanded dataset with detailed prompt | [🔗 MadJ99/musicgen-melody-as-traes](https://huggingface.co/MadJ99/musicgen-melody-as-traes) |
| **Indian Classical Fusion 3** | Metadata updated for Anoushka Shankar style | [🔗 MadJ99/musicgen-melody-traes-updated](https://huggingface.co/MadJ99/musicgen-melody-traes-updated) |
| **Indian Classical Fusion 4** | Latest model with more training data | [🔗 MadJ99/musicgen-melody-as-new-updated](https://huggingface.co/MadJ99/musicgen-melody-as-new-updated) |


## Setup and Installation

Run either of these commands to install the requirements for the project

```bash
pip install --quiet git+https://github.com/ylacombe/musicgen-dreamboothing demucs msclap transformers wordcloud python-dotenv
```

or

```bash
pip install -r requirements.txt
```

## Running and Reproducibility

If you are using your own dataset locally, we provide two scripts in order to structure the audio files in a suitable format for the pre-processing and training using the MusicGen model
1. Run the chunkmusic.py to segment your audio files into 30 second or lesser chunks in the desired format
How to run chunkmusic.py (example): 

```bash
python chunkmusic.py --input-dir "path/to/data" --output-dir "data/audio/" --chunk-duration 30 --format mp3
```

2. Run preprocess_data.py to format and create your dataset metadata file using:

```bash
python preprocess_data.py --chunks-dir "path" --output-dir "data/datasets/" --sample-rate 32000
```

3. Use the notebook provided and follow the steps to preprocess data and train with MusicGen-melody model
- We use **demucs** in the case the audio has vocals. MusicGen is trained only on Instrumental music, so requires music without vocals in them
- **CLAP** embeddings are used to extract metadata from the dataset

---
---
---

## Task

Think of your Audio Final Project as a conference research paper.
It does not have to be "an original contribution to the literature", but can be a report on what you design as a learning project for yourself.
You may have up to a maximum of 3 co-authors (projects evaluated accordingly).

The components should include:

* A paper with a clear statement of what you set out to explore
* a literature review of relevant papers
* a description of what you did (supporting figures welcome)
* and code that comes with a ~5 minute demo (video such as a screen grab with talk-over).

Just as a few suggestions for topics (meant to inspire, not to limit):

* Explore an architecture such as the transformer input representation, architectural features, sensitivity to data, conditioning. (Train if you have the resources)

* Find trainable DDSP or RAVE models, NoiseBandNet, etc. explore parameters (eg f threshold in RAVE SVD) or module components (e.g. positional encoding)

* Exploring Alternative Tokenization Strategies for Discrete Audio Codecs (DAC) Soundstream, Encodec bit rate efficiency, evaluation strategies, streaming capabilities. Can the Descript DAC stream?

* Latent Space Interpolation and Manipulation in RAVE for Timbre Morphing

* Push the creative potential of a generative audio network

## Deliverables

The final project deliverables are:

1. 4-5 page "conference" paper
2. A GitHub repository link containing the implementation code in the course Github Classroom.
3. A 5-minute video (like you would send to a conference) presenting your work including any demos.


## License

The code in this repository is released under the Apache license as found in the LICENSE file. The pre-trained MusicGen 
weights are licenced under CC-BY-NC 4.0.

## Acknowledgements

This project builds on top of a number of open-source projects, to whom we'd like to extend our warmest thanks for providing these tools!

Special thanks to:
- MusicGen Dreamboothing by Yoach Lacombe (ylacombe) - [Repo](https://github.com/ylacombe/musicgen-dreamboothing)
- The MusicGen team from Meta AI and their [audiocraft](https://github.com/facebookresearch/audiocraft) repository.
- the many libraries used, to name but a few: [🤗 datasets](https://huggingface.co/docs/datasets/v2.17.0/en/index), [🤗 accelerate](https://huggingface.co/docs/accelerate/en/index), and [🤗 transformers](https://huggingface.co/docs/transformers/index).


## Citation

If you found this repository useful, please consider citing the original MusicGen paper:

```
@misc{copet2024simple,
      title={Simple and Controllable Music Generation}, 
      author={Jade Copet and Felix Kreuk and Itai Gat and Tal Remez and David Kant and Gabriel Synnaeve and Yossi Adi and Alexandre Défossez},
      year={2024},
      eprint={2306.05284},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```