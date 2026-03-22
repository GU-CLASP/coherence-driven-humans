# Multimodal character grounding

We define multimodal character grounding as a measure of multimodal coherence specific to story characters.
Unlike character persistence, this metric explicitly uses visual modality.

The metric has two components:
1. Multimodal Character Continuity (MCC):
   - measures how closely continuous references to characters in text match character appearances in the visual sequence.
2. GROOViST (GV):
   - a complementary story-level visual grounding metric.

These two components are combined into a single MCG score.

For each story:
1. Start from the character list used for character persistence.
2. For each matched character `c`, construct:
   - a text-side binary sequence (character mentioned per story segment)
   - an image-side binary sequence (character present per segment-aligned image)
3. Compute continuity on both sequences:
   - text continuity: `tc_c`
   - image continuity: `ic_c`
4. Character-level multimodal continuity:
   - `mcc_c = 1 - |tc_c - ic_c|`
5. Story-level MCC:
   - mean of `mcc_c` over matched characters in the story
6. Final MCG score:
   - combines MCC and GROOViST into one multimodal character grounding value.

Interpretation:
- Higher values indicate stronger character-level continuity alignment relative to overall grounding.
- If a story has no valid character annotation or no matched character between image-side annotations and text-side coreference chains, assign `0` for MCC and `0` for MCG.


Main notebook:
- [analysis/mci_profile.ipynb](../../analysis/mci_profile.ipynb)


## Inputs and Where They Come From

MCI notebook inputs:
1. GROOViST per-story table
- [analysis/analysis_data/groovist/groovist_metrics_per_story.csv](../../analysis/analysis_data/groovist/groovist_metrics_per_story.csv)
- Produced by: [analysis/groovist_profile.ipynb](../../analysis/groovist_profile.ipynb)

2. MCC per-story tables
- [analysis/analysis_data/mcc/mcc_metrics_story_original_60.csv](../../analysis/analysis_data/mcc/mcc_metrics_story_original_60.csv)
- [analysis/analysis_data/mcc/mcc_metrics_story_original_54.csv](../../analysis/analysis_data/mcc/mcc_metrics_story_original_54.csv)
- [analysis/analysis_data/mcc/mcc_metrics_story_large_54.csv](../../analysis/analysis_data/mcc/mcc_metrics_story_large_54.csv)
- Produced by: [analysis/mcc_profile.ipynb](../../analysis/mcc_profile.ipynb)


## Run Order

1. Produce/refresh MCC outputs in [analysis/analysis_data/mcc](../../analysis/analysis_data/mcc) via [analysis/mcc_profile.ipynb](../../analysis/mcc_profile.ipynb).
2. Produce/refresh GROOViST outputs in [analysis/analysis_data/groovist](../../analysis/analysis_data/groovist) via [analysis/groovist_profile.ipynb](../../analysis/groovist_profile.ipynb).
3. Run [analysis/mci_profile.ipynb](../../analysis/mci_profile.ipynb) to merge MCC + GROOViST and compute MCI outputs.

## Output Files

Output directory:
- [analysis/analysis_data/mci](../../analysis/analysis_data/mci)

## References

- Surikuchi, A. K., Pezzelle, S., and Fernandez, R. (2023). GROOViST: A Metric for Grounding Objects in Visual Storytelling. Proceedings of EMNLP 2023, 3331-3339. https://aclanthology.org/2023.emnlp-main.202/
- Huang, Q., Xiong, Y., Rao, A., Wang, J., and Lin, D. (2020). MovieNet: A Holistic Dataset for Movie Understanding. https://arxiv.org/abs/2007.10937
- Hong, X., Sayeed, A., Mehra, K., Demberg, V., and Schiele, B. (2023). Visual Writing Prompts: Character-Grounded Story Generation with Curated Image Sequences. Transactions of the Association for Computational Linguistics, 11, 565-581. https://aclanthology.org/2023.tacl-1.33/
- Bohnet, B., Alberti, C., and Collins, M. (2023). Coreference Resolution through a seq2seq Transition-Based System. Transactions of the Association for Computational Linguistics, 11, 212-226. https://doi.org/10.1162/tacl_a_00543
- Porada, I., Zou, X., and Cheung, J. C. K. (2024). A Controlled Reevaluation of Coreference Resolution Models. Proceedings of LREC-COLING 2024, 256-263. https://aclanthology.org/2024.lrec-main.23
