# Character Persistence

We use character persistence to measure how consistently visually grounded characters are maintained through a story.
The metric combines external character annotations from MovieNet (Huang et al., 2020), linked to VWP stories (Hong et al., 2023).
Text-side coreference chains are extracted using the Link-Append model (Otmazgin et al., 2023; Porada et al., 2024).

For each story, we extract the set of character names associated with its image sequence from VWP annotations.
We then run coreference resolution over the story text and link a coreference cluster to a character when any mention in the cluster matches the character name by case-insensitive string matching.
For each matched character this yields the sentence positions in which that character is mentioned.

From these mentions, aligned across image-side character annotations and text-side coreference chains, we compute two sub-metrics:
- Character Continuity (`char_continuity`)
- Character Spread (`char_spread`)

Characters mentioned again after short gaps receive higher Character Continuity, while characters whose mentions cover a larger portion of the story receive higher Character Spread.

For each matched character in each story, we define persistence as:

`char_coherence = tanh(char_continuity / (char_spread + epsilon))`

where `epsilon` is a small constant to avoid division by zero (`1e-8` in the notebook implementation).
The final story-level character persistence score is the mean of these per-character persistence values over all matched characters in the story.

If a story has no valid character annotation or no character-coreference match, we assign a score of `0` for that case.

## Main Implementation

Primary notebook:
- [analysis/character_profiles.ipynb](../../analysis/character_profiles.ipynb)

## Metric Definition (Code Level)

For each story document with `N` sentences and each matched character:

1. Build a sentence-level presence vector:
   - `presence[s] = 1` if the character is present in sentence `s`, else `0`.
2. Character Continuity (`char_continuity`):
   - average over adjacent sentence pairs of indicator
   - `1` if `presence[s] == 1` and `presence[s+1] == 1`, else `0`
3. Character Spread (`char_spread`):
   - if `s_min` and `s_max` are first/last sentence indices with character presence,
   - `char_spread = (s_max - s_min) / (N - 1)`
4. Character persistence/coherence:
   - `char_coherence = tanh(char_continuity / (char_spread + epsilon))`

## Required Input Files

Character annotations:
- [data/sampled_60/sampled_60_stories.json](../../data/sampled_60/sampled_60_stories.json)

Coreference outputs consumed by the notebook:
- [models/linkappend/data-out/conll_to_json](../../models/linkappend/data-out/conll_to_json)

## Output Files

Output directory:
- [analysis/analysis_data/character](../../analysis/analysis_data/character)

## References

- Huang, Q., Xiong, Y., Rao, A., Wang, J., and Lin, D. (2020). MovieNet: A Holistic Dataset for Movie Understanding.

- Hong, X., Sayeed, A., Mehra, K., Demberg, V., and Schiele, B. (2023). Visual Writing Prompts: Character-Grounded Story Generation with Curated Image Sequences. Transactions of the Association for Computational Linguistics, 11, 565-581.
- Bohnet, B., Alberti, C., and Collins, M. (2023). Coreference Resolution through a seq2seq Transition-Based System. Transactions of the Association for Computational Linguistics, 11, 212-226.

- Porada, I., Zou, X., and Cheung, J. C. K. (2024). A Controlled Reevaluation of Coreference Resolution Models. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), 256-263.