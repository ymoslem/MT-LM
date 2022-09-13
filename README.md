# Domain-Specific Text Generation for Machine Translation

Scripts and config files for our paper, [Domain-Specific Text Generation for Machine Translation](https://aclanthology.org/2022.amta-research.2)


## Citation

```bibtex
@inproceedings{moslem-etal-2022-domain,
    title = "Domain-Specific Text Generation for Machine Translation",
    author = "Moslem, Yasmin  and
      Haque, Rejwanul  and
      Kelleher, John  and
      Way, Andy",
    booktitle = "Proceedings of the 15th biennial conference of the Association for Machine Translation in the Americas (Volume 1: Research Track)",
    month = sep,
    year = "2022",
    address = "Orlando, USA",
    publisher = "Association for Machine Translation in the Americas",
    url = "https://aclanthology.org/2022.amta-research.2",
    pages = "14--30",
    abstract = "Preservation of domain knowledge from the source to target is crucial in any translation workflow. It is common in the translation industry to receive highly-specialized projects, where there is hardly any parallel in-domain data. In such scenarios where there is insufficient in-domain data to fine-tune Machine Translation (MT) models, producing translations that are consistent with the relevant context is challenging. In this work, we propose leveraging state-of-the-art pretrained language models (LMs) for domain-specific data augmentation for MT, simulating the domain characteristics of either (a) a small bilingual dataset, or (b) the monolingual source text to be translated. Combining this idea with back-translation, we can generate huge amounts of synthetic bilingual in-domain data for both use cases. For our investigation, we used the state-of-the-art MT architecture, Transformer. We employed mixed fine-tuning to train models that significantly improve translation of in-domain texts. More specifically, our proposed methods achieved improvements of approximately 5-6 BLEU and 2-3 BLEU, respectively, on Arabic-to-English and English-to-Arabic language pairs. Furthermore, the outcome of human evaluation corroborates the automatic evaluation results.",
}
```
