The following files are the final predictions for the back translation method:
* pred_tatoeba.en
* pred_wmt_en
* pred_tatoeba_ft.en
* pred_wmt_ft_en

In addition to these predictions, intermediate data was produced and stored in
a folder depending on which model the data belongs to.
* marianmt
* nllb
* mbart
* t5

Each folder contains *.zh and *_sim files.

*.zh files are obtained by translating the predicted english texts back into
Chinese.

*_sim files are obtained by computing the cosine similarity between the back
translated text and the original untranslated text.
