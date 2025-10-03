Weighted Transformer Trainer for Data Imbalanced Multi Emotion Text Classification
-------------------

This work explores the application of a simple weighted loss function to Transformer-based models for multi-label emotion detection in SemEval-2025 Shared Task 11. Our approach addresses data imbalance by dynamically adjusting class weights, thereby enhancing performance on minority emotion classes without the computational burden of traditional resampling methods. We evaluate BERT, RoBERTa, and BART on the BRIGHTER dataset, using evaluation metrics such as Micro F1, Macro F1, ROC-AUC, Accuracy, and Jaccard similarity coefficients. The results demonstrate that the weighted loss function improves performance on high-frequency emotion classes but shows limited impact on minority classes. These findings underscore both the effectiveness and the challenges of applying this approach to imbalanced multi-label emotion detection.




Cite the work
```
@inproceedings{cui-2025-xiacui,
    title = "xiacui at {S}em{E}val-2025 Task 11: Addressing Data Imbalance in Transformer-Based Multi-Label Emotion Detection with Weighted Loss",
    author = "Cui, Xia",
    editor = "Rosenthal, Sara  and
      Ros{\'a}, Aiala  and
      Ghosh, Debanjan  and
      Zampieri, Marcos",
    booktitle = "Proceedings of the 19th International Workshop on Semantic Evaluation (SemEval-2025)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.semeval-1.37/",
    pages = "247--256",
    ISBN = "979-8-89176-273-2",
    abstract = "This paper explores the application of a simplified weighted loss function to Transformer-based models for multi-label emotion detection in SemEval-2025 Shared Task 11. Our approach addresses data imbalance by dynamically adjusting class weights, thereby enhancing performance on minority emotion classes without the computational burden of traditional resampling methods. We evaluate BERT, RoBERTa, and BART on the BRIGHTER dataset, using evaluation metrics such as Micro F1, Macro F1, ROC-AUC, Accuracy, and Jaccard similarity coefficients. The results demonstrate that the weighted loss function improves performance on high-frequency emotion classes but shows limited impact on minority classes. These findings underscore both the effectiveness and the challenges of applying this approach to imbalanced multi-label emotion detection."
}
    
```
