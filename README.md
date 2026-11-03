# News Topic Classification on Reuters-21578: MLP vs SVM

Comparison of a Multi-Layer Perceptron and Support Vector Machines for multi-class text classification of Reuters newswire articles. Coursework for the Neural Computing module, MSc Data Science, City St George's, University of London.

## Question

Can an MLP learn semantic patterns in news text better than an SVM, or does its larger parameter count make it overfit?

## Results

Test set, 10 classes, macro-averaged:

| Model | F1 (macro) | Jaccard (macro) |
|---|---|---|
| **MLP** (2 hidden layers, dropout + weight decay) | **0.82** | **0.71** |
| LinearSVC (best SVM) | 0.81 | 0.69 |

- Both models are competitive. The MLP is slightly ahead, but only after regularisation.
- LinearSVC trained faster and matched the other SVM kernels after tuning. On default settings, LinearSVC (one-vs-rest) scored 63.3% Jaccard, while SVC with a linear kernel (one-vs-one) dropped to 49%.
- Both models struggled most on one minority class (class 4): F1 0.64 for MLP and 0.60 for SVM.
- Compute cost: the MLP grid search (96 combinations) took about 40 minutes on 2 cores. The SVM (RBF) search (36 combinations) took about 50 minutes on 4 cores.

## Data

[Reuters-21578](https://archive.ics.uci.edu/dataset/137/reuters+21578+text+categorization+collection) (UCI). After extracting usable documents there were **11,367** articles with 120 topics. Because the class distribution is heavily imbalanced, only the top 15 topics were considered, and the multi-label problem was converted to a **single-label, 10-class** problem by keeping each article's least frequent label, which improved balance.

## Method

1. **Preprocessing:** title + body merged into one text field; link, Unicode and number removal; stemming.
2. **Features:** weighted TF-IDF (probabilistic term weighting to help minority classes), max 1,500 features.
3. **Split:** 70 / 15 / 15 train / validation / test.
4. **MLP (PyTorch):** 128×64 ReLU network, cross-entropy loss, Adam. The default model overfit, so dropout, weight decay and momentum were added, which cut the train–validation loss gap to 0.025. More hidden layers increased overfitting, so the 2-layer design was kept. 96-combination grid search.
5. **SVM (scikit-learn):** LinearSVC, SVC (linear) and SVC (RBF), each with its own grid search.
6. **Selection metric:** Jaccard accuracy, which is more robust to class imbalance. The final models were chosen by looking at train and test scores together, not test alone.

## Lessons learned

- Weighted TF-IDF mattered: standard TF-IDF failed to represent minority classes.
- SVM gives strong, cheap performance on high-dimensional text. The MLP can model richer patterns but needs careful regularisation.
- Next steps: class weighting for the SVM, and a denser MLP grid search with more compute.

## Repository structure

<!-- Adjust to match your actual files -->
```
├── notebook.ipynb     # preprocessing, MLP, SVM, evaluation
├── figures/           # loss curves, confusion matrix, grid search plots
├── report/ -(after results)           # coursework report PDF (optional)
├── requirements.txt
└── README.md
```

## How to run

<!-- Replace with your real commands -->
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

## Tech stack

Python, PyTorch, scikit-learn, pandas, NumPy, NLTK.
<!-- Remove anything you did not use -->

## Author

Deepak Joshi — [LinkedIn](<your-link>)
