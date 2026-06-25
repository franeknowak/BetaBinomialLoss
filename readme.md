# B² Loss for Evidential Classification of Critical View of Safety in Laparoscopic Cholecystectomy

Official repository for the paper **"B² Loss for Evidential Classification of Critical View of Safety in Laparoscopic Cholecystectomy."**

> Published at MICCAI 2026 (link TBC)

---

## Authors

- **Franciszek M. Nowak**¹ ✉ — [franciszek.nowak.23@ucl.ac.uk](mailto:franciszek.nowak.23@ucl.ac.uk) ([ORCID](https://orcid.org/0009-0006-4969-0139))
- **Evangelos B. Mazomenos**¹ ([ORCID](https://orcid.org/0000-0003-0357-5996))
- **Brian Davidson**¹˒² ([ORCID](https://orcid.org/0000-0002-9152-5907))
- **Matthew J. Clarkson**¹ ([ORCID](https://orcid.org/0000-0002-5565-1252))

¹ UCL Hawkes Institute, Department of Medical Physics and Biomedical Engineering, UCL, London, UK

² Division of Surgery and Interventional Science, UCL, London, UK

---

## Abstract 
Automated recognition of the Critical View of Safety (CVS) in Laparoscopic Cholecystectomy is considered a deterministic classification task using majority vote labels. However, expert annotators' disagreement does not equate annotation error, rather reflecting informative ambiguity in judgement. Reducing such labels to hard targets ignores the underlying sampling process and discards information about observer's variability. 

Our work models annotator votes as Binomial observations of a latent visibility probability and learns its predictive distribution via a conjugate Beta-Binomial ($B^2$) loss with evidential parametrisation. The network outputs a Beta distribution over the latent visibility probability, enabling direct estimation of the probability that a majority of experts would judge a CVS criterion as visible. 

On Endoscapes2023, the proposed approach significantly improves average balanced accuracy by +2.1% compared to the state-of-the-art. We also introduce a precision-critical video-level evaluation protocol reflecting surgical decision-making, reporting recall under a strict precision constraint (precision = 1.0). Under this, the model maintains meaningful recall while avoiding false positives. Modelling annotator disagreement probabilistically provides a principled and clinically aligned alternative to deterministic CVS classification and supports precision-focused evaluation for surgical deployment.

## Key Features

- **Probabilistic modelling of annotator disagreement**: We model the three annotations as Binomial observations of a latent visibility probability, avoiding information loss in majority vote labels.
- **Beta-Binomial ($B^{2}$) loss**: We propose a statistically grounded loss derived from the marginal Beta-Binomial likelihood, enabling learning directly from annotator vote distributions.
- **Evidential uncertainty modelling**: We employ evidential learning to estimate both the latent visibility probability and the strength of evidence supporting it via a Beta-Binomial evidential framework.
- **Precision-critical video-level evaluation**: We introduce an evaluation protocol aligned with clinical requirements, prioritising precision and assessing criterion achievement at video level.

---

## Installation

```bash
git clone https://github.com/franeknowak/BetaBinomialLoss.git
cd BetaBinomialLoss
pip install -r requirements.txt
TBC
```

## Usage

```bash
TBC
```

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{nowak2025bbloss,
  title     = {B$^2$ Loss for Evidential Classification of Critical View of Safety in Laparoscopic Cholecystectomy},
  author    = {Nowak, Franciszek M. and Mazomenos, Evangelos B. and Davidson, Brian and Clarkson, Matthew J.},
  booktitle = {TBC},
  year      = {TBC},
}
```

## License

> _TBC_

## Acknowledgements

Researcher's time was funded in whole, or in part, by the EPSRC-funded Centre for Doctoral Training in Intelligent, Integrated Imaging in Healthcare (i4health) [EP/S021930/1], the NIHR Central London Patient Safety Research Collaboration (CL PSRC) [NIHR204297]and Human-centred Machine Intelligence to optimise Robotic Surgical Training [EP/Z534754/1]. The views expressed are those of the authors and not necessarily those of the NIHR or the Department of Health and Social Care. 
