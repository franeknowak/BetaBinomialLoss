# B² Loss for Evidential Classification of Critical View of Safety in Laparoscopic Cholecystectomy

Official repository for the paper **"B² Loss for Evidential Classification of Critical View of Safety in Laparoscopic Cholecystectomy."**

> Published at [MICCAI 2026](https://doi.org/10.1007/978-3-032-38233-7_3), LNCS vol. 16892

---

## Authors

- **Franciszek M. Nowak**¹ ([ORCID](https://orcid.org/0009-0006-4969-0139))
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

Requires Python 3.10 and an NVIDIA GPU. The pinned `requirements.txt` installs
the CUDA 12.1 build of PyTorch.

```bash
git clone https://github.com/franeknowak/BetaBinomialLoss.git
cd BetaBinomialLoss
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For a different CUDA version, install `torch` and `torchvision` first following
the [official selector](https://pytorch.org/get-started/locally/), then run
`pip install -r requirements.txt` to pick up the rest.

### Dataset

Nothing to do: on the first run the training script checks for the dataset and,
if it is missing, downloads Endoscapes2023 from the
[official CAMMA release](https://s3.unistra.fr/camma_public/datasets/endoscapes/endoscapes.zip),
unpacks it into `./dataset/` and deletes the archive. The download is about
5.9GB and happens only once; every later run detects the dataset and skips it.

To fetch the data ahead of time:

```bash
python -m scripts.download_dataset
```

If you already have Endoscapes2023 elsewhere, point `DATASET_DIR` in your config
at it (or symlink it to `./dataset/endoscapes`) and no download takes place.

## Usage

Each experiment is defined by a single YAML config:

```bash
python main.py --config_path ./config/examples/b2_dinov3_gatedpool.yaml
```

Ready-to-run configs are in [`config/examples/`](config/examples), covering the
B² Model from the paper, the SwinV2 backbone, the BCE baseline and the
single-frame variant. See [`docs/configuration.md`](docs/configuration.md) for
what to change to run a different experiment.

Each run writes `results/<EXPERIMENT_NAME><SEED>_results.json` and saves the
best epoch's weights to `./weights`. Running `python results/generate_summary.py`
collects every results file into `results/experiment_summary.md`.

## Citation

If you use this work, please cite:

> Nowak, F.M., Mazomenos, E.B., Davidson, B., Clarkson, M.J. (2027). B² Loss for
> Evidential Classification of Critical View of Safety in Laparoscopic
> Cholecystectomy. In: Yang, G., et al. Medical Image Computing and Computer
> Assisted Intervention – MICCAI 2026. Lecture Notes in Computer Science,
> vol 16892. Springer, Cham. https://doi.org/10.1007/978-3-032-38233-7_3

```bibtex
@inproceedings{nowak2027b2loss,
  title     = {B$^2$ Loss for Evidential Classification of Critical View of Safety in Laparoscopic Cholecystectomy},
  author    = {Nowak, Franciszek M. and Mazomenos, Evangelos B. and Davidson, Brian and Clarkson, Matthew J.},
  editor    = {Yang, Guang-Zhong and others},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026},
  series    = {Lecture Notes in Computer Science},
  volume    = {16892},
  publisher = {Springer},
  address   = {Cham},
  year      = {2027},
  doi       = {10.1007/978-3-032-38233-7_3},
  isbn      = {978-3-032-38233-7},
}
```

## License

The code in this repository is released under the [MIT License](LICENSE): you
are free to use, modify and redistribute it, including commercially, provided
the copyright notice is retained. If you use it in academic work, please also
cite the paper (see [Citation](#citation)).

The published paper itself is **not** open access: © 2027 The Author(s), under
exclusive license to Springer Nature Switzerland AG. Cite the DOI above rather
than redistributing the chapter PDF.

## Acknowledgements

Researcher's time was funded in whole, or in part, by the EPSRC-funded Centre for Doctoral Training in Intelligent, Integrated Imaging in Healthcare (i4health) [EP/S021930/1], the NIHR Central London Patient Safety Research Collaboration (CL PSRC) [NIHR204297]and Human-centred Machine Intelligence to optimise Robotic Surgical Training [EP/Z534754/1]. The views expressed are those of the authors and not necessarily those of the NIHR or the Department of Health and Social Care. 
