# VulSCP

VulSCP is a C/C++ function-level vulnerability detection framework based on semantically weighted graph representation, sequential convolution, and a parallel attention mechanism.

## Dataset

VulSCP is evaluated on C/C++ function-level vulnerability detection. The main dataset is constructed from three public sources:

- National Vulnerability Database (NVD): [https://nvd.nist.gov](https://nvd.nist.gov)
- Software Assurance Reference Dataset (SARD): [https://samate.nist.gov/SRD/index.php](https://samate.nist.gov/SRD/index.php)
- BigVul / MSR 2020 Code Vulnerability Dataset: [https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset](https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset)

The main dataset contains:

- 1,384 vulnerable C/C++ functions collected from NVD
- 12,303 vulnerable C/C++ functions collected from SARD
- 26,970 non-vulnerable C/C++ functions sampled from the BigVul corpus

Therefore, the full main dataset contains 40,657 function samples in total, including 13,687 vulnerable samples and 26,970 non-vulnerable samples.

For cross-dataset generalization experiments, we additionally evaluate on the ReVeal benchmark dataset:

- ReVeal: [https://github.com/VulDetProject/ReVeal](https://github.com/VulDetProject/ReVeal)

## Environment

Install the required packages with:

```powershell
pip install -r requirements.txt
```

## Run

Run the main experiment with:

```powershell
python VulSCP.py --epochs 200 --batch-size 32 --runs 5 --seed-base 42 --data-path ./data/pkl/
```

## Output

Experiment results are saved under:

```powershell
./data/results/
```

The script writes per-run results together with aggregated mean and standard deviation to `experiment_summary.json`.
