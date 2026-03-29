# 0118_vincent Experiment

Run the 0118_vincent workflow generation pipeline on `n8n_templates/testing_data`,
then evaluate node accuracy and connection accuracy using the existing evaluation
framework.

## Usage

```bash
python experiments_vincent_0118/run_experiment.py
```

Optional flags:

```bash
python experiments_vincent_0118/run_experiment.py --resume --limit 20
```

## Outputs

Results are saved under:

- `outputs/vincent_0118_experiment/llm_generated_workflows/`
- `outputs/vincent_0118_experiment/evaluation_results/`
