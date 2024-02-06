

# SASRec+

`SASRec+` augments each sequence into multiple sub sequences in a rolling fashion,
and only the last item is used as the target.

## Usage

Run with full ranking:

    python main.py --config=configs/xxx.yaml --ranking=full

or with sampled-based ranking

    python main.py --config=configs/xxx.yaml --ranking=pool