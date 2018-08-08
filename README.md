# PCNN with word-level attention

## Requirements

- numpy

- tensorflow

- keras

## Usage

### Input file

Input file must be placed at `data/input.cln`. The input file can contain multiple lines, one for each record. Each line must be follow this pattern

    e1start e1end e2start e2end word-1 word-2 ... word-n

- `e1start`, `e1end`, `e2start`, `e2end` are starting index and ending index of two entities in the sentence

- From `word-1` to `word-n` are the preprocessed sentence (tokenized, lowercase)

Example one line of input file:

    0 0 12 12 văn_hanh ( tên thật nguyễn_văn_hanh ) ( sinh 1927 ) là một ca_sĩ việt_nam

### Run

Run `test.py`:

    python test.py
    

### Output file

For each record in input file, the model predict one relation. Name of these relations is logged into `data/output.cln` line by line. 

Example one line of output file:

    Nghề-Nghiệp

List of relation provided at `origin_data/relations.txt` 

