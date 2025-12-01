# Downloaded Datasets

This directory stores datasets for the project. Large binaries are ignored from git; see `.gitignore`.

## Dataset: AIWolf 2019 5-Player Game Logs
- **Source**: http://aiwolf.org/archive/2019final-log05.tar.gz  
- **Size**: 6.5 MB tar.gz containing 10,000 gzipped game logs  
- **Format**: Plain-text log files (`*.log.gz`) with structured turn-by-turn events  
- **Task Fit**: Social deduction (Werewolf) dialogue + action traces for modeling hidden roles and belief updates  
- **Splits**: Unsplit competition logs
- **License**: Provided for research use by AIWolf competition (no explicit license in archive)

### Download Instructions
- Already downloaded to `datasets/aiwolf_logs/2019final-log05.tar.gz`.
- To re-download:
```bash
mkdir -p datasets/aiwolf_logs
curl -L 'http://aiwolf.org/archive/2019final-log05.tar.gz' -o datasets/aiwolf_logs/2019final-log05.tar.gz
```

### Inspecting Logs
```bash
# Count number of games
tar -tzf datasets/aiwolf_logs/2019final-log05.tar.gz | grep -c '\\.log\\.gz$'

# Peek at one log
tar -xzf datasets/aiwolf_logs/2019final-log05.tar.gz 20190813-064848-517/game/000.log.gz -O | gunzip | head
```

### Sample Data
A snippet from game `000` is stored in `datasets/aiwolf_logs/samples/000_head.txt` (first 40 lines) for quick schema reference.

### Notes
- Logs include system metadata and agent utterances; parsing requires decompressing each `*.log.gz`.
- Suitable for training/evaluating role inference, belief tracking, and deception strategies in Werewolf agents.
