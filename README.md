# go-probe 

This is the code repository for *Understanding Game-Playing Agents with Natural Language Annotations*

## Getting Started

```{bash}
conda create -n go python=3.6
conda activate go
conda install pytorch==1.7.1 cudatoolkit=10.1 -c pytorch

cd ~
git clone https://github.com/AndreHe02/go-probe.git
export PYTHONPATH=$PYTHONPATH:~/go-probe/
git clone https://github.com/maxpumperla/deep_learning_and_the_game_of_go.git
export PYTHONPATH=$PYTHONPATH:~/deep_learning_and_the_game_of_go/code/
```

## Downloading the Dataset
Convert raw sgf files into a list of board-comment pairs
```{bash}
git clone https://github.com/AndreHe02/go.git
cd go/src/
python board2comments.py -d ../data/annotations/ -o b2c/
mv b2c/annotations.pkl ~/go-probe/
```

## Preproccessing
Sort keywords by frequency and pick control words as described in the paper. Generate input representations for the imitation learning model and ELF OpenGo and store them on disk. This will write about 38 GB of data into dataset/
```{bash}
cd ~/go-probe/
python go_probe/datasets/filter_annotations.py
python go_probe/datasets/generate_word_sets.py -f annotations_filtered.pkl -g go_dict.txt
mkdir dataset
python go_probe/datasets/generate_dataset.py -f annotations_filtered.pkl -d dataset/
```

## Running Experiments
By default, this runs probes for both keyword-based features and pattern-based features and stores results in a numpy arrays with shape (# cross validation folds, # probed layers, # features). 
```{bash}
python go_probe/experiments/go_experiment.py -d dataset/ -n 10
python go_probe/experiments/elf_experiment.py -d dataset/ -n 10
```