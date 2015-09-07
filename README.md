# GLAD: Groningen Lightweight Authorship Detection 

## Example uses of `glad-main.py` ##
In the following examples the following placeholders are used to have shorter commands (this assumes the PAN2014/15 datasets are downloaded):

`$trainingDataset`:   `path/to/data/pan15-authorship-verification-training-dataset-english-2015-03-02`

`$inputDataset`:   `path/to/data/pan14-author-verification-test-corpus2-english-both-2014-04-22`

`$modelDir`:   `path/to/models/example_model`

### Train a model ###

`python3 glad-main.py --training $trainingDataset -i $inputDataset --save_model $modelDir`
>  `--training` Use this data set to train a model
> 
>  `--save_model` Store the model to this directory. If the directory already exists, it will write anyways (and update).


### Make predictions using an existing model ###

`python3 glad-main.py -i $inputDataset -m $modelDir`
>  `-m` Load the model; alternative flag: `--model`
> 
> `-i` make predictions on the `$inputDataset`; alternative flags: `--test` `--input` 

### Train and test a model ###
`python3 glad-main.py --training $trainingDataset --test $inputDataset`
>  `--training` Use this data set to train a model
>
> `--test` make predictions on the `$inputDataset`; alternative flags: `-i` `--input`

`python3 glad-main.py --training $trainingDataset --split`
>  `--training` Use this data set to train a model
>
> `--split` split on the training data. Default split: 70%. Define your own split like this: `--split 0.5`

### Writing the predictions ###

`python3 glad-main.py -i $inputDataset -m $modelDir -o Out`
> `-o` store the *answers.txt* file to the directory `Out`; alternative flag: `--out`

`python3 glad-main.py -i $inputDataset -m $modelDir -a path/to/answers.file`
> `-a` store the predictions to a file; alternative flag: `--answers`

## Requirements ##

- Python 3.x
- NLTK 
- NumPy
- scikit-learn
- (liac-arff)
