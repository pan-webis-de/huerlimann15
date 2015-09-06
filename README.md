# GLAD: Groningen Lightweight Authorship Detection 

## Example uses of `dataManipulation.py` ##
In the following examples the following placeholders are used to have shorter commands:

`$trainingDataset`:   `Data/SettingA/pan15-authorship-verification-training-dataset-2015-03-02/pan15-authorship-verification-training-dataset-english-2015-03-02`

`$inputDataset`:   `Data/Setting0/pan14-author-verification-test-corpus2-english-both-2014-04-22`

`$modelDir`:   `Models/example_model`

### Train a model ###

`python3 Scripts/dataManipulation.py --training $trainingDataset -i $inputDataset --save_model $modelDir`
>  `--training` Use this data set to train a model
> 
>  `--save_model` Store the model to this directory. If the directory already exists, it will write anyways (and update).


### Make predictions using an existing model ###

`python3 Scripts/dataManipulation.py -i $inputDataset -m $modelDir`
>  `-m` Load the model; alternative flag: `--model`
> 
> `-i` make predictions on the `$inputDataset`; alternative flags: `--test` `--input` 

### Train and test a model ###
`python3 Scripts/dataManipulation.py --training $trainingDataset --test $inputDataset`
>  `--training` Use this data set to train a model
>
> `--test` make predictions on the `$inputDataset`; alternative flags: `-i` `--input`

`python3 Scripts/dataManipulation.py --training $trainingDataset --split`
>  `--training` Use this data set to train a model
>
> `--split` split on the training data. Default split: 70%. Define your own split like this: `--split 0.5`

### Writing the predictions ###

`python3 Scripts/dataManipulation.py -i $inputDataset -m $modelDir -o Out`
> `-o` store the *answers.txt* file to the directory `Out`; alternative flag: `--out`

`python3 Scripts/dataManipulation.py -i $inputDataset -m $modelDir -a path/to/answers.file`
> `-a` store the predictions to a file; alternative flag: `--answers`