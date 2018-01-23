seq2seq chatbot implement using PyTorch.

## Requirements
- Python 3.5
- Pytorch 0.3.0

## Corpus
- [Cornell Movie Dialogue](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)

## Usage
### Preprocess
```python
python preprocess.py -a 0
```
This script will create file `dialogue_corpus.txt` in `./data` directory.

### Training Model
```python
python train.py -c config
```
The hyperparameters of model define in configuration file in config folder.
Ex: python train.py -c 256_4_4
In my local environment(GTX1080), training model need about 12 hours.

### Testing
```python
python chatbot.py -c config -e number -f list_question.
```
number is the number of dump_ in folder checkpoint

#### Test Example
```
> hi.
bot: hi .
> nice to meet you .
bot: and this is manray .
> What ' s your name ?
bot: jacob singer .
> how are you ?
bot: fine .
> How old are you ?
bot: thirty five .
> Are you a robot ?
bot: no , i ' m not .
> Are you a human ?
bot: no , i ' m not .
> Where are you from ?
bot: UNK , holiness .
> Where do you live ?
bot: excuse me .
> Where do you come from ?
bot: california . oakland .

```
The chatbot can answer these simple questions, but in most cases it is a silly bot.

pre-train model: https://drive.google.com/drive/folders/1vS_IKIidFoxTa8D6wKNfwuwCkvni13_a?usp=sharing
