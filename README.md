
# Репозиторий из дипломной работы по теме 

# Генерация музыкального сопровождения для видео на основе архитектуры Transformer

Данный репозиторий представляет из себя копию части рабочего репозитория стартапа, под руководством которого писался диплом. 



# Glinka Transformer v0.2

## About
This is an imlementation of the MusicTransformer (Huang et al., 2018) for Pytorch with several improvements:
1. We added some global and local features which improved quality and customizability of generated music and made a pipeline to add new features easily.
2. We implemented two level network to add deviation from some of additional features to the loss function.
3. We implemented a new algorithm based on Gradient Boosting Decision Trees to predict harmony. It will be added to the additional features soon.
4. We made a new approach to generate music which corresponds to a given video.
5. We implemented RankGen-like architecture to improve generation using beam search.

## Currently supported global additional features:
1. Genre
2. Author

## Currently supported local additional features:
1. Nnotes (number of currently active notes)
2. Timings (proportion of the composition up to the present moment)
3. Harmony (root note and magor/minor). Will be implemented soon

## Training and testing
Training and testing are described in https://github.com/glinkamusic/music-transformer/blob/master/Instruction.md


