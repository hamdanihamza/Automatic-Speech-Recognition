# An Automatic Speech Recognition system for Smart Virutal Assistants.

## Requirements : 
- Pandas
- Matplotlib
- Scipy.io's Wavfile
- Python_speech_features' MFCC
- Json
- Sklearn
- Numpy
- Keras 2.3.0
- Tensorflow/Tensorflow-gpu 1.15
- CTCModel : https://github.com/cyprienruffino/CTCModel

## Dataset : Fluent.ai
A dataset made specifically for Speech Recognition purposes. The Fluent Speech Commands dataset contains 30,043 utterances such as "Turn on the lights, Change language, etc..." from 97 speakers of different age, accent, gender, level of fluency. 
URL : https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/
* There are other choices of dataset (Librispeech, TIMIT, etc...)

## Data Preprocessing :
### Processing Audio Signals : 
The audio signal is converted to MFCC features, MFCC is the closest technique that simulates the human hearing process, it returns a matrix of 13 features but varies in term of length of the audio signal, but the neural network expects a fixed size inputs. To fix this we perform a padding on all the elements calculated by the MFCC until they are all of the same size.
### Processing Transcriptions :
- Removing special characters (, . / ? ! ...).
- Tokenization ("Hello World" => ["Hello", "World"]).
- Lowercasing ("HELLO" => hello).
- Converting to Phonemes ("turn" => "t er n")
- Labelling phonemes : By using a dictionary to label each phoneme into a list of integer values. And we’ll use this list to train the model.
- Padding all vectors.<br>

**Example :**
1. Turn Down the Heat !
2. Turn Down the Heat
3. ["Turn", "Down", "The", "Heat"]
4. ["turn", "down", "the", "heat"]
5. ['t', 'er', 'n', '', 'd', 'aw', 'n', '', 'dh', 'ah', '', 'hh', 'iy', 't']
6. [30, 18, 20, 0, 26, 12, 20, 0, 24, 5, 0, 38, 41, 30])

## Model :
The model is a Deep Learning model, made with a GRU Neural Network of 1.6M parameters. 
The architecture of the model is an input layer of 13 features (MFCC) 2 Bidirectional GRU layers of 256 units, a TimeDistributed Layer and a CTC Layer provided by Cyprien RUFFINO (https://github.com/cyprienruffino).
* The model is provided in 'final_results' folder.
* The model was trained on Google Colab's CPU, Golab only give you a limited period of using its GPU, so the model was trained for only 9 hours/15 epochs with a small amount of data. With more data and computational resources the model could improve a lot.

For more details : ha.h.hamdani@gmail.com
* You are free to use the model.
