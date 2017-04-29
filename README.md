
# Classifying Fisher's Iris data set with python

This is my first baby machine learning project so take it easy. Also, it's pretty much 100% copied from [this](http://machinelearningmastery.com/machine-learning-in-python-step-by-step/) tutorial put together by the main man Jason Brownlee.

It was very handholdy, but I have literally zero idea about what I'm doing so I was kind of relieved about that.

## What is it?
Good question. Basically, some guy called [Ronald Fisher](https://en.wikipedia.org/wiki/Iris_flower_data_set) took 50 measurements for 3 different varieties of Iris flowers back in 1936. Little did he know that this would become the 'Hello-World' dataset of machine learning nearly a century later.
The idea of this programme is to compare the effectiveness of different classification models on the Iris data set. After finding the best classification method, it's then possible to guess what type of Iris a flower is given its petal measurements! 

![alt text](https://maxpull-gdvuch3veo.netdna-ssl.com/wp-content/uploads/2012/05/Siberian-iris.jpg "Sick Flower Pic")


## How to get this steaming hot flowery mess of a programme running
### 1) Clone this repo
```
  cd /wherever/you/want/to/clone/this/repo
  git clone https://github.com/whatsrupp/iris-classification
```
### 2) Get pip 'n' python 
I'm not going to go into this in too much detail as I'd rather be coding. A quick google 'installing pip' and installing python' should do the job.
### 3) Install Packages
This requires a number of scipi packages
- scipy
- numpy
- matplotlib
- pandas
- sklearn

```
pip install --user numpy scipy matplotlib pandas sklearn
```
any problems with the install and I would head over [here](https://www.scipy.org/install.html). Alternatively, I'd just leave it, as this isn't much to look at anyway.

### 4) Launch

Rather hysterically, the interesting output from this project is a tiny table of data. It's not too flasy but it's about what's under the hood right?

#### a) Go to the directory root
```
cd location/of/iris-classification
```
#### b) Run the programme
```
python
execfile('petal_classifier.py')
```
and then follow further prompting - (that's right, brace yourself for a horrendous command line interface)



