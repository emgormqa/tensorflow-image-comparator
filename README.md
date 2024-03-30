# Tenserflow image classifyer 

With this code you can create model wicth can compare one thing from another.

I was able to create a model that can tell a human from a cat using 20 images and 50 training cycles.


First you need put the teaching pictures in folders 1 and 2


then run data.py

``` python3 data.py ```

then run model.py, and at the end of the training, model will be saved to the x_classifier.h5

``` python3 model.py ```

Now you can put test.jpg in the root of the project and run main.py, then you will get the result of the comparison

``` python3 main.py ```
