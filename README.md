This project uses <a href="https://www.kaggle.com/competitions/forest-cover-type-prediction">this dataset</a> 

As for an environment manager I used conda,  feel free to download conva_env.yaml

to save a model use 
```sh
python saving.py save
```
save supports n_estimators number as the first arg and max_depth as the second
for example, you can specify it as 
```sh
python saving.py save(340, 53)
```
right from the root folder
This is achieved using fire module

Been tracking progress via comet.ml:
![изображение](https://user-images.githubusercontent.com/96877411/167421635-f5e9fdcc-b0bc-4618-acf4-a239335b18ad.png)
![изображение](https://user-images.githubusercontent.com/96877411/167421854-39c94743-a138-4539-a448-9dee5c5e7ebf.png)

Result on kaggle was like
![изображение](https://user-images.githubusercontent.com/96877411/167413229-8ee51df6-32c9-492d-97d6-45cb79315c57.png)