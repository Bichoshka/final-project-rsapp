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
![изображение](https://user-images.githubusercontent.com/96877411/167432521-97c9a72d-80bd-4473-964d-6f8a7d9099e4.png)
as for metrics:
multiclass accuracy looks like a good choice, f1 score almost completely follows it as expected, and got somewhat huge AUC number which seems gucci
 flip flap
 let's stick to accuracy from now on
![изображение](https://user-images.githubusercontent.com/96877411/167421854-39c94743-a138-4539-a448-9dee5c5e7ebf.png)

Result on kaggle was like
![изображение](https://user-images.githubusercontent.com/96877411/167413229-8ee51df6-32c9-492d-97d6-45cb79315c57.png)
