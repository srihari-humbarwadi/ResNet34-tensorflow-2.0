## Training and dataset info
 - Download dataset [Imagenette dataset](https://s3.amazonaws.com/fast-ai-imageclas/imagenette.tgz)
 - More details about the dataset [https://github.com/fastai/imagenette](https://github.com/fastai/imagenette)
 - batch size 32 per worker, 3 workers
 
## Performance metrics
```
              precision    recall  f1-score   support

           0       0.96      0.94      0.95        50
           1       1.00      0.96      0.98        50
           2       0.94      0.94      0.94        50
           3       0.98      0.92      0.95        50
           4       0.82      0.94      0.88        50
           5       0.79      0.88      0.83        50
           6       0.92      0.96      0.94        50
           7       0.96      0.90      0.93        50
           8       0.98      0.86      0.91        50
           9       0.94      0.94      0.94        50

   macro avg       0.93      0.92      0.93       500
weighted avg       0.93      0.92      0.93       500
```
