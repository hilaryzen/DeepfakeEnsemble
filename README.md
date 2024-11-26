# DeepfakeEnsemble
MIT-Itau Group 4 ensemble model for deepfake detection

To setup:

```
git clone https://github.com/hilaryzen/DeepfakeEnsemble.git
cd DeepfakeEnsemble
pip install -r requirements.txt
```

To prepare data:

Load deepfaked test images into `data/itau/df/test` and real test images into `data/itau/real/test`. The `ItauDataset` and `ItauFakeDataset` classes in `data.py` point to these folders. More datasets can be added by creating new classes in `dolos/data.py` and updating `PREDICT_CONFIGS` in `dolos/predict.py`.

To generate predictions:

Run `ensemble.py` with the commands below to generate ensemble model predictions for the real and fake datasets respectively. To run on a different dataset, update the -p flag to another key in `PREDICT_CONFIGS` in `dolos/predict.py`.

```
python ensemble.py -s weak -t setup-b -p itau-test
python ensemble.py -s weak -t setup-b -p itau-fake-test
```

To get ensemble metrics:

Run `evaluate.py` with the command below (after generating predictions with `ensemble.py`) to get accuracy, precision, recall, and f1. The -r flag holds the dataset name for real images, and the -d flag holds the dataset name for fake images. Both can be changed to other keys in `PREDICT_CONFIGS` in `dolos/predict.py`.

```
python evaluate.py -s weak -t setup-b -r itau-test -d itau-fake-test
```