# ETRI_KSB
Source-codes about ETRI KSB framework.

## Goal
- Predict public bicycle demand

## Feature
- Learn random forest model using Tashu 2013-2015 dataset

## Usage
- Model using R.F. based on tashu history

### Train in local
```bash
./random_forest_train_local/random_forest_init.py --input ./data/history --model ./model/ --checkpoint ./checkpoint/local
```

### Check models
```bash
$ saved_model_cli show --dir ./model/1/ --all

MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['predict']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['x'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 8)
        name: Placeholder:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['y'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, -1)
        name: probabilities:0
  Method name is: tensorflow/serving/predict
```

### Serving using docker
```bash
docker run --rm -p 8501:8501 -v MODELPATH:MODELPATHINDOCKER -e MODEL_NAME=tashu tensorflow/serving
```

### Serving test (local)
```bash
curl -i -d '{"signature_name": "predict", "instances": [[1,0.0,0.0,0.0,0.0,0.0,0.0,0.0]]}' http://IPADDRESS:PORT/v1/models/default:predict
```

## Code by
- LuHa(munhyunsu@gmail.com)
