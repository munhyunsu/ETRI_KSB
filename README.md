# ETRI_KSB
Source-codes about ETRI KSB framework.

## Goal
- Predict public bicycle demand

## Feature
- Learn random forest model using Tashu 2013-2015 dataset

## Usage
- Model using R.F. based on tashu history


### Train
```bash
./random_forest_train_local/random_forest_init.py --input ./data/history --model ./model/ --checkpoint ./checkpoint/local
```

### Serving test(local)
```bash
curl -i -d '{"signature_name": "predict", "instances": [[1,0.0,0.0,0.0,0.0,0.0,0.0,0.0]]}' http://IPADDRESS:PORT/v1/models/default:predict
```

## Code by
- LuHa(munhyunsu@gmail.com)
