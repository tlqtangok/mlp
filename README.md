# mlp
a simple mlp : from training to cpp inference. 


## to train
```bash
python train_mlp.py 
```

## to build
```bash
mkdir build
cd build
cmake ..
cmake --build . 
```


## console out

### train process

```text
Starting training...
Epoch [10/50] finished.
Epoch [20/50] finished.
Epoch [30/50] finished.
Epoch [40/50] finished.
Epoch [50/50] finished.
Training finished.
PyTorch model saved to mlp_classifier.pth
Model exported to ONNX format at mlp_classifier.onnx
Scaler parameters saved to scaler_params.json
```


### inference process

```text
root@838e9d354ef9:/home/jd/t/git/mlp# ./build/mlp_inference
Model loaded successfully
  Input size: 10
  Output size: 3
Loaded scaler parameters:
  Mean size: 10
  Scale size: 10

Prediction result:
  Predicted class: 0
  Probabilities: Class 0: 0.697764, Class 1: 0.0985958, Class 2: 0.20364
  Raw logits: Class 0: 1.00371, Class 1: -0.953139, Class 2: -0.227815
  Probability sum: 1 (should be ~1.0)

```

