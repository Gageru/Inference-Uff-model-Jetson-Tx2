# Inference-Uff-model-Jetson-Tx2
API provides serialization / deserialization and inference of uff models using Tensorrt (c++) methods on JetsonTx2

### Foreword
If you are looking to optimize Cnn performance through Tensorrt (C ++ API) on JetsonTx2, then this tutorial may help you.

### My versions:
- JetPack 4.4
- CUDA V10.2.89
- CUDNN 8.0.0
- OpenCV 4.4.0
- Tensorrt 7.1.3

### RUN
1. Place the supplied folders in ```/usr/src/tensorrt/samples```.

2. Replace Makefile and Makefile.config respectively.

3. Run ```/inferUffModel$ make```. 

4. You must first have a .uff network model. Use the converter
python3 /usr/lib/python3.6/dist-packages/uff/bin/convert_to_uff.py \
  /frozen_inference_graph.pb -O NMS \
  -p /config.py \
  -o /frozen_inference_graph.uff
```
5. Use jetsonStend similarly for caffe (ssd) models
