import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/yolo11.yaml')
    #model.load('best.pt') # loading pretrain weights
    model.train(data='data/Rail.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=16,
                workers=0,
                device='0',
                optimizer='SGD', # using SGD
                # resume='', # last.pt path
                # amp=False # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )