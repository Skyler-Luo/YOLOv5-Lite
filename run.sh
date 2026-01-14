nohup python train.py --weights '' --cfg models/yolov5n.yaml --data dataset_seaship/data.yaml \
    --hyp data/hyps/hyp.scratch-low.yaml --cache --name yolov5n_baseline --batch-size 64 --workers 8 --epochs 200 > logs/yolov5n_baseline.out 2>&1 & tail -f logs/yolov5n_baseline.out

nohup python train.py --weights '' --cfg models/yolov5n-light.yaml --data dataset_seaship/data.yaml \
    --hyp data/hyps/hyp.scratch-low.yaml --cache --name yolov5n_light --batch-size 64 --workers 8 --epochs 200 > logs/yolov5n_light.out 2>&1 & tail -f logs/yolov5n_light.out

nohup python compress.py --model yolov5n --dataset VOC --data dataset_seaship/data.yaml --batch 64 --epochs 100 --weights runs/train/yolov5n_light/weights/best.pt \
    --workers 8 --initial_rate 0.06 --initial_thres 6. --topk 0.8 --exp --cache --name yolov5n_light_prune > logs/yolov5n_light_prune.out 2>&1 & tail -f logs/yolov5n_light_prune.out

nohup python compress.py --model yolov5n --dataset VOC --data dataset_seaship/data.yaml --batch 64 --epochs 100 --weights runs/train/yolov5n_light/weights/best.pt \
    --workers 8 --initial_rate 0.2 --initial_thres 10. --topk 0.8 --exp --cache --name yolov5n_light_prune2 > logs/yolov5n_light_prune2.out 2>&1 & tail -f logs/yolov5n_light_prune2.out

nohup python compress.py --model yolov5n --dataset VOC --data dataset_seaship/data.yaml --batch 64 --epochs 100 --weights runs/train/yolov5n_light/weights/best.pt \
    --workers 8 --initial_rate 0.4 --initial_thres 20. --topk 0.8 --exp --cache --name yolov5n_light_prune3 > logs/yolov5n_light_prune3.out 2>&1 & tail -f logs/yolov5n_light_prune3.out

nohup python compress.py --model yolov5n --dataset VOC --data dataset_seaship/data.yaml --batch 64 --epochs 100 --weights runs/train/yolov5n_baseline/weights/best.pt \
    --workers 8 --initial_rate 0.4 --initial_thres 20. --topk 0.8 --exp --cache --name yolov5n_baseline_prune > logs/yolov5n_baseline_prune.out 2>&1 & tail -f logs/yolov5n_baseline_prune.out

python val.py --data dataset_seaship/data.yaml --weights runs/train/yolov5n_baseline/weights/best.pt --task test --name yolov5n_baseline --exist-ok --device 0
python val.py --data dataset_seaship/data.yaml --weights runs/train/yolov5n_baseline_prune/weights/best.pt --task test --name yolov5n_baseline_prune --exist-ok --device 0
python val.py --data dataset_seaship/data.yaml --weights runs/train/yolov5n_light/weights/best.pt --task test --name yolov5n_light --exist-ok --device 0
python val.py --data dataset_seaship/data.yaml --weights runs/train/yolov5n_light_prune/weights/best.pt --task test --name yolov5n_light_prune --exist-ok --device 0
python val.py --data dataset_seaship/data.yaml --weights runs/train/yolov5n_light_prune2/weights/best.pt --task test --name yolov5n_light_prune2 --exist-ok --device 0
python val.py --data dataset_seaship/data.yaml --weights runs/train/yolov5n_light_prune3/weights/best.pt --task test --name yolov5n_light_prune3 --exist-ok --device 0

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple thop torch_pruning==0.2.7