#max_len xian:143 chengdu:322   3392/4315
python -u run_ETA.py \
  --data_path ./xian_dataset/ \
  --data_name xian \
  --model PathModel \
  --road_size 3392 \
  --embedding_dim 768 \
  --max_len 143 \
  --out_dim 768 \
  --batch_size 8 \
  --train_epochs 40