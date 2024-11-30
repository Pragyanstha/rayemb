rayemb train arbitrary-landmark deepfluoro \
--max_epochs 100 \
--image_size 224 \
--num_samples 40 \
--sampling_distance 1 \
--emb_dim 32 \
--lr 1e-4 \
--batch_size 8 \
--num_workers 4 \
--temperature 1e-4 \
--num_templates 4 \
--data_dir data/deepfluoro_synthetic \
--template_dir data/deepfluoro_templates