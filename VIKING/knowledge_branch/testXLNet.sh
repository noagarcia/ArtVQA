CUDA_VISIBLE_DEVICES=1 python ./XLNet/run_squad.py  \
--model_type xlnet \
--model_name_or_path ./Models/XLNet  \
--do_eval  \
--do_lower_case  \
--train_file ./Cache/xlnet_train.json  \
--predict_file ./Cache/xlnet_pipeline.json \
--per_gpu_eval_batch_size 1  \
--output_dir ./Models/XLNet

