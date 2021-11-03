
export CUDA_VISIBLE_DEVICES=3

cd /home/dale/dialogue-censor/metric


python metric.py \
	--inputs ../data/yelp/sentiment.test.0 \
	--preds ../data/yelp/sentiment.test.0  \
	--classifier_path ../classification/yelp/roberta_for_sentiment_classification_v2/model_out \
	--labels_path ../classification/yelp/roberta_for_sentiment_classification_v2 \
	--task_name yelp  --toxification
python metric.py \
	--inputs ../data/yelp/sentiment.test.1 \
	--preds ../data/yelp/sentiment.test.1 \
	--classifier_path ../classification/yelp/roberta_for_sentiment_classification_v2/model_out \
	--labels_path ../classification/yelp/roberta_for_sentiment_classification_v2 \
	--task_name yelp

python metric.py \
	--inputs ../data/yelp/sentiment.test.0 \
	--preds ../data/yelp/mask_infill.0 \
	--classifier_path ../classification/yelp/roberta_for_sentiment_classification_v2/model_out \
	--labels_path ../classification/yelp/roberta_for_sentiment_classification_v2 \
	--task_name yelp  --toxification
python metric.py \
	--inputs ../data/yelp/sentiment.test.1 \
	--preds ../data/yelp/mask_infill.1 \
	--classifier_path ../classification/yelp/roberta_for_sentiment_classification_v2/model_out \
	--labels_path ../classification/yelp/roberta_for_sentiment_classification_v2 \
	--task_name yelp

python metric.py \
	--inputs ../data/yelp/sentiment.test.0 \
	--preds /home/dale/projects/DualRL/outputs/yelp/DualRL/test.0.tsf  \
	--classifier_path ../classification/yelp/roberta_for_sentiment_classification_v2/model_out \
	--labels_path ../classification/yelp/roberta_for_sentiment_classification_v2 \
	--task_name yelp  --toxification
python metric.py \
	--inputs ../data/yelp/sentiment.test.1 \
	--preds /home/dale/projects/DualRL/outputs/yelp/DualRL/test.1.tsf \
	--classifier_path ../classification/yelp/roberta_for_sentiment_classification_v2/model_out \
	--labels_path ../classification/yelp/roberta_for_sentiment_classification_v2 \
	--task_name yelp
	
	
python metric.py \
	--inputs ../data/yelp/sentiment.test.0 \
	--preds ../data/yelp/human.0  \
	--classifier_path ../classification/yelp/roberta_for_sentiment_classification_v2/model_out \
	--labels_path ../classification/yelp/roberta_for_sentiment_classification_v2 \
	--task_name yelp  --toxification
python metric.py \
	--inputs ../data/yelp/sentiment.test.1 \
	--preds ../data/yelp/human.1 \
	--classifier_path ../classification/yelp/roberta_for_sentiment_classification_v2/model_out \
	--labels_path ../classification/yelp/roberta_for_sentiment_classification_v2 \
	--task_name yelp




python metric.py \
	--inputs ../data/yelp/sentiment.test.0 \
	--preds /home/dale/projects/DualRL/outputs/yelp/UnsuperMT_Zhang/test.0.tsf  \
	--classifier_path ../classification/yelp/roberta_for_sentiment_classification_v2/model_out \
	--labels_path ../classification/yelp/roberta_for_sentiment_classification_v2 \
	--task_name yelp  --toxification
python metric.py \
	--inputs ../data/yelp/sentiment.test.1 \
	--preds /home/dale/projects/DualRL/outputs/yelp/UnsuperMT_Zhang/test.1.tsf \
	--classifier_path ../classification/yelp/roberta_for_sentiment_classification_v2/model_out \
	--labels_path ../classification/yelp/roberta_for_sentiment_classification_v2 \
	--task_name yelp


python metric.py \
	--inputs ../data/yelp/sentiment.test.0 \
	--preds /home/dale/projects/DualRL/outputs/yelp/TemplateBase_Li/test.0.tsf  \
	--classifier_path ../classification/yelp/roberta_for_sentiment_classification_v2/model_out \
	--labels_path ../classification/yelp/roberta_for_sentiment_classification_v2 \
	--task_name yelp  --toxification
python metric.py \
	--inputs ../data/yelp/sentiment.test.1 \
	--preds /home/dale/projects/DualRL/outputs/yelp/TemplateBase_Li/test.1.tsf \
	--classifier_path ../classification/yelp/roberta_for_sentiment_classification_v2/model_out \
	--labels_path ../classification/yelp/roberta_for_sentiment_classification_v2 \
	--task_name yelp


python metric.py \
	--inputs ../data/yelp/sentiment.test.0 \
	--preds /home/dale/projects/DualRL/outputs/yelp/RetrieveOnly_Li/test.0.tsf  \
	--classifier_path ../classification/yelp/roberta_for_sentiment_classification_v2/model_out \
	--labels_path ../classification/yelp/roberta_for_sentiment_classification_v2 \
	--task_name yelp  --toxification
python metric.py \
	--inputs ../data/yelp/sentiment.test.1 \
	--preds /home/dale/projects/DualRL/outputs/yelp/RetrieveOnly_Li/test.1.tsf \
	--classifier_path ../classification/yelp/roberta_for_sentiment_classification_v2/model_out \
	--labels_path ../classification/yelp/roberta_for_sentiment_classification_v2 \
	--task_name yelp
	

python metric.py \
	--inputs ../data/yelp/sentiment.test.0 \
	--preds ../data/yelp/sst_75_07.0  \
	--classifier_path ../classification/yelp/roberta_for_sentiment_classification_v2/model_out \
	--labels_path ../classification/yelp/roberta_for_sentiment_classification_v2 \
	--task_name yelp  --toxification
python metric.py \
	--inputs ../data/yelp/sentiment.test.1 \
	--preds ../data/yelp/sst_75_07.1  \
	--classifier_path ../classification/yelp/roberta_for_sentiment_classification_v2/model_out \
	--labels_path ../classification/yelp/roberta_for_sentiment_classification_v2 \
	--task_name yelp


python metric.py \
	--inputs ../data/yelp/sentiment.test.0 \
	--preds ../data/yelp/sst_0_07.0  \
	--classifier_path ../classification/yelp/roberta_for_sentiment_classification_v2/model_out \
	--labels_path ../classification/yelp/roberta_for_sentiment_classification_v2 \
	--task_name yelp  --toxification
python metric.py \
	--inputs ../data/yelp/sentiment.test.1 \
	--preds ../data/yelp/sst_0_07.1  \
	--classifier_path ../classification/yelp/roberta_for_sentiment_classification_v2/model_out \
	--labels_path ../classification/yelp/roberta_for_sentiment_classification_v2 \
	--task_name yelp


python metric.py \
	--inputs ../data/yelp/sentiment.test.0 \
	--preds ../data/yelp/results/gedi_coef4_batch10_rerank.0.txt  \
	--classifier_path ../classification/yelp/roberta_for_sentiment_classification_v2/model_out \
	--labels_path ../classification/yelp/roberta_for_sentiment_classification_v2 \
	--task_name yelp  --toxification
python metric.py \
	--inputs ../data/yelp/sentiment.test.1 \
	--preds ../data/yelp/results/gedi_coef4_batch10_rerank.1.txt  \
	--classifier_path ../classification/yelp/roberta_for_sentiment_classification_v2/model_out \
	--labels_path ../classification/yelp/roberta_for_sentiment_classification_v2 \
	--task_name yelp


python metric.py \
	--inputs ../data/yelp/sentiment.test.0 \
	--preds ../data/yelp/results/condbert_tox1.5_w2_sim20.0.txt  \
	--classifier_path ../classification/yelp/roberta_for_sentiment_classification_v2/model_out \
	--labels_path ../classification/yelp/roberta_for_sentiment_classification_v2 \
	--task_name yelp  --toxification
python metric.py \
	--inputs ../data/yelp/sentiment.test.1 \
	--preds ../data/yelp/results/condbert_tox1.5_w2_sim20.1.txt  \
	--classifier_path ../classification/yelp/roberta_for_sentiment_classification_v2/model_out \
	--labels_path ../classification/yelp/roberta_for_sentiment_classification_v2 \
	--task_name yelp



python metric.py \
	--inputs ../data/yelp/sentiment.test.0 \
	--preds ../data/yelp/results/condbert_tox3_w2_sim20.0.txt  \
	--classifier_path ../classification/yelp/roberta_for_sentiment_classification_v2/model_out \
	--labels_path ../classification/yelp/roberta_for_sentiment_classification_v2 \
	--task_name yelp  --toxification
python metric.py \
	--inputs ../data/yelp/sentiment.test.1 \
	--preds ../data/yelp/results/condbert_tox3_w2_sim20.1.txt  \
	--classifier_path ../classification/yelp/roberta_for_sentiment_classification_v2/model_out \
	--labels_path ../classification/yelp/roberta_for_sentiment_classification_v2 \
	--task_name yelp

