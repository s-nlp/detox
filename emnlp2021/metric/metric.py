import os
import gc
import tqdm
import torch
import argparse
import numpy as np

from nltk.translate.bleu_score import sentence_bleu
from tqdm.auto import trange

from wieting_similarity.similarity_evaluator import SimilarityEvaluator


from transformers import AutoModelForSequenceClassification, AutoTokenizer, \
    RobertaTokenizer, RobertaForSequenceClassification

from fairseq.models.roberta import RobertaModel
from fairseq.data.data_utils import collate_tokens


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def classify_preds(args, preds, soft=False):
    print('Calculating style of predictions')
    results = []

    model_name = args.classifier_path or 'SkolkovoInstitute/roberta_toxicity_classifier'

    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name)

    for i in tqdm.tqdm(range(0, len(preds), args.batch_size)):
        batch = tokenizer(preds[i:i + args.batch_size], return_tensors='pt', padding=True)
        with torch.inference_mode():
            logits = model(**batch).logits
        if soft:
            result = torch.softmax(logits, -1)[:, 1].cpu().numpy()
        else:
            result = (logits[:, 1] > args.threshold).cpu().numpy()
        results.extend([1 - item for item in result])
    return results


def calc_bleu(inputs, preds):
    bleu_sim = 0
    counter = 0
    print('Calculating BLEU similarity')
    for i in range(len(inputs)):
        if len(inputs[i]) > 3 and len(preds[i]) > 3:
            bleu_sim += sentence_bleu([inputs[i]], preds[i])
            counter += 1
        
    return float(bleu_sim / counter)


def wieting_sim(args, inputs, preds):
    assert len(inputs) == len(preds)
    print('Calculating similarity by Wieting subword-embedding SIM model')

    sim_evaluator = SimilarityEvaluator()
    
    sim_scores = []
    
    for i in tqdm.tqdm(range(0, len(inputs), args.batch_size)):
        sim_scores.extend(
            sim_evaluator.find_similarity(inputs[i:i + args.batch_size], preds[i:i + args.batch_size])
        )
        
    return np.array(sim_scores)


def detokenize(x):
    return x.replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(" ?", "?").replace(" )",")").replace("( ", "(")  # noqa


def do_cola_eval(args, preds, soft=False):
    print('Calculating CoLA acceptability stats')

    path_to_data = os.path.join(args.cola_classifier_path, 'cola-bin')

    cola_roberta = RobertaModel.from_pretrained(
        args.cola_classifier_path, checkpoint_file=args.cola_checkpoint, data_name_or_path=path_to_data
    )
    cola_roberta.eval()
    if torch.cuda.is_available():
        cola_roberta.cuda()
    
    cola_stats = []
    
    for i in tqdm.tqdm(range(0, len(preds), args.batch_size), total=len(preds) // args.batch_size):
        sentences = preds[i:i + args.batch_size]

        # detokenize and BPE encode input
        sentences = [cola_roberta.bpe.encode(detokenize(sent)) for sent in sentences]

        batch = collate_tokens(
            [cola_roberta.task.source_dictionary.encode_line("<s> " + sent + " </s>", append_eos=False)
             for sent in sentences], 
            pad_idx=1
        )

        batch = batch[:, :512]

        with torch.no_grad():
            predictions = cola_roberta.predict('sentence_classification_head', batch.long())
        
        if soft:
            prediction_labels = torch.softmax(predictions, axis=1)[:, 1].cpu().numpy()
        else:
            prediction_labels = predictions.argmax(axis=1).cpu().numpy()
        # label 0 means acceptable. Need to inverse
        cola_stats.extend(list(1 - prediction_labels))
    
    return np.array(cola_stats)


def do_cola_eval_transformers(args, preds, soft=False):
    print('Calculating CoLA acceptability stats')
    path = args.cola_classifier_path

    model = AutoModelForSequenceClassification.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)

    results = []
    bs = args.batch_size
    for i in trange(0, len(preds), bs):
        batch = [detokenize(t) for t in preds[i: i + bs]]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = torch.softmax(model(**inputs).logits, -1)[:, 0].cpu().numpy()
            if soft:
                results.append(out)
            else:
                results.append((out > 0.5).astype(int))
    return np.concatenate(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--inputs", help="path to test sentences", required=True)
    parser.add_argument('-p', "--preds", help="path to predictions of a model", required=True)
    
    parser.add_argument("--classifier_path", default='SkolkovoInstitute/roberta_toxicity_classifier')
    parser.add_argument("--threshold", default=0.8, type=float)

    parser.add_argument("--cola_classifier_path", default='models/cola')
    parser.add_argument("--cola_checkpoint", default='checkpoint_best.pt')
    parser.add_argument("--batch_size", default=32, type=int)

    args = parser.parse_args()

    with open(args.inputs, 'r') as input_file, open(args.preds, 'r') as preds_file:
        inputs = input_file.readlines()
        preds = preds_file.readlines()

    # accuracy of style transfer
    accuracy_by_sent = classify_preds(args, preds)
    accuracy = sum(accuracy_by_sent)/len(preds)
    cleanup()
    
    # similarity
    bleu = calc_bleu(inputs, preds)
    
    similarity_by_sent = wieting_sim(args, inputs, preds)
    avg_sim_by_sent = similarity_by_sent.mean()
    cleanup()
    
    # fluency
    cola_stats = do_cola_eval(args, preds)
    cola_acc = sum(cola_stats) / len(preds)
    cleanup()
    
    # count metrics
    joint = sum(accuracy_by_sent * similarity_by_sent * cola_stats) / len(preds)
    
    # write res to table
    name = args.preds.split('/')[-1]
    print('| Model | ACC | SIM | FL | J | BLEU |\n')
    print('| ----- | --- | --- | -- | - | ---- |\n')
    print(f'{name}|{accuracy:.4f}|{avg_sim_by_sent:.4f}|{cola_acc:.4f}|{joint:.4f}|{bleu:.4f}|\n')
