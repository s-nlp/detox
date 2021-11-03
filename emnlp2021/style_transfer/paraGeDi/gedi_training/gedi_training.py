# adapted from https://github.com/salesforce/GeDi
import os
import random
import argparse
import logging
logger = logging.getLogger(__name__)

import wandb

import torch
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader

import numpy as np
from sklearn.metrics import f1_score

from modeling_gpt2 import GPT2LMHeadModel
from transformers import AutoModel, AutoTokenizer, GPT2Config, AdamW, get_linear_schedule_with_warmup


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def adjust_embeddings(model, old_tokenizer, new_tokenizer, new_num_embeddings=None): 
    new_tok_vocab = new_tokenizer.get_vocab()
    old_tok_vocab = old_tokenizer.get_vocab()
    new2old = {}
    
    for token, new_idx in new_tok_vocab.items():
        token = token.replace('▁', 'Ġ')
        if token == new_tokenizer.eos_token:
            new2old[new_idx] = old_tokenizer.eos_token_id
        # elif token == new_tokenizer.bos_token:
            # old_token_idx = old_tokenizer.eos_token_id
        elif token == new_tokenizer.unk_token:
            new2old[new_idx] = old_tokenizer.unk_token_id
        elif token == new_tokenizer.pad_token:
            new2old[new_idx] = old_tokenizer.pad_token_id
        elif token in old_tok_vocab:
            new2old[new_idx] = old_tok_vocab[token]
        else:
            new2old[new_idx] = old_tokenizer.encode(token.replace('Ġ', ' '))
            
    # change embeddings
    if new_num_embeddings is None:
        new_num_embeddings = len(new_tok_vocab)
    
    old_embedding = model.get_input_embeddings().weight
    emb_dim = old_embedding.shape[1]
    
    new_embedding = torch.zeros([new_num_embeddings, emb_dim])
    
    for k, v in new2old.items():
        if isinstance(v, int):
            new_embedding[k] = old_embedding[v]
        elif isinstance(v, list):
            new_embedding[k] = old_embedding[v].mean(axis=0)
        else:
            print(k, v)
            raise ValueError()
        
    # apply changes
    param = torch.nn.Parameter(new_embedding)
    model.get_input_embeddings().weight = param
    model.get_input_embeddings().num_embeddings = param.shape[0]
    
    model.get_output_embeddings().weight = param
    model.get_output_embeddings().out_features = param.shape[0]
    
    # change config
    model.config.vocab_size = param.shape[0]
    model.config.pad_token_id = new_tokenizer.pad_token_id
    model.config.eos_token_id = new_tokenizer.eos_token_id
    model.config.unk_token_id = new_tokenizer.unk_token_id
    # model.config.bos_token_id = new_tokenizer.bos_token_id
    
    return model


def prepare_dataset(args, tokenizer, mode='train'):
    res_file_path = os.path.join(args.cache_dir, f"{mode}_features_{args.max_seq_length}")
    if os.path.exists(res_file_path) and not args.overwrite_cache_dir:
        logger.info('Features already exist, loading...')
        features = torch.load(res_file_path)
    else:
        logger.info('Creating features...')
        texts = []
        labels = []
        for label in [0, 1]:
            if args.yelp:
                file_name = os.path.join(args.data_dir, f"sentiment.{mode}.{label}")
            else:
                text_label = 'toxic' if label == 0 else 'normal'
                file_name = os.path.join(args.data_dir, f"{mode}_{text_label}")
            with open(file_name, 'r') as f:
                for line in f.readlines():
                    texts.append(line.strip())
                    labels.append(label)
        
        features = tokenizer(texts, truncation=True, padding=True, max_length=args.max_seq_length)
        features['labels'] = labels
        
        logger.info(f"Saving features into {res_file_path}")
        torch.save(features, res_file_path)
        
    input_ids = torch.tensor(features['input_ids'], dtype=torch.long)
    attention_masks = torch.tensor(features['attention_mask'], dtype=torch.long)
    labels = torch.tensor(features['labels'], dtype=torch.long)
    
    dataset = TensorDataset(input_ids, attention_masks, labels)
    
    return dataset


def calc_metrics(preds, labels):
    accuracy = (preds == labels).mean()
    f1 = f1_score(y_true=labels, y_pred=preds)
    
    return accuracy, f1
    
    
def forward_step(args, model, batch, src_id, tgt_id, evaluate=False):
    # send everything to cuda
    batch = tuple(t.to(args.device) for t in batch)
    
    with torch.set_grad_enabled(not evaluate):
        batch_0 = batch[0]
        batch_size = batch_0.shape[0]
        # attach ids of labels
        seq_a = (torch.ones(batch_size) * src_id).type_as(batch_0).view(-1, 1)
        seq_b = (torch.ones(batch_size) * tgt_id).type_as(batch_0).view(-1, 1)
        
        # minus one token because we attached labels
        seq_a = torch.cat((seq_a, batch_0), dim=1)[:, :-1] 
        seq_b = torch.cat((seq_b, batch_0), dim=1)[:, :-1]
        
        # stack batches
        seq_batched = torch.cat((seq_a, seq_b), dim=0)
        
        # in LM task labels are inputs
        inputs = {'input_ids' : seq_batched, 'attention_mask' : None, 'labels' : seq_batched}
        
        # modelling_gpt2.py changed outputs to have no reduction in loss
        outputs = model(**inputs)

        losses = outputs[0].view(seq_batched.shape[0], -1)

        # Generative Loss
        # by default they mask eos token if you don't mind then just shift losses because of label_id
        if args.mask_eos_token:
            loss_mask = batch[1][:, :-1].to(torch.float32).to(args.device)
        else:
            loss_mask = batch[1][:, :-1].to(torch.float32).to(args.device)
            label_loss = torch.ones(loss_mask.shape[0], 1).type_as(loss_mask)
            loss_mask = torch.cat((label_loss, loss_mask[:, :-1]), dim=1)

        loss_src = losses[:batch_size] * loss_mask
        loss_tgt = losses[batch_size:] * loss_mask

        loss_lengths = torch.sum(loss_mask, 1, keepdim=True)
        gen_loss_src = (batch[2] == 0).to(torch.float32).unsqueeze(1) * loss_src / loss_lengths
        gen_loss_tgt = (batch[2] == 1).to(torch.float32).unsqueeze(1) * loss_tgt / loss_lengths

        gen_loss = torch.sum(gen_loss_src + gen_loss_tgt) / batch_size

        # Discriminative Loss
        loss_src = (loss_src / loss_lengths).sum(dim=1)
        loss_tgt = (loss_tgt / loss_lengths).sum(dim=1)

        class_logits = torch.stack((-loss_src, -loss_tgt), dim=1)

        if args.logit_scale:
            class_logits *= model.logit_scale
        if args.outbias:
            class_logits += model.bias

        bce_loss = torch.nn.CrossEntropyLoss()
        disc_loss = bce_loss(class_logits, batch[2])

        loss = args.disc_weight * disc_loss + args.gen_weight * gen_loss
   
    return {'loss': loss, 'logits' : class_logits, 'disc_loss' : disc_loss, 'gen_loss' : gen_loss}
    
    
def train(args, model, tokenizer, writer):
    # init train and eval datasets
    train_dataset = prepare_dataset(args, tokenizer)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    
    eval_dataset = prepare_dataset(args, tokenizer, mode='dev')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    
    # define n_steps
    if args.max_steps > 0:
        steps_total = args.max_steps
        args.n_epochs = args.max_steps // len(train_dataloader) * args.gradient_accumulation_steps + 1
    else:
        steps_total = len(train_dataloader) // args.gradient_accumulation_steps * args.n_epochs
        
    # init optimizer and scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params' : [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay' : 0.0
        },
        {
            'params' : [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay' : args.weight_decay
        }
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=steps_total
    )
    # try to load optimizer and scheduler
    optimizer_path = os.path.join(args.model_name_or_path, 'optimizer.pt')
    if os.path.isfile(optimizer_path):
        logger.info('Loading optimizer from working dir')
        optimizer.load_state_dict(torch.load(optimizer_path, map_location='cpu'))
    
    scheduler_path = os.path.join(args.model_name_or_path, 'scheduler.pt')
    if os.path.isfile(scheduler_path):
        logger.info('Loading scheduler from working dir')
        scheduler.load_state_dict(torch.load(scheduler_path))
    
    # train
    logger.info("Training begins!")
    logger.info(f"Total optimization steps: {steps_total}")
    
    global_step = 0
    epochs_trained = 0
    steps_trained_current_epoch = 0
    running_loss_t, running_loss_g, running_loss_d = 0., 0., 0.
    prev_loss_t, prev_loss_g, prev_loss_d = 0., 0., 0.
    
    # try to load trained model
    if os.path.exists(args.model_name_or_path):
        # assumed name of trained model: model_checkpoint_globalstep
        global_step = int(args.model_name_or_path.split('_')[-1][:-1])
        epochs_trained = global_step * args.gradient_accumulation_steps // len(train_dataloader) 
        steps_trained_current_epoch = global_step * args.gradient_accumulation_steps % len(train_dataloader)
        logger.info(f"epochs trained: {epochs_trained}, global_step: {global_step}/{steps_total}, "
                    f"batches trained in current epoch: {steps_trained_current_epoch}/{len(train_dataloader)}")
    
    model.zero_grad()
    
    print(tokenizer.encode(args.code_0), tokenizer.encode(args.code_1))
    src_id = tokenizer.encode(args.code_0)[0]
    tgt_id = tokenizer.encode(args.code_1)[0]
    
    for epoch in range(epochs_trained, args.n_epochs):
        logger.info(f"Starting epoch {epoch}")
        # train
        model.train()
        for step, batch in enumerate(train_dataloader):
            if steps_trained_current_epoch > 0:
                steps_trained_current_epoch -= 1
                continue
            
            results = forward_step(args, model, batch, src_id, tgt_id)
            loss = results['loss']
            loss.backward()
            
            running_loss_t += loss.item()
            running_loss_g += results['gen_loss'].item()
            running_loss_d += results['disc_loss'].item()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_max_norm)
                
                optimizer.step()
                scheduler.step()
                global_step += 1
                model.zero_grad()
                
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # print training info
                    loss_info_t = (running_loss_t - prev_loss_t) / args.logging_steps
                    loss_info_g = (running_loss_g - prev_loss_g) / args.logging_steps
                    loss_info_d = (running_loss_d - prev_loss_d) / args.logging_steps
                    
                    logger.info(
                        f"epoch: {epoch}, global step: {global_step}/{steps_total}, training loss: {loss_info_t:.6f}"
                    )
                    if writer:
                        writer.add_scalar('Total_Loss/train', loss_info_t, global_step)
                        writer.add_scalar('Gen_Loss/train', loss_info_g, global_step)
                        writer.add_scalar('Disc_Loss/train', loss_info_d, global_step)
                        writer.add_scalar('Learning rate', scheduler.get_lr()[0], global_step)
                    if args.wandb:
                        wandb.log({'Total_Loss/train': loss_info_t,
                                   'Gen_Loss/train' : loss_info_g,
                                   'Disc_Loss/train' : loss_info_d,
                                   'Learning rate' : scheduler.get_lr()[0]}
                                 ) 
                        
                    prev_loss_t = running_loss_t
                    prev_loss_g = running_loss_g
                    prev_loss_d = running_loss_d
                    
                if args.saving_steps > 0 and global_step % args.saving_steps == 0:
                    # save model checkpoint
                    output_dir = os.path.join(args.working_dir, f"model_checkpoint_{global_step}")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                   
                    model.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info(f"Saving model, args, optimizer, scheduler to {output_dir}")

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optimizer.pt'))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, 'scheduler.pt'))
            
            if args.max_steps > 0 and global_step > args.max_steps:
                break
        
        # evaluate
        logger.info(f"Training epoch {epoch} finished. Evaluating.")
        preds, labels = None, None
        total_loss, disc_loss, gen_loss = 0.0, 0.0, 0.
        model.eval()
        
        for step, batch in enumerate(eval_dataloader):
            outputs = forward_step(args, model, batch, src_id, tgt_id, evaluate=True)
            total_loss += outputs['loss'].item() * args.gradient_accumulation_steps
            disc_loss += outputs['disc_loss'].item() * args.gradient_accumulation_steps
            gen_loss += outputs['gen_loss'].item() * args.gradient_accumulation_steps
            logits = outputs['logits'].detach().cpu().numpy()
            true_labels = batch[2].detach().cpu().numpy()
            
            if preds is None:
                preds = logits
                labels = true_labels
            else:
                preds = np.append(preds, logits, axis=0)
                labels = np.append(labels, true_labels, axis=0)
        
        total_loss /= (step + 1)
        disc_loss /= (step + 1)
        gen_loss /= (step + 1)
        preds = np.argmax(preds, axis=1)
        
        accuracy, f1 = calc_metrics(preds, labels)
        if writer:
            writer.add_scalar('Total_Loss/eval', total_loss, global_step)
            writer.add_scalar('Gen_Loss/eval', gen_loss, global_step)
            writer.add_scalar('Disc_Loss/eval', disc_loss, global_step)
            writer.add_scalar('Accuracy', accuracy, epoch)
            writer.add_scalar('F1', f1, epoch)
        if args.wandb:
            wandb.log({'Total_Loss/eval' : total_loss,
                       'Gen_Loss/eval' : gen_loss,
                       'Disc_Loss/eval' : disc_loss,
                       'Accuracy' : accuracy,
                       'F1' : f1}
                     )
        
        logger.info(f"Evaluation discriminative loss after epoch {epoch} is {disc_loss:.6f}")
        logger.info(f"Evaluation generative loss after epoch {epoch} is {gen_loss:.6f}")
        logger.info(f"Evaluation accuracy after epoch {epoch} is {accuracy:.4f}")
        logger.info(f"Evaluation F1 score after epoch {epoch} is {f1:.4f}")
        
        if args.max_steps > 0 and global_step > args.max_steps:
                break
    
    return model
        

def main():
    parser = argparse.ArgumentParser()
    # main arguments
    parser.add_argument("--model_name_or_path", type=str, default='',
                       help="Path to the folder with pretrained model"
                       )
    parser.add_argument("--tokenizer_name",  default="ceshine/t5-paraphrase-paws-msrp-opinosis", type=str,
                        help="Name or path to the tokenizer of paraphrase model that will be used"
                       )
    parser.add_argument("--data_dir", required=True, type=str,
                       help="Path to the folder where train/dev/test files lie"
                       )
    parser.add_argument("--working_dir", default='.', type=str,
                        help="Path to the folder where model checkpoints will be saved"
                       )
    parser.add_argument("--cache_dir", default="cache", type=str,
                       help="Path to the folder where processed data and logs will be stored"
                       )
    parser.add_argument("--log_dir", default="logs", type=str,
                       help="Path to the folder where tensorboard logs will be written"
                       )
    parser.add_argument("--overwrite_working_dir", action='store_true',
                       help="If used models in the working directory might be overwriten"
                       )
    parser.add_argument("--overwrite_cache_dir", action='store_true',
                       help="If used features and logs in the cache directory might be overwriten"
                       )
    parser.add_argument("--code_0", default="toxic", type=str, help="Token describing 0 label")
    parser.add_argument("--code_1", default="normal", type=str, help="Token describing 1 label")
    parser.add_argument("--random_seed", default=0, type=int, help="Set seed for reproducibility")
    parser.add_argument("--cuda_device", default=0, type=int, help="On which GPU run training")
    parser.add_argument("--keep_embeddings", action='store_true',
                        help="If used no embeddings adjustment will be done"
                       )
    parser.add_argument("--logging_steps", default=-1, type=int,
                        help="Number of steps after which train performance is reported"
                       )
    parser.add_argument("--saving_steps", default=-1, type=int,
                       help="Number of steps after which checkpoint of model is saved"
                       )
    
    # training arguments
    parser.add_argument("--disc_weight", type=float, default=0.5, help="Weight of discriminative loss in learning")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="sequences longer than this value will be truncated"
                       )
    parser.add_argument("--train_batch_size", type=int, default=8,
                        help="Number of sequences in batch during training"
                       )
    parser.add_argument("--eval_batch_size", type=int, default=8,
                        help="Number of sequences in batch during evaluation"
                       )
    parser.add_argument("--n_epochs", type=int, default=2, help="Number of epochs during training")
    parser.add_argument("--max_steps", type=int, default=-1,
                       help="Number of iterations over train dataloader. If >0 "
                       "implicitly overwrites number of epochs"
                       )    
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Number of iterations over train dataloader after which backward step occurs"
                       )
    parser.add_argument("--mask_eos_token", action="store_true", help="Whether to mask <eos> token during training")
    parser.add_argument("--logit_scale", action="store_true", help="Learn scale parameter during training")
    parser.add_argument("--outbias", action="store_true", help="Learn class bias parameter during training")
    
    parser.add_argument("--weight_decay", type=float, default=0., help="Regularization parameter during training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Epsilon for Adam optimizer")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of steps for linear warmup")
    parser.add_argument("--grad_max_norm", type=float, default=1.0, help="Maximum norm of gradients")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability in GPT2LMHead")
    parser.add_argument("--tensorboard", action="store_true", help="Use tensorboard for training visualization")
    parser.add_argument("--wandb", action="store_true", help="Use Wandb for experiment tracking")
    parser.add_argument("--yelp", action="store_true")
    
    args = parser.parse_args()
    
    args.gen_weight = 1.0 - args.disc_weight
    
    # need to add space in the beginning because of t5 tokenizer
    args.code_0 = ' ' + args.code_0
    args.code_1 = ' ' + args.code_1
    
    args.device = f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu"
    
    if (os.path.exists(args.working_dir) and os.listdir(args.working_dir) and not args.overwrite_working_dir):
        raise ValueError(
            f"Working directory ({args.working_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.")
        
    for folder in [args.working_dir, args.cache_dir, args.log_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    experiment_name = f"gedi_finetuning_discweight{args.disc_weight}_lr{args.learning_rate}_warmupsteps{args.warmup_steps}"
        
    writer = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(args.log_dir, experiment_name))
    
    if args.wandb:
        wandb.login()
        wandb.init(project="gedi_finetuning", group=experiment_name)
    
    # Setup Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    
    set_seeds(args.random_seed)
    
    new_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    if args.model_name_or_path:
        gedi_model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
    else:
        # Setup GPT2 Config
        config = GPT2Config.from_pretrained('gpt2-medium', num_labels=2)
        config.nbias = 2 if args.outbias else 0
        config.logit_scale = args.logit_scale
        config.embd_pdrop = args.dropout
        config.attn_pdrop = args.dropout
        config.resid_pdrop = args.dropout
        config.output_past = True

        gedi_model = GPT2LMHeadModel.from_pretrained('gpt2-medium', config=config)
    
    #if args.model_name_or_path:
    #    old_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    #else:
    old_tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')
    old_tokenizer.pad_token = '[PAD]'
        
    if not args.keep_embeddings:
        logger.info('Changing input and output embeddings of GeDi model')
        gedi_model = adjust_embeddings(gedi_model, old_tokenizer, new_tokenizer)
    
    gedi_model.to(args.device)
    gedi_model = train(args, gedi_model, new_tokenizer, writer)
    
    output_dir = os.path.join(args.working_dir, "model_last_checkpoint")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
                   
    gedi_model.save_pretrained(output_dir)

    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
    logger.info(f"Saving model, args to {output_dir}")
    
    return

if __name__ == "__main__":
    main()