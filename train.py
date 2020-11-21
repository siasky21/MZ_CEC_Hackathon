import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from attrdict import AttrDict

import torch
from transformers import (
  AdamW,
  ElectraConfig,
  ElectraTokenizer,
  get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from fastprogress.fastprogress import master_bar, progress_bar
from mz import MzTextClassificationDataset
from koelectra import koElectraForSequenceClassification


def simple_accuracy(labels, preds):
    return np.mean(labels == preds)


def acc_score(labels, preds):
    return {
        "acc": simple_accuracy(labels, preds),
    }

def train(args,
          model,
          train_dataset,
          dev_dataset=None,
          test_dataset=None):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * args.warmup_proportion), num_training_steps=t_total)

    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Train!
    print("***** Running training *****")
    print("  Num examples = %d" % len(train_dataset))
    print("  Num Epochs = %d" % args.num_train_epochs)
    print("  Total train batch size = %d" % args.train_batch_size)
    print("  Gradient Accumulation steps = %d" % args.gradient_accumulation_steps)
    print("  Total optimization steps = %d" % t_total)
    print("  Logging steps = %d" % args.logging_steps)
    print("  Save steps = %d" % args.save_steps)

    global_step = 0
    tr_loss = 0.0

    model.zero_grad()
    mb = master_bar(range(int(args.num_train_epochs)))
    for epoch in mb:
        epoch_iterator = progress_bar(train_dataloader, parent=mb)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            # batch = tuple(t.to(args.device) for t in batch)
            '''
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3]
            }
            '''
            inputs = {
                "input_ids": batch['input_ids'],
                "attention_mask": batch['attention_mask'],
                "labels": batch['labels']
            }
            # if args.model_type not in ["distilkobert", "xlm-roberta"]:
            #     inputs["token_type_ids"] = batch[2]  # Distilkobert, XLM-Roberta don't use segment_ids
            outputs = model(**inputs)

            loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    len(train_dataloader) <= args.gradient_accumulation_steps
                    and (step + 1) == len(train_dataloader)
            ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.evaluate_test_during_training:
                        evaluate(args, model, test_dataset, "test", global_step)
                    else:
                        evaluate(args, model, dev_dataset, "dev", global_step)

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )
                    model_to_save.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    print("Saving model checkpoint to {}".format(output_dir))

                    if args.save_optimizer:
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        print("Saving optimizer and scheduler states to {}".format(output_dir))

            if args.max_steps > 0 and global_step > args.max_steps:
                break

        mb.write("Epoch {} done".format(epoch + 1))

        if args.max_steps > 0 and global_step > args.max_steps:
            break

    return global_step, tr_loss / global_step


def evaluate(args, model, eval_dataset, mode, global_step=None):
    results = {}
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    if global_step != None:
        print("***** Running evaluation on {} dataset ({} step) *****".format(mode, global_step))
    else:
        print("***** Running evaluation on {} dataset *****".format(mode))
    print("  Num examples = {}".format(len(eval_dataset)))
    print("  Eval Batch size = {}".format(args.eval_batch_size))
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    for batch in progress_bar(eval_dataloader):
        model.eval()
        # batch = tuple(t.to(args.device) for t in batch)


        with torch.no_grad():
            '''
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3]
            }
            if args.model_type not in ["distilkobert", "xlm-roberta"]:
                inputs["token_type_ids"] = batch[2]  # Distilkobert, XLM-Roberta don't use segment_ids
            '''
            inputs = {
                "input_ids": batch['input_ids'],
                "attention_mask": batch['attention_mask'],
                "labels": batch['labels']
            }
            outputs = model(**inputs)
            # outputs = model(**batch)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    result = acc_score(out_label_ids, preds)
    results.update(result)

    output_dir = os.path.join(args.output_dir, mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_eval_file = os.path.join(output_dir, "{}-{}.txt".format(mode, global_step) if global_step else "{}.txt".format(mode))
    with open(output_eval_file, "w") as f_w:
        print("***** Eval results on {} dataset *****".format(mode))
        for key in sorted(results.keys()):
            print("  {} = {}".format(key, str(results[key])))
            f_w.write("  {} = {}\n".format(key, str(results[key])))

    return results


if __name__ == '__main__':
    config_path = './config/koelectra-small-v2.json'

    with open(config_path) as f:
        args = AttrDict(json.load(f))

    print("Training/evaluation parameters {}".format(args))

    args.output_dir = os.path.join(args.ckpt_dir, args.output_dir)

    config = ElectraConfig.from_pretrained(args.model_name_or_path)
    tokenizer = ElectraTokenizer.from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case
    )
    model = koElectraForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
        num_labels=60
    )

    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    model.to(args.device)

    # Load Dataset
    train_dataset = MzTextClassificationDataset(file_path=args.train_file, tokenizer=tokenizer, device=args.device)
    dev_dataset = MzTextClassificationDataset(file_path=args.dev_file, tokenizer=tokenizer, device=args.device) if args.dev_file else None
    test_dataset = MzTextClassificationDataset(file_path=args.test_file, tokenizer=tokenizer, device=args.device) if args.test_file else None

    global_step, tr_loss = train(args, model, train_dataset, dev_dataset, test_dataset)
    print(" global_step = {}, average loss = {}".format(global_step, tr_loss))