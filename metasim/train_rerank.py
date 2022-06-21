import sys

sys.path += ['./']

from mwoz_driver.train_mwoz import load_data, Dataset, MWOZData, pad_sequence, Accelerator, T5Tokenizer, \
    T5ForConditionalGeneration, Adafactor, get_linear_schedule_with_warmup
import torch
import torch.nn.functional as F
import numpy as np
import json
from tqdm import tqdm
import os


class ReRankData(MWOZData):
    def __init__(self, *args, neg_num=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.neg_num = neg_num

    def __getitem__(self, index, **kwargs):
        goal = self.goal[index] if 'goal' not in kwargs else kwargs['goal']
        context = self.context[index] if 'context' not in kwargs else kwargs['context']
        actions = self.actions[index] if 'actions' not in kwargs else kwargs['actions']
        template = self.template[index] if 'template' not in kwargs else kwargs['template']
        memory = self.memory[index] if 'memory' not in kwargs else kwargs['memory']

        goal = self.tokenizer.encode(goal, truncation=True, max_length=self.goal_len)
        context = self.tokenizer.encode(context)
        actions = [self.tokenizer.encode(item, truncation=True, max_length=self.response_len) for item in actions]
        template = self.tokenizer.encode(template, truncation=True, max_length=self.response_len)
        memory = [self.tokenizer.encode(m, truncation=True, max_length=self.prompt_len) for m in memory]
        np.random.shuffle(memory)
        memory = memory[:self.neg_num]
        inputs, targets = [], []
        inp, _ = self.collector(goal, context, None, None, actions, template, None, None, self.context_len)
        inputs.append(inp)
        targets.append(self.tokenizer.encode('yes'))
        for one in memory:
            inp, _ = self.collector(goal, context, None, None, actions, one, None, None, self.context_len)
            inputs.append(inp)
            targets.append(self.tokenizer.encode('no'))

        inputs = [torch.tensor(m[:self.context_len]) for m in inputs]
        targets = [torch.tensor(m[:self.response_len]) for m in targets]

        return inputs, targets

    @staticmethod
    def collate_fn(data):
        context, response = zip(*data)
        context = sum(context, [])
        response = sum(response, [])
        context = pad_sequence(context, batch_first=True, padding_value=0)
        return {
            'input_ids': context,
            'attention_mask': context.ne(0),
            'labels': pad_sequence(response, batch_first=True, padding_value=-100),
        }


class TestRankData(ReRankData):
    def collate_fn(self, data):
        context, response = zip(*data)
        context = [x + [x[-1]] * (self.neg_num + 1 - len(x)) for x in context]
        response = [x + [x[-1]] * (self.neg_num + 1 - len(x)) for x in response]
        context = sum(context, [])
        response = sum(response, [])
        context = pad_sequence(context, batch_first=True, padding_value=0)
        return {
            'input_ids': context,
            'attention_mask': context.ne(0),
            'labels': pad_sequence(response, batch_first=True, padding_value=-100),
        }


def rerank_collector(goal=None, context=None, response=None, prompt=None, actions=None, template=None,
                     delex_context=None, memory=None, context_len=512):
    context_actions = sum(actions[:-1], [])[-(context_len - len(goal)):]
    context = context[-(context_len - len(goal) - len(context_actions) - len(template)):]
    inputs = goal + context + context_actions + [32099] + template
    targets = actions[-1]
    return inputs, targets


def add_retrieval(data, file):
    if file is None:
        return data
    file = json.load(open(file))
    assert len(data) == len(file)
    out = []
    for line, ret in zip(data, file):
        line = line + [ret]
        out.append(line)
    return out


def train():
    accelerator = Accelerator()
    batch_size = 6
    epochs = 30
    save_path = 'ckpt/mwoz-rerank-4'
    print(save_path)
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained('t5-base')

    # optimizer = AdamW(generator.parameters(), 1e-4)
    optimizer = Adafactor(model.parameters(), lr=1e-3, relative_step=False, scale_parameter=False)
    # optimizer = Adafactor(model.parameters())

    dataset = ReRankData(
        # load_data('dataset/mwoz/MultiWOZ_2.1/train_data.json'),
        add_retrieval(load_data('dataset/mwoz/MultiWOZ_2.1/train_data.json'),
                      'dataset/mwoz/train_template.json'),
        context_len=256, response_len=64, goal_len=64, prompt_len=64, neg_num=4,
        tokenizer=tokenizer, collector=rerank_collector)

    # dataset.replace_templates([line[:-1] for line in open('ckpt/mwoz-metaphor-train/text/10.txt')])

    accelerator.print(f'data size={len(dataset)}')
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=batch_size,
                                              shuffle=True, num_workers=8)

    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=500, num_training_steps=epochs * len(data_loader))
    # scheduler = None

    os.makedirs(save_path, exist_ok=True)

    # accelerator.print(tokenizer.decode(dataset[128][0]))
    # accelerator.print('==>')
    # accelerator.print(tokenizer.decode(dataset[128][1]))

    for epoch in range(epochs):
        accelerator.print(f'Training epoch {epoch}')
        accelerator.wait_for_everyone()
        model.train()
        tk0 = tqdm(data_loader, total=len(data_loader))
        loss_report = []
        for batch in tk0:
            out = model(**batch)
            loss = out.loss
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            loss_report.append(loss.item())
            tk0.set_postfix(loss=sum(loss_report) / len(loss_report))
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            accelerator.save(accelerator.unwrap_model(model).state_dict(), f'{save_path}/{epoch}.pt')


def eval_acc(predicts, answers):
    return sum([int(p == a) for p, a in zip(predicts, answers)]) / max(len(predicts), 1)


def test():
    batch_size = 8
    save_path = 'ckpt/mwoz-rerank'
    # out_path = save_path + '-agenda'
    out_path = save_path
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained('t5-base')

    dataset = TestRankData(
        # load_data('dataset/mwoz/MultiWOZ_2.1/train_data.json'),
        add_retrieval(load_data('dataset/mwoz/MultiWOZ_2.1/test_data.json'),
                      'dataset/mwoz/test_template.json'),
        context_len=256, response_len=64, goal_len=64, prompt_len=64, neg_num=16,
        tokenizer=tokenizer, collector=rerank_collector)

    # dataset.replace_system_actions([line[:-1] for line in open('ckpt/mwoz-nlu-final/text/13.txt')])
    # dataset.replace_user_actions([line[:-1] for line in open('ckpt/mwoz-policy-final/text/6.txt')])
    # dataset.replace_templates([line[:-1] for line in open('ckpt/mwoz-metaphor/text/10.txt')])
    model = model.cuda()

    for epoch in range(0, 100, 5):
        if os.path.exists(f'{save_path}/{epoch}.pt'):
            print(f'eval model {save_path}/{epoch}.pt, {out_path}')

            model.load_state_dict(torch.load(f'{save_path}/{epoch}.pt'))
            model.eval()
            data_loader = torch.utils.data.DataLoader(
                dataset,
                collate_fn=dataset.collate_fn,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8)
            tk0 = tqdm(data_loader, total=len(data_loader))
            output_predict = []
            output_labels = []
            acc = []
            with torch.no_grad():
                for batch in tk0:
                    batch = {k: v.cuda() for k, v in batch.items()}
                    # predict = model.generate(
                    #     input_ids=batch['input_ids'].cuda().long(),
                    #     attention_mask=batch['attention_mask'].cuda(),
                    #     decoder_start_token_id=0,
                    #     # num_beams=3,
                    #     max_length=128)
                    out = model(**batch)
                    logits = out.logits[:, 0, :]
                    logits = logits.view(-1, 17, 32128)
                    logits = F.softmax(logits, dim=-1)
                    yes_prob = logits[:, :, 4273]
                    no_prob = logits[:, :, 150]
                    prob = yes_prob
                    acc.append((prob.argmax(dim=-1) == 0).float().mean().item())
                    tk0.set_postfix(acc=sum(acc) / len(acc))
                    # print(yes_prob.size())
                    # print(logits.size())
                    # predict = predict.cpu().tolist()
                    # predict = tokenizer.batch_decode(predict, skip_special_tokens=True)
                    # output_predict.extend(predict)
                    #
                    # batch['labels'][batch['labels'] == -100] = 0
                    # output_labels.extend(tokenizer.batch_decode(batch['labels'], skip_special_tokens=True))
            output_labels = dataset.response
            print(eval_acc(output_predict, output_labels))


if __name__ == '__main__':
    train()
    # test()
#
