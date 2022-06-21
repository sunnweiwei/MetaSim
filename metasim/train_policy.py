import sys

sys.path += ['./']
from train_mwoz import *
import torch.nn.functional as F


class DuoMWOZData(MWOZData):
    def __init__(self, *args, teacher_collector=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_collector = teacher_collector

    def __getitem__(self, index, **kwargs):
        goal = self.goal[index] if 'goal' not in kwargs else kwargs['goal']
        context = self.context[index] if 'context' not in kwargs else kwargs['context']
        response = self.response[index] if 'response' not in kwargs else kwargs['response']
        prompt = self.prompt[index] if 'prompt' not in kwargs else kwargs['prompt']
        actions = self.actions[index] if 'actions' not in kwargs else kwargs['actions']
        template = self.template[index] if 'template' not in kwargs else kwargs['template']
        delex_context = self.delex_context[index] if 'delex_context' not in kwargs else kwargs['delex_context']

        goal = self.tokenizer.encode(goal, truncation=True, max_length=self.goal_len)
        context = self.tokenizer.encode(context)
        response = self.tokenizer.encode(response, truncation=True, max_length=self.response_len)
        prompt = self.tokenizer.encode(prompt, truncation=True, max_length=self.prompt_len)
        actions = [self.tokenizer.encode(item, truncation=True, max_length=self.prompt_len) for item in actions]
        template = self.tokenizer.encode(template, truncation=True, max_length=self.response_len)
        delex_context = self.tokenizer.encode(delex_context)

        inputs, targets = self.collector(goal, context, response,
                                         prompt, actions, template, delex_context, self.context_len)
        teacher_inputs, teacher_targets = self.teacher_collector(goal, context, response, prompt, actions, template,
                                                                 delex_context, self.context_len)
        inputs = inputs[:self.context_len]
        targets = targets[:self.response_len]
        inputs = torch.tensor(inputs)
        targets = torch.tensor(targets)

        teacher_inputs = teacher_inputs[:self.context_len]
        teacher_targets = teacher_targets[:self.response_len]
        teacher_inputs = torch.tensor(teacher_inputs)
        teacher_targets = torch.tensor(teacher_targets)

        return inputs, targets, teacher_inputs, teacher_targets

    @staticmethod
    def collate_fn(data):
        context, response, teacher_context, teacher_response = zip(*data)
        context = pad_sequence(context, batch_first=True, padding_value=0)
        teacher_context = pad_sequence(teacher_context, batch_first=True, padding_value=0)
        return {
            'input_ids': context,
            'attention_mask': context.ne(0),
            'labels': pad_sequence(response, batch_first=True, padding_value=-100),
            'teacher_input_ids': teacher_context,
            'teacher_attention_mask': teacher_context.ne(0),
            'teacher_labels': pad_sequence(teacher_response, batch_first=True, padding_value=-100),
        }


def train():
    accelerator = Accelerator()
    batch_size = 6
    epochs = 20
    save_path = 'ckpt/mwoz-policy-kl'
    print(save_path)
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained('t5-base')

    # optimizer = AdamW(generator.parameters(), 1e-4)
    optimizer = Adafactor(model.parameters(), lr=1e-3, relative_step=False, scale_parameter=False)
    # optimizer = Adafactor(model.parameters())

    dataset = DuoMWOZData(load_data('dataset/mwoz/MultiWOZ_2.1/train_data.json'),
                          context_len=512, response_len=128, goal_len=128, prompt_len=128,
                          tokenizer=tokenizer, collector=policy_collector, teacher_collector=teacher_policy_collector)

    teacher_model = T5ForConditionalGeneration.from_pretrained('t5-base')
    teacher_model.load_state_dict(
        torch.load(f'ckpt/mwoz-teacher-policy/10.pt', map_location=lambda s, loc: s))
    teacher_model = teacher_model.cuda()

    # dataset.replace_templates([line[:-1] for line in open('ckpt/mwoz-metaphor-train/text/10.txt')])

    accelerator.print(f'data size={len(dataset)}')
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=batch_size,
                                              shuffle=True, num_workers=8)

    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=len(data_loader) // 5, num_training_steps=epochs * len(data_loader))

    os.makedirs(save_path, exist_ok=True)

    for epoch in range(epochs):
        accelerator.print(f'Training epoch {epoch}')
        accelerator.wait_for_everyone()
        model.train()
        teacher_model.eval()
        tk0 = tqdm(data_loader, total=len(data_loader))
        loss_report = []
        for batch in tk0:
            out = model(**{k: v for k, v in batch.items() if not k.startswith('teacher')})
            with torch.no_grad():
                teacher_out = teacher_model(**{k[8:]: v for k, v in batch.items() if k.startswith('teacher')})
            loss = F.kl_div(F.log_softmax(out.logits, dim=-1), F.log_softmax(teacher_out.logits, dim=-1),
                            reduction='none', log_target=True)
            loss = loss[batch['labels'] != 100]
            loss = loss.sum() / batch['labels'].size(0)
            loss = loss

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


if __name__ == '__main__':
    train()
