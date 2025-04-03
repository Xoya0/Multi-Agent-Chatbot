import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from typing import List, Dict, Set, Optional, Tuple
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

class DialogueDataset(Dataset):
    def __init__(self, conversations: List[Dict[str, str]], tokenizer, max_length: int = 512):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        conversation = self.conversations[idx]
        text = self._format_conversation(conversation)
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {key: val.squeeze(0) for key, val in encoding.items()}

    def __len__(self) -> int:
        return len(self.conversations)

    def _format_conversation(self, conversation: Dict[str, str]) -> str:
        formatted = ''
        if 'personality' in conversation:
            formatted += 'Personality Traits:\n'
            formatted += '\n'.join([f'• {trait}' for trait in conversation['personality']])
            formatted += '\n\n'
        formatted += 'Conversation:\n' + conversation['dialogue']
        return formatted

def process_movie_dialogue(line: str, line_dict: Dict[str, str]) -> Optional[Dict[str, str]]:
    try:
        parts = line.strip().split(' +++$+++ ')
        if len(parts) >= 4:
            line_ids = eval(parts[3])
            dialogue_turns = []
            current_speaker = 'Human:'
            
            for line_id in line_ids:
                if line_id in line_dict:
                    dialogue_turns.append(f'{current_speaker} {line_dict[line_id]}')
                    current_speaker = 'Assistant:' if current_speaker == 'Human:' else 'Human:'
            
            if len(dialogue_turns) >= 2:
                return {'dialogue': '\n'.join(dialogue_turns)}
    except Exception as e:
        logging.warning(f'Error processing dialogue line: {str(e)}')
    return None

def load_movie_dialogues(data_path: str) -> List[Dict[str, str]]:
    conversations = []
    lines_file = os.path.join(data_path, 'movie_lines.txt')
    conversations_file = os.path.join(data_path, 'movie_conversations.txt')

    try:
        # Load lines with parallel processing
        line_dict = {}
        if os.path.exists(lines_file):
            with open(lines_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.strip().split(' +++$+++ ')
                    if len(parts) >= 5:
                        line_dict[parts[0]] = parts[4].strip()

        # Process conversations in parallel
        if os.path.exists(conversations_file):
            with open(conversations_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                with ThreadPoolExecutor() as executor:
                    process_func = partial(process_movie_dialogue, line_dict=line_dict)
                    for result in executor.map(process_func, lines):
                        if result:
                            conversations.append(result)

    except Exception as e:
        logging.error(f'Error processing movie dialogues: {str(e)}')

    return conversations

def load_persona_chat() -> Tuple[List[Dict[str, str]], Set[Tuple[str, ...]]]:
    conversations = []
    personalities = set()
    
    try:
        dataset = load_dataset('AlekseyKorshuk/persona-chat', split='train')
        for example in tqdm(dataset, desc='Processing Persona-Chat'):
            if 'personality' in example and 'utterances' in example:
                personality = tuple(example['personality'])
                personalities.add(personality)
                
                dialogue_turns = []
                current_speaker = 'Human:'
                for utterance in example['utterances']:
                    if isinstance(utterance, dict) and 'text' in utterance:
                        dialogue_turns.append(f'{current_speaker} {utterance["text"]}')
                        current_speaker = 'Assistant:' if current_speaker == 'Human:' else 'Human:'
                
                if dialogue_turns:
                    conversations.append({
                        'personality': list(personality),
                        'dialogue': '\n'.join(dialogue_turns)
                    })
    except Exception as e:
        logging.error(f'Error loading Persona-Chat dataset: {str(e)}')
    
    return conversations, personalities

def prepare_training_data(model_name: str = 'bigscience/bloom-560m') -> Tuple[DialogueDataset, AutoTokenizer]:
    logging.info(f'Preparing training data using {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load and combine datasets with emphasis on emotional content
    persona_conversations, personalities = load_persona_chat()
    movie_conversations = load_movie_dialogues('data')
    
    # Filter and enhance conversations for emotional content
    enhanced_conversations = []
    for conv in persona_conversations + movie_conversations:
        if 'dialogue' in conv:
            # Add emotional context hints
            if 'personality' in conv:
                conv['dialogue'] = f"Emotional context: warm and engaging\n{conv['dialogue']}"
            # Trim long responses
            turns = conv['dialogue'].split('\n')
            processed_turns = []
            for turn in turns:
                if len(turn.split()) > 30:  # Limit response length
                    turn = ' '.join(turn.split()[:30]) + '...'
                processed_turns.append(turn)
            conv['dialogue'] = '\n'.join(processed_turns)
            enhanced_conversations.append(conv)
    
    all_conversations = enhanced_conversations
    
    logging.info(f'Total conversations loaded: {len(all_conversations)}')
    dataset = DialogueDataset(all_conversations, tokenizer)
    
    return dataset, tokenizer

def train_model(
    training_args: TrainingArguments,
    model_name: str = 'bigscience/bloom-560m',
    resume_from_checkpoint: Optional[str] = None
) -> None:
    try:
        logging.info(f'Initializing model: {model_name}')
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            gradient_checkpointing=True,
            torch_dtype=torch.float16 if training_args.fp16 else torch.float32
        )
        
        dataset, tokenizer = prepare_training_data(model_name)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            optimizers=(torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate), None)
        )

        logging.info('Starting training...')
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        output_dir = training_args.output_dir
        logging.info(f'Saving model to {output_dir}')
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

    except Exception as e:
        logging.error(f'Error during model training: {str(e)}')
        raise

def main():
    try:
        training_args = TrainingArguments(
            output_dir='./trained_model',
            num_train_epochs=5,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            logging_dir='./logs',
            logging_steps=50,
            save_strategy='steps',
            save_steps=500,
            evaluation_strategy='steps',
            eval_steps=500,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model='loss',
            greater_is_better=False,
            report_to='tensorboard',
            fp16=True,
            weight_decay=0.01,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            gradient_checkpointing=True,
            optim='adamw_torch',
            lr_scheduler_type='cosine',
            remove_unused_columns=False,
            group_by_length=True,
            length_column_name='length',
            max_grad_norm=1.0,
            dataloader_drop_last=True
        )

        train_model(training_args)

    except Exception as e:
        logging.error(f'Training failed: {str(e)}')
        raise

if __name__ == '__main__':
    main()