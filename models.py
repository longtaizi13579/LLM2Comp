from llama_model import CustomLlamaForCausalLM
from torch import nn
import torch
from transformers import (
    LlamaTokenizer, 
    LlamaForCausalLM, 
    AutoTokenizer
)
import copy
import os
from peft import LoraConfig, TaskType, get_peft_model
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class EncoderWrapper(torch.nn.Module):
    """A wrapper to duplicate input and split encoder hidden states into two parts."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_masks, dummy_tensor):
        # Copy attention masks
        attention_mask = attention_masks.to(input_ids.device)

        # Duplicate inputs -> batch size doubles
        input_ids = torch.cat([input_ids, input_ids.clone()], dim=0)
        attention_mask = torch.cat([attention_mask, attention_mask.clone()], dim=0)

        # Run encoder
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        hidden_states = output.hidden_states[-1]
        hidden_state_first, hidden_state_second = hidden_states.chunk(2, dim=0)

        return hidden_state_first, hidden_state_second

    def save_pretrained(self, address):
        self.encoder.save_pretrained(address)


class EncoderWrapperSupervised(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    

    def forward(self, input_ids, attention_masks, dummy_tensor):
        # Copy attention masks
        attention_mask = attention_masks.to(input_ids.device)
    
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        
        hidden_states = output.hidden_states[-1]
        
        return hidden_states

    def save_pretrained(self, address):
        self.encoder.save_pretrained(address)


class Llm2Comp_NLL_First_Stage_Bidirection(nn.Module):
    """First-stage model with NLL loss for bidirectional training."""
    def __init__(self, args, local_rank=0, **kwargs):
        super().__init__()
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        self.model = CustomLlamaForCausalLM.from_pretrained(model_name)
        self.origin_model = LlamaForCausalLM.from_pretrained(model_name)

        for param in self.origin_model.parameters():
            param.requires_grad = False

        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)

        lora_config = LoraConfig(
            r=16,
            target_modules=["q_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
            task_type=TaskType.FEATURE_EXTRACTION,
            lora_alpha=32,
            lora_dropout=0.05
        )
        lora_model = get_peft_model(self.model, lora_config)
        lora_model.print_trainable_parameters()

        self.vocab_size = len(self.tokenizer)
        self.model = lora_model
        self.add_sequence_length = args.sequence_length
        self.model.resize_token_embeddings(self.vocab_size + self.add_sequence_length)
        self.origin_model.resize_token_embeddings(self.vocab_size + self.add_sequence_length)

        torch.cuda.set_device(local_rank)
        self.dim = self.model.config.hidden_size
        self.model.base_model.model.model.embed_tokens.weight.requires_grad = True
        self.global_step = 0

    def forward(self, input_ids, embedding_indices, continuation_labels, whole_input_ids, continuation_length):
        self.global_step += 1
        fir, sec = input_ids.shape
        first, second = zip(*embedding_indices)

        # Attention mask for continuation
        attention_masks = torch.zeros(fir, sec, dtype=int)
        for idx, site in enumerate(second):
            attention_masks[idx][:site + self.add_sequence_length] = 1

        # Forward pass
        whole_weight_embedding = self.model(
            input_ids,
            attention_mask=attention_masks.cuda(),
            output_hidden_states=True
        ).hidden_states[-1]

        # Extract embeddings
        weight_embedding = []
        for idx, now_weight_embedding in enumerate(whole_weight_embedding):
            begin_ptr = second[idx]
            weight_embedding.append(
                now_weight_embedding[begin_ptr:begin_ptr+self.add_sequence_length].unsqueeze(0)
            )
        weight_embedding = torch.cat(weight_embedding, dim=0)

        # Prepare labels
        new_labels = copy.deepcopy(continuation_labels)
        new_labels[new_labels == self.tokenizer.eos_token_id] = -100
        new_labels[:, 0] = -100

        sentence_embedding = self.origin_model.base_model.embed_tokens(continuation_labels)
        inputs_embeds = torch.cat(
            [sentence_embedding[:, :1], weight_embedding, sentence_embedding[:, 1:]],
            dim=1
        ).cuda()

        negative_ones = -100 * torch.ones(
            (new_labels.shape[0], self.add_sequence_length),
            dtype=continuation_labels.dtype
        ).cuda()
        new_labels = torch.cat((negative_ones, new_labels), dim=1)

        fir, sec, thi = inputs_embeds.shape
        ct_loss = self.origin_model(
            inputs_embeds=inputs_embeds,
            attention_mask=torch.ones(fir, sec, dtype=torch.int64).cuda(),
            labels=new_labels,
            output_hidden_states=True
        ).loss

        return ct_loss


class Llm2Comp_KL_First_Stage_Bidirection(nn.Module):
    """First-stage model with KL divergence loss for bidirectional training."""
    def __init__(self, args, local_rank=0, **kwargs):
        super().__init__()
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        self.model = CustomLlamaForCausalLM.from_pretrained(model_name)
        self.origin_model = LlamaForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.eos_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id

        for param in self.origin_model.parameters():
            param.requires_grad = False

        lora_config = LoraConfig(
            r=16,
            target_modules=["q_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
            task_type=TaskType.FEATURE_EXTRACTION,
            lora_alpha=32,
            lora_dropout=0.05
        )
        lora_model = get_peft_model(self.model, lora_config)
        lora_model.print_trainable_parameters()

        self.vocab_size = len(self.tokenizer)
        self.model = lora_model
        self.add_sequence_length = args.sequence_length
        self.model.resize_token_embeddings(self.vocab_size + self.add_sequence_length)
        self.origin_model.resize_token_embeddings(self.vocab_size + self.add_sequence_length)

        torch.cuda.set_device(local_rank)
        self.dim = self.model.config.hidden_size
        self.model.base_model.model.model.embed_tokens.weight.requires_grad = True
        self.global_step = 0

    def forward(self, input_ids, embedding_indices, continuation_labels, whole_input_ids, continuation_length):
        self.global_step += 1
        fir, sec = input_ids.shape
        first, second = zip(*embedding_indices)

        attention_masks = torch.zeros(fir, sec, dtype=int)
        for idx, site in enumerate(second):
            attention_masks[idx][:site + self.add_sequence_length] = 1

        whole_weight_embedding = self.model(
            input_ids,
            attention_mask=attention_masks.cuda(),
            output_hidden_states=True
        ).hidden_states[-1]

        continuation_logit = self.origin_model(
            whole_input_ids,
            attention_mask=torch.ones(whole_input_ids.shape, dtype=int).cuda(),
            output_hidden_states=True
        ).logits

        weight_embedding, label_logit = [], []
        for idx, now_weight_embedding in enumerate(whole_weight_embedding):
            now_continuation_logit = continuation_logit[idx]
            begin_ptr = second[idx]
            end_ptr = begin_ptr + continuation_length[idx]

            label_logit.append(now_continuation_logit[begin_ptr - 1:end_ptr].unsqueeze(0))
            weight_embedding.append(now_weight_embedding[begin_ptr:begin_ptr+self.add_sequence_length].unsqueeze(0))

        weight_embedding = torch.cat(weight_embedding, dim=0)

        new_labels = copy.deepcopy(continuation_labels)
        new_labels[new_labels == self.eos_token_id] = -100
        new_labels[:, 0] = -100

        sentence_embedding = self.origin_model.base_model.embed_tokens(continuation_labels)
        inputs_embeds = torch.cat(
            [sentence_embedding[:, :1], weight_embedding, sentence_embedding[:, 1:]],
            dim=1
        ).cuda()

        negative_ones = -100 * torch.ones(
            (new_labels.shape[0], self.add_sequence_length),
            dtype=continuation_labels.dtype
        ).cuda()
        new_labels = torch.cat((negative_ones, new_labels), dim=1)

        ct_logit = self.origin_model(
            inputs_embeds=inputs_embeds,
            attention_mask=torch.ones(inputs_embeds.shape[:2], dtype=torch.int64).cuda(),
            labels=new_labels,
            output_hidden_states=True
        ).logits

        # Compute KL divergence
        all_kl_loss = []
        for idx in range(len(ct_logit)):
            label_length = continuation_length[idx]
            now_logit = ct_logit[idx][self.add_sequence_length:self.add_sequence_length + 1 + label_length]
            now_logit_softmax = F.softmax(now_logit, dim=-1)
            target_logit_softmax = F.softmax(label_logit[idx][0], dim=-1)
            kl_loss = (target_logit_softmax.log() - now_logit_softmax.log()) * target_logit_softmax
            kl_loss = torch.mean(torch.sum(kl_loss, dim=-1), dim=-1)
            all_kl_loss.append(kl_loss)

        return torch.mean(torch.stack(all_kl_loss, dim=0), dim=0)

class Llm2Comp_Rc_First_Stage_Bidirection(nn.Module):
    """First-stage model with Reconstruction (Rc) objective for bidirectional training."""
    def __init__(self, args, local_rank=0, **kwargs):
        super().__init__()
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        self.model = CustomLlamaForCausalLM.from_pretrained(model_name)
        self.origin_model = LlamaForCausalLM.from_pretrained(model_name)
        for param in self.origin_model.parameters():
            param.requires_grad = False
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)

        lora_config = LoraConfig(
            r=16,
            target_modules=["q_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
            task_type=TaskType.FEATURE_EXTRACTION,
            lora_alpha=32,
            lora_dropout=0.05
        )
        lora_model = get_peft_model(self.model, lora_config)
        lora_model.print_trainable_parameters()
        self.vocab_size = len(self.tokenizer)
        self.model = lora_model
        self.add_sequence_length = args.sequence_length
        self.model.resize_token_embeddings(self.vocab_size + self.add_sequence_length)
        self.origin_model.resize_token_embeddings(self.vocab_size + self.add_sequence_length)

        torch.cuda.set_device(local_rank)
        self.dim = self.model.config.hidden_size
        self.model.base_model.model.model.embed_tokens.weight.requires_grad = True
        self.global_step = 0

    def forward(self, input_ids, embedding_indices):
        fir, sec = input_ids.shape
        first, second = zip(*embedding_indices)

        attention_masks = torch.zeros(fir, sec, dtype=int)
        for idx, site in enumerate(second):
            attention_masks[idx][:site + self.add_sequence_length] = 1

        whole_weight_embedding = self.model(
            input_ids,
            attention_mask=attention_masks.cuda(),
            output_hidden_states=True
        ).hidden_states[-1]

        weight_embedding = []
        for idx, now_weight_embedding in enumerate(whole_weight_embedding):
            begin_ptr = second[idx]
            weight_embedding.append(
                now_weight_embedding[begin_ptr:begin_ptr+self.add_sequence_length].unsqueeze(0)
            )
        weight_embedding = torch.cat(weight_embedding, dim=0)

        new_labels = copy.deepcopy(input_ids)
        new_labels[new_labels == self.tokenizer.eos_token_id] = -100
        new_labels[new_labels > 32000] = -100
        new_labels[:, 0] = -100

        sentence_embedding = self.origin_model.base_model.embed_tokens(input_ids)
        inputs_embeds = torch.cat(
            [sentence_embedding[:, :1], weight_embedding, sentence_embedding[:, 1:]],
            dim=1
        ).cuda()

        negative_ones = -100 * torch.ones(
            (new_labels.shape[0], self.add_sequence_length),
            dtype=input_ids.dtype
        ).cuda()
        new_labels = torch.cat((negative_ones, new_labels), dim=1)

        fir, sec, thi = inputs_embeds.shape
        ct_loss = self.origin_model(
            inputs_embeds=inputs_embeds,
            attention_mask=torch.ones(fir, sec, dtype=torch.int64).cuda(),
            labels=new_labels,
            output_hidden_states=True
        ).loss

        return ct_loss


class Llm2Comp_Rc_First_Stage_Bidirection_Checkpoint(nn.Module):
    """First-stage Rc model with gradient checkpointing to reduce memory usage."""
    def __init__(self, args, local_rank=0, **kwargs):
        super().__init__()
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        self.model = CustomLlamaForCausalLM.from_pretrained(model_name)
        self.origin_model = LlamaForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.eos_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id

        for param in self.origin_model.parameters():
            param.requires_grad = False

        lora_config = LoraConfig(
            r=16,
            target_modules=["q_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
            task_type=TaskType.FEATURE_EXTRACTION,
            lora_alpha=32,
            lora_dropout=0.05
        )
        lora_model = get_peft_model(self.model, lora_config)
        lora_model.print_trainable_parameters()

        self.vocab_size = len(self.tokenizer)
        self.model = lora_model
        self.add_sequence_length = args.sequence_length

        torch.cuda.set_device(local_rank)
        self.dim = self.model.config.hidden_size
        self.model.base_model.model.model.embed_tokens.weight.requires_grad = True
        self.global_step = 0
        self.model_wrapper()

    def model_wrapper(self):
        self.model = EncoderWrapperSupervised(self.model)
        self.encoder_gpu_train_limit = 4

    def forward(self, input_ids, embedding_indices):
        fir, sec = input_ids.shape
        first, second = zip(*embedding_indices)

        attention_masks = torch.zeros(fir, sec, dtype=int)
        for idx, site in enumerate(second):
            attention_masks[idx][:site + self.add_sequence_length] = 1

        dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
        all_weight_embedding = []

        for sub_b in range(0, input_ids.shape[0], self.encoder_gpu_train_limit):
            sub_input_ids = input_ids[sub_b:sub_b+self.encoder_gpu_train_limit]
            sub_attention_mask = attention_masks[sub_b:sub_b+self.encoder_gpu_train_limit]
            sub_weight_embedding = checkpoint(self.model, sub_input_ids, sub_attention_mask, dummy_tensor)
            all_weight_embedding.append(sub_weight_embedding)

        whole_weight_embedding = torch.cat(all_weight_embedding, dim=0)

        weight_embedding = []
        for idx, now_weight_embedding in enumerate(whole_weight_embedding):
            begin_ptr = second[idx]
            weight_embedding.append(
                now_weight_embedding[begin_ptr:begin_ptr+self.add_sequence_length].unsqueeze(0)
            )
        weight_embedding = torch.cat(weight_embedding, dim=0)

        new_labels = copy.deepcopy(input_ids)
        new_labels[new_labels == self.tokenizer.eos_token_id] = -100
        new_labels[new_labels > 32000] = -100
        new_labels[:, 0] = -100

        sentence_embedding = self.origin_model.base_model.embed_tokens(input_ids)
        inputs_embeds = torch.cat(
            [sentence_embedding[:, :1], weight_embedding, sentence_embedding[:, 1:]],
            dim=1
        ).cuda()

        negative_ones = -100 * torch.ones(
            (new_labels.shape[0], self.add_sequence_length),
            dtype=input_ids.dtype
        ).cuda()
        new_labels = torch.cat((negative_ones, new_labels), dim=1)

        fir, sec, thi = inputs_embeds.shape
        ct_loss = self.origin_model(
            inputs_embeds=inputs_embeds,
            attention_mask=torch.ones(fir, sec, dtype=torch.int64).cuda(),
            labels=new_labels,
            output_hidden_states=True
        ).loss

        return ct_loss

class Llm2Comp_Second_Stage_Bidirection(nn.Module):
    """Second-stage model for bidirectional training with contrastive loss."""
    def __init__(self, args, local_rank=0, **kwargs):
        super().__init__()
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        self.model = CustomLlamaForCausalLM.from_pretrained(model_name)
        self.origin_model = LlamaForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        for param in self.origin_model.parameters():
            param.requires_grad = False

        lora_config = LoraConfig(
            r=16,
            target_modules=["q_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
            task_type=TaskType.CAUSAL_LM,
            lora_alpha=32,
            lora_dropout=0.05
        )
        lora_model = get_peft_model(self.model, lora_config)
        self.vocab_size = len(self.tokenizer)
        self.model = lora_model
        self.add_sequence_length = args.sequence_length
        self.model.resize_token_embeddings(self.vocab_size + self.add_sequence_length)
        self.origin_model.resize_token_embeddings(self.vocab_size + self.add_sequence_length)

        torch.cuda.set_device(local_rank)
        self.dim = self.model.config.hidden_size
        self.global_step = 0
        self.model_wrapper()

    def freeze(self):
        """Freeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def set_dropout_rate_recursive(self, module, dropout_rate=0.3, parent_name=""):
        """Recursively update dropout rate for all dropout layers."""
        dropout_types = (nn.Dropout, nn.AlphaDropout, nn.Dropout2d, nn.Dropout3d)
        for name, child in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            if isinstance(child, dropout_types):
                child.p = dropout_rate
                print(f"Set Dropout layer '{full_name}' dropout rate to {dropout_rate}")
            else:
                self.set_dropout_rate_recursive(child, dropout_rate, full_name)

    def set_attn_dropout(self, dropout_rate=0.3):
        """Set dropout rate for attention layers."""
        for idx in range(len(self.model.base_model.model.model.model.layers)):
            self.model.base_model.model.model.model.layers[idx].self_attn.attention_dropout = dropout_rate
            print(f"Set attention dropout rate at layer {idx} to {dropout_rate}")

    def model_wrapper(self):
        """Wrap encoder for contrastive training."""
        self.model = EncoderWrapper(self.model)
        self.encoder_gpu_train_limit = 2
        self.temperature = 0.05
        self.scale = 1

    def forward(self, input_ids, embedding_indices):
        fir, sec = input_ids.shape
        first, second = zip(*embedding_indices)

        attention_masks = torch.zeros(fir, sec, dtype=int)
        for idx, site in enumerate(second):
            attention_masks[idx][:site + self.add_sequence_length] = 1

        dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
        all_pooled_output_first = []
        all_pooled_output_second = []

        for sub_b in range(0, input_ids.shape[0], self.encoder_gpu_train_limit):
            sub_input_ids = input_ids[sub_b:sub_b + self.encoder_gpu_train_limit]
            sub_attention_mask = attention_masks[sub_b:sub_b + self.encoder_gpu_train_limit]
            pooler_first, pooler_second = checkpoint(self.model, sub_input_ids, sub_attention_mask, dummy_tensor)
            all_pooled_output_first.append(pooler_first)
            all_pooled_output_second.append(pooler_second)

        whole_weight_embedding_first = torch.cat(all_pooled_output_first, dim=0)
        whole_weight_embedding_second = torch.cat(all_pooled_output_second, dim=0)

        weight_embedding_first = []
        weight_embedding_second = []
        for idx in range(len(whole_weight_embedding_first)):
            begin_ptr = second[idx]
            weight_embedding_first.append(
                whole_weight_embedding_first[idx][begin_ptr:begin_ptr+self.add_sequence_length].unsqueeze(0)
            )
            weight_embedding_second.append(
                whole_weight_embedding_second[idx][begin_ptr:begin_ptr+self.add_sequence_length].unsqueeze(0)
            )
        weight_embedding_first = torch.cat(weight_embedding_first, dim=0)
        weight_embedding_second = torch.cat(weight_embedding_second, dim=0)

        # Mean pooling
        weight_embedding_first = torch.mean(weight_embedding_first, dim=1)
        weight_embedding_second = torch.mean(weight_embedding_second, dim=1)

        # Normalize embeddings
        weight_embedding_first_norm = F.normalize(weight_embedding_first, p=2, dim=-1)
        weight_embedding_second_norm = F.normalize(weight_embedding_second, p=2, dim=-1)

        # Compute cosine similarity
        dot_products = torch.matmul(weight_embedding_first_norm, weight_embedding_second_norm.transpose(0, 1))
        probs = F.log_softmax(dot_products / self.temperature, dim=1)
        targets = torch.arange(probs.size(0)).to(probs.device)
        loss = F.nll_loss(probs, targets) * self.scale

        return loss


class Llm2Comp_Second_Stage_Bidirection_Eval(nn.Module):
    """Evaluation model for the second stage of bidirectional training."""
    def __init__(self, args, local_rank=0, **kwargs):
        super().__init__()
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        self.model = CustomLlamaForCausalLM.from_pretrained(model_name)
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)

        lora_config = LoraConfig(
            r=16,
            target_modules=["q_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
            task_type=TaskType.CAUSAL_LM,
            lora_alpha=32,
            lora_dropout=0.05
        )
        lora_model = get_peft_model(self.model, lora_config)
        self.vocab_size = len(self.tokenizer)
        self.model = lora_model
        self.add_sequence_length = args.sequence_length
        self.model.resize_token_embeddings(self.vocab_size + self.add_sequence_length)
        torch.cuda.set_device(local_rank)
        self.dim = self.model.config.hidden_size
        self.global_step = 0
        self.model.merge_and_unload()
        self.model = get_peft_model(self.model, lora_config)
        self.model_wrapper_new()

    def freeze(self):
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def set_dropout_rate_recursive(self, module, dropout_rate=0.3, parent_name=""):
        """Recursively update dropout rate for all dropout layers."""
        dropout_types = (nn.Dropout, nn.AlphaDropout, nn.Dropout2d, nn.Dropout3d)
        for name, child in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            if isinstance(child, dropout_types):
                child.p = dropout_rate
                print(f"Set Dropout layer '{full_name}' dropout rate to {dropout_rate}")
            else:
                self.set_dropout_rate_recursive(child, dropout_rate, full_name)

    def set_attn_dropout(self, dropout_rate=0.3):
        """Set dropout rate for attention layers."""
        for idx in range(len(self.model.encoder.base_model.model.model.model.layers)):
            self.model.encoder.base_model.model.model.model.layers[idx].self_attn.attention_dropout = dropout_rate
            print(f"Set attention dropout rate at layer {idx} to {dropout_rate}")

    def model_wrapper_new(self):
        """Wrap encoder for evaluation."""
        self.model = EncoderWrapperSupervised(self.model)
        self.encoder_gpu_train_limit = 4
        self.temperature = 0.05
        self.scale = 1

    def forward(self, input_ids, embedding_indices):
        fir, sec = input_ids.shape
        first, second = zip(*embedding_indices)

        attention_masks = torch.zeros(fir, sec, dtype=int)
        for idx, site in enumerate(second):
            attention_masks[idx][:site + self.add_sequence_length] = 1

        dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
        all_pooled_output = []

        for sub_b in range(0, input_ids.shape[0], self.encoder_gpu_train_limit):
            sub_input_ids = input_ids[sub_b:sub_b+self.encoder_gpu_train_limit]
            sub_attention_mask = attention_masks[sub_b:sub_b+self.encoder_gpu_train_limit]
            pooler_output = checkpoint(self.model, sub_input_ids, sub_attention_mask, dummy_tensor)
            all_pooled_output.append(pooler_output)

        whole_weight_embedding = torch.cat(all_pooled_output, dim=0)

        weight_embedding = []
        for idx, now_weight_embedding in enumerate(whole_weight_embedding):
            begin_ptr = second[idx]
            weight_embedding.append(
                now_weight_embedding[begin_ptr:begin_ptr+self.add_sequence_length].unsqueeze(0)
            )
        weight_embedding = torch.cat(weight_embedding, dim=0)
        weight_embedding = torch.mean(weight_embedding, dim=1)

        return weight_embedding

class Llm2Comp_Third_Stage_Bidirection_Eval(nn.Module):
    """Third-stage evaluation model for bidirectional training."""
    def __init__(self, args, local_rank=0, **kwargs):
        super().__init__()
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        self.model = CustomLlamaForCausalLM.from_pretrained(model_name)
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.eos_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id

        lora_config = LoraConfig(
            r=16,
            target_modules=["q_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
            task_type=TaskType.CAUSAL_LM,
            lora_alpha=32,
            lora_dropout=0.05
        )
        lora_model = get_peft_model(self.model, lora_config)
        self.vocab_size = len(self.tokenizer)
        self.model = lora_model
        self.add_sequence_length = args.sequence_length
        self.model.resize_token_embeddings(self.vocab_size + self.add_sequence_length)

        torch.cuda.set_device(local_rank)
        self.dim = self.model.config.hidden_size
        self.global_step = 0
        self.model.merge_and_unload()
        self.model = get_peft_model(self.model, lora_config)
        self.model_wrapper()
        self.model = self.model.encoder
        self.model.merge_and_unload()
        self.model = get_peft_model(self.model, lora_config)
        self.model_wrapper_new()

    def freeze(self):
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def set_dropout_rate_recursive(self, module, dropout_rate=0.3, parent_name=""):
        """Recursively update dropout rate for dropout layers."""
        dropout_types = (nn.Dropout, nn.AlphaDropout, nn.Dropout2d, nn.Dropout3d)
        for name, child in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            if isinstance(child, dropout_types):
                child.p = dropout_rate
                print(f"Set Dropout layer '{full_name}' dropout rate to {dropout_rate}")
            else:
                self.set_dropout_rate_recursive(child, dropout_rate, full_name)

    def set_attn_dropout(self, dropout_rate=0.3):
        """Set attention dropout for all transformer layers."""
        for idx in range(len(self.model.encoder.base_model.model.model.model.model.layers)):
            self.model.encoder.base_model.model.model.model.model.layers[idx].self_attn.attention_dropout = dropout_rate
            print(f"Set attention dropout at layer {idx} to {dropout_rate}")

    def model_wrapper_new(self):
        """Wrap encoder for evaluation with supervised wrapper."""
        self.model = EncoderWrapperSupervised(self.model)
        self.encoder_gpu_train_limit = 2
        self.temperature = 0.05
        self.scale = 1

    def model_wrapper(self):
        """Wrap encoder for evaluation with bidirectional wrapper."""
        self.model = EncoderWrapper(self.model)
        self.encoder_gpu_train_limit = 2
        self.temperature = 0.05
        self.scale = 1

    def forward(self, input_ids, embedding_indices):
        fir, sec = input_ids.shape
        first, second = zip(*embedding_indices)

        attention_masks = torch.zeros(fir, sec, dtype=int)
        for idx, site in enumerate(second):
            attention_masks[idx][:site + self.add_sequence_length] = 1

        dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
        all_pooled_output = []

        for sub_b in range(0, input_ids.shape[0], self.encoder_gpu_train_limit):
            sub_input_ids = input_ids[sub_b:sub_b+self.encoder_gpu_train_limit]
            sub_attention_mask = attention_masks[sub_b:sub_b+self.encoder_gpu_train_limit]
            pooler_output = checkpoint(
                self.model, sub_input_ids, sub_attention_mask, dummy_tensor, use_reentrant=False
            )
            all_pooled_output.append(pooler_output)

        whole_weight_embedding = torch.cat(all_pooled_output, dim=0)

        weight_embedding = []
        for idx, now_weight_embedding in enumerate(whole_weight_embedding):
            begin_ptr = second[idx]
            weight_embedding.append(
                now_weight_embedding[begin_ptr:begin_ptr+self.add_sequence_length].unsqueeze(0)
            )

        weight_embedding = torch.cat(weight_embedding, dim=0)
        weight_embedding = torch.mean(weight_embedding, dim=1)

        return weight_embedding
