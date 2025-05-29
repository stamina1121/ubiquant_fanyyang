import torch
from transformers import PreTrainedTokenizer # type: ignore
from verl import DataProto


class TestUtil:
    @staticmethod
    def print_masked(response: torch.Tensor, mask: torch.Tensor, tokenizer: PreTrainedTokenizer):
        # Decode the original response
        original_text = tokenizer.decode(response, skip_special_tokens=True)
        print("Original Response:")
        print(original_text)

        # Apply the mask
        masked_response = response[mask.bool()]
        print("\nMask Value:")
        print(mask.bool().tolist())
        # Decode the masked response
        masked_text = tokenizer.decode(masked_response, skip_special_tokens=True)
        print("\nMasked Response:")
        print(masked_text)
        
    @staticmethod
    def print_masked_batch(batch: DataProto, tokenizer: PreTrainedTokenizer):
        if 'action_mask' in batch.batch.keys():
            TestUtil.print_masked(batch.batch['responses'][-1], batch.batch['action_mask'][-1], tokenizer)
        else:
            print('There is no action mask here.')