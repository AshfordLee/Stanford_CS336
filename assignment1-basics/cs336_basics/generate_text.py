from cs336_basics import transformer,tokenizer
import torch

def generate_text_demo():

    tok = tokenizer.from_files(
        vocab_filepath='./../TinyStories_Result/vocab.json',
        merges_filepath='./../TinyStories_Result/merges.txt',
        special_tokens=['<|endoftext|>']
    )

    model = transformer.transformer_lm(
        d_model=512, num_heads=16, d_ff=1344,
        vocab_size=10000, context_length=256,
        num_layers=4, use_rope=True,
        max_seq_len=256, theta=10000.0
    )

    checkpoint = torch.load('checkpoint.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()


    prompt = "Once upon a time, there was a"
    prompt_tokens = tok.encode(prompt)

    generated_ids = transformer.decoding(
        model=model,
        prompt_tokens=prompt_tokens,
        max_tokens=256,
        temperature=0.7,
        top_p=0.9,
        end_token_id=tok.vocab_reverse[b'<|endoftext|>']
    )


    generated_text = tok.decode(generated_ids)
    print(generated_text)


if __name__ == "__main__":
    generate_text_demo()