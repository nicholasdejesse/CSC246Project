# Code for tokenizer

# Currently character level, needs to be "chunks" of characters at some point
class Tokenizer:
    def __init__(self, data):
        """
        Args:
            data (string): The training data used to determine the tokens.
        """
        vocab = sorted(list(set(data)))
        # For now, just maps individual characters to ints
        self.n = len(vocab)
        self.tok_to_int = {t:i for i, t in enumerate(vocab)}
        self.int_to_tok = {i:t for i, t in enumerate(vocab)}

    def encode(self, txt):
        """Takes in a string and encodes it to a list of tokens."""
        return [self.tok_to_int[c] for c in txt]

    def decode(self, ls):
        """Takes in a list of tokens and decodes it to a string."""
        return "".join([self.int_to_tok[i] for i in ls])

# Quick test to make sure everything works
if __name__ == "__main__":
    text = "Wowwy, CSC 246 is an amazing course!"

    tkn = Tokenizer(text)
    print(tkn.encode(text))
    print(tkn.decode(tkn.encode(text)))