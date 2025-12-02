# Nicholas DeJesse (ndejesse@u.rochester.edu)
# Diego Rodriguez (drodri41@u.rochester.edu)

import math
from collections import defaultdict

def calculate_bleu(translated, target, n=4):
    """
    Computes the BLEU score for translated tokens vs. target tokens. Inputs need to be lists not tensors.

    Args:
        translated (list): list of tokens that were translated
        target (list): list of target tokens
        n (int): size of n-grams (default = 4)
    """
    # If either translated or target list length is less than n, score is meaningless so we return 0.
    # Current implementation is inefficient, should switch to a version that doesn't use for loops at some point.
    if len(translated) < n or len(target) < n:
        return 0
    brev_pen = 1 if len(translated) > len(target) else math.exp(1-len(target)/len(translated))
    total_precision = 0
    for i in range(1, n+1):
        translated_igrams = defaultdict(int)
        for j in range(len(translated)-i+1):
            translated_igrams[tuple(translated[j:j+i])] += 1

        target_igrams = defaultdict(int)
        for j in range(len(target)-i+1):
            target_igrams[tuple(target[j:j+i])] += 1
        
        clip_count = sum([min(translated_igrams[g], target_igrams[g]) for g in translated_igrams.keys()])
        translated_total = sum(target_igrams.values())

        if clip_count == 0:
            # Log gives -inf so BLEU will be zero
            return 0
        total_precision += math.log(clip_count / translated_total)
    total_precision /= n
        
    return brev_pen * math.exp(total_precision)
        



def main():
    # Small test for BLEU score
    test = ["I", "like", "computer", "science", ".", "It", "is", "my", "favorite", "subject", "."]
    target = ["I", "like", "computer", "science", ".", "It", "is", "my", "favorite", "subject", "!"]

    print(calculate_bleu(test, target))

if __name__ == "__main__":
    main()