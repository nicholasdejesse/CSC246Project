import argparse
import json

# Hyperparameters
min_str_len = 2 # Any string with length less than this will be removed
max_str_len = 400 # Any string with length greater than this will be removed

# Translation table to replace weird characters (credit: https://stackoverflow.com/questions/10294032/python-replace-typographical-quotes-dashes-etc-with-their-ascii-counterparts)
transl_table = dict( [ (ord(x), ord(y)) for x,y in zip( u"‘’´“”–-",  u"'''\"\"--") ] ) 

def get_brit_amer_dict(json_path):
    # Use a dict of British spellings and their American counterparts
    # https://github.com/hyperreality/American-British-English-Translator/blob/master/data/british_spellings.json
    with open(json_path) as f:
        brit_to_amer = json.load(f)
    # Make a few more dictionaries for special cases in case they appear
    brit_to_amer_cap = {k.capitalize(): brit_to_amer[k].capitalize() for k in brit_to_amer.keys()}
    brit_to_amer_upper = {k.upper(): brit_to_amer[k].upper() for k in brit_to_amer.keys()}
    brit_to_amer = brit_to_amer | brit_to_amer_cap | brit_to_amer_upper
    return brit_to_amer

def load_doc(file_name):
    """Reads a large text file and converts to a list where each line is an element."""
    with open(file_name, "r", encoding="utf-8") as f:
        file = f.read()
    return file.strip().splitlines()

def get_clean_rows(l1, l2):
    rows = []
    assert(len(l1) == len(l2)) # Assuming languages are downloaded from parallel corpus and thus are already sentences aligned
    for i in range(len(l1)):
        if should_remove_str(l1[i]) or should_remove_str(l2[i]):
            continue
        rows.append([normalize_str(l1[i], True), normalize_str(l2[i], False)])
    return rows

def should_remove_str(string):
    """Conditions for when a string should be fully removed from training."""
    if len(string) < min_str_len or len(string) > max_str_len:
        return True
    if "http" in string or "www." in string: # Remove links
        return True
    if "@" in string: # Usually problematic (either email or some other weird string)
        return True
    return False

def normalize_str(string, is_english):
    """Updates string to replace problematic characters and convert to American English"""
    bad_start_punc = ".,!?;:)}]-" # Punctuation which should never appear as the first character
    string = string.strip()
    string = string.replace("' s ", "'s") # A lot of the apostrophies are broken
    if string[0] in bad_start_punc:
        string = string[1:]
    string = string.strip().translate(transl_table) # Strip again in case we removed a bad punctuation and now have more whitespace at start

    if is_english:
        brit_amer_dict = get_brit_amer_dict("./british_spellings.json")
        string = " ".join([brit_amer_dict[w] if w in brit_amer_dict else w for w in string.split()])
    return string

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("l1_path")
    parser.add_argument("l2_path")
    parser.add_argument("csv_dest")
    args = parser.parse_args()

    # l1 = load_doc(args.l1_path)
    # l2 = load_doc(args.l2_path)

    # print(f"l1 min: {min(l1, key=len)}")
    # print(f"l1 max: {max(l1, key=len)}")

    print(normalize_str("“This Aeon is also intended to finance a feasibility study on the establishment of a European civil peace corps going beyond humanitarian Aeon.”", True))

    # count = 0
    # for i in range(len(l1)):
    #     if should_remove_str(l1[i]) or should_remove_str(l2[i]):
    #         count += 1
    # print(count)
    # print(count / len(l1))
    # print(l1[0:5])
    # print(l2[0:5])

if __name__ == "__main__":
    main()