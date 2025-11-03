import nltk
from nltk.corpus import wordnet as wn
import random

# nltk.download('wordnet')

SUBTREE_ROOTS = {
    0: 'organism.n.01',      # living things
    1: 'person.n.01',        # people
    2: 'artifact.n.01',      # man-made objects
    3: 'substance.n.01',     # materials, matter
    4: 'food.n.01',          # edibles
    5: 'body_part.n.01',     # anatomy
    6: 'location.n.01',      # places
    7: 'group.n.01',         # collections
    8: 'communication.n.01', # messages, language
    9: 'psychological_feature.n.01'  # cognition, emotion
}

SUBTREE_SYNSETS = {
    class_id: wn.synset(synset_name) 
    for class_id, synset_name in SUBTREE_ROOTS.items()
}

def get_hypernyms(synset: wn.synset) -> list:
    """Get all hypernyms of a given synset."""
    paths = synset.hypernym_paths()
    longest_path = max(paths, key=len) if paths else [synset]

    hypernyms = longest_path[:-1] # Exclude the synset itself
    hypernyms.reverse()

    return hypernyms

def get_hyponyms(synset: wn.synset, max_hyponyms: int = 5) -> list:
    """Get all hyponyms of a given synset."""
    hyponyms = synset.hyponyms()

    if len(hyponyms) > max_hyponyms:
        hyponyms = hyponyms[:max_hyponyms]

    return hyponyms

# def create_sequence(synset: wn.synset, max_hyponyms: int = 5) -> tuple:
#     """Create a sequence of synset name, hypernyms, and hyponyms."""
#     target = [synset]

#     hyponyms = get_hyponyms(synset, max_hyponyms=max_hyponyms)

#     synsets = target + hyponyms
#     names = [syn.lemmas()[0].name() for syn in synsets]

#     return synsets, names

def create_sequence(synset: wn.synset, max_hyponyms=5) -> tuple[list[wn.synset], list[str]]:
    """Target + direct hypernyms + direct hyponyms (1-hop only)."""
    target = [synset]
    
    # Get DIRECT parents only (not full path)
    direct_hypernyms = synset.hypernyms()
    
    # Get direct children
    hyponyms = get_hyponyms(synset, max_hyponyms)
    
    synsets = target + direct_hypernyms + hyponyms
    names = [syn.lemmas()[0].name() for syn in synsets]
    
    return synsets, names

def get_subtree_label(synset: wn.synset) -> int | None:
    """Get the subtree label for a given synset."""
    path = get_hypernyms(synset) 

    matches = []
    for class_id, root_synset in SUBTREE_SYNSETS.items():
        if root_synset in path:
            position = path.index(root_synset)
            matches.append((class_id, position))
    
    if not matches:
        return None
    
    # Return the class_id with the closest match
    best_class_id, _ = min(matches, key=lambda x: x[1])
    return best_class_id

def collect_dataset_synsets() -> list:
    """Collect all synsets that belong to the defined subtrees."""
    nouns = list(wn.all_synsets('n'))

    dataset = []
    no_label = 0

    for synset in nouns:
        label = get_subtree_label(synset)
        if label is not None:
            dataset.append((synset, label))
        else:
            no_label += 1
    
    return dataset

def build_vocab(dataset: list[tuple[wn.synset, int]], max_hyponyms: int = 5) -> dict:
    """Build a vocabulary from the dataset."""
    synsets = set()

    for synset, label in dataset:
        synsets_in_seq, _ = create_sequence(synset, max_hyponyms=max_hyponyms)
        synsets.update(synsets_in_seq)
    
    synset_to_id = {synset: idx + 1 for idx, synset in enumerate(sorted(synsets, key=lambda s: s.name()))}
    synset_to_id['PAD'] = 0 

    id_to_synset = {idx: synset for synset, idx in synset_to_id.items()}

    vocab_size = len(synset_to_id)
    
    return synset_to_id, id_to_synset, vocab_size

def create_example(synset: wn.synset, label: int, synset_to_id: dict[str, int], max_length: int = 50, max_hyponyms: int = 5) -> tuple:
    """Create a single training example."""
    synsets_in_seq, _ = create_sequence(synset, max_hyponyms=max_hyponyms) 
    token_ids = [synset_to_id[syn] for syn in synsets_in_seq]
    
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]
    
    mask = [True] * len(token_ids)
    
    num_pad = max_length - len(token_ids)
    token_ids.extend([synset_to_id['PAD']] * num_pad)
    mask.extend([False] * num_pad)
    
    return token_ids, mask, label

def prepare_dataset(max_length: int = 10, max_hyponyms: int = 5, seed: int = 42) -> dict:
    """Prepare the dataset for training, validation, and testing."""
    random.seed(seed)
    dataset = collect_dataset_synsets()
    synset_to_id, id_to_synset, vocab_size = build_vocab(dataset, max_hyponyms=max_hyponyms)

    dataset_examples = []
    for synset, label in dataset:
        token_ids, mask, label = create_example(synset, label, synset_to_id, max_length, max_hyponyms)
        dataset_examples.append((token_ids, mask, label))
    
    random.shuffle(dataset_examples)

    return {
        'dataset': dataset_examples,
        'vocab_size': vocab_size,
        'num_classes': len(SUBTREE_ROOTS),
        'class_names': SUBTREE_ROOTS
    }

# if __name__ == "__main__":
#     raw_dataset = collect_dataset_synsets()
    
#     from collections import defaultdict
#     examples_by_class = defaultdict(list)
#     for synset, label in raw_dataset:
#         examples_by_class[label].append(synset)

#     for class_id in range(min(1, len(SUBTREE_ROOTS))):  
#         synsets_in_class = examples_by_class[class_id]
        
#         print(f"{'='*60}")
#         print(f"CLASS {class_id}: {SUBTREE_ROOTS[class_id]}")
#         print(f"Total examples in class: {len(synsets_in_class)}")
#         print(f"{'='*60}")
        
#         # Show 2 examples from this class
#         for i in range(min(1, len(synsets_in_class))):
#             synset = synsets_in_class[i]
#             synsets_in_seq, names = create_sequence(synset, max_hyponyms=5)
            
#             print(f"\nExample {i+1}: {synset.name()}")
#             print(f"Sequence length: {len(names)}")
#             print(f"Tokens: {names}")