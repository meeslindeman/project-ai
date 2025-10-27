import nltk
from nltk.corpus import wordnet as wn

nltk.download('wordnet')

def get_subtrees(roots: list) -> list:
    subtrees = []
    for r in roots:
        descendants = set(r.closure(lambda s: s.hyponyms()))
        descendants.add(r)
        subtrees.append(descendants)
    return subtrees

def assign_labels(subtrees: list) -> dict:
    label_of = {}
    for idx, subtree in enumerate(subtrees):
        for s in subtree:
            if s not in label_of:
                label_of[s] = idx
            else:
                # if it already had a label from another subtree,
                # mark it ambiguous so we can filter later
                label_of[s] = None

    items = {}
    for s, lbl in label_of.items():
        if lbl is not None:
            items[s] = lbl
    return items

def get_data(roots: list, nodes: list, node_ids: dict, labels: dict, MAX_TOKENS: int, PAD_ID: int) -> dict:
    sample_tokens = {}  # node_id -> [MAX_TOKENS] of int ids (including PAD_ID)
    sample_masks = {}   # node_id -> [MAX_TOKENS] of 1/0
    train_nodes = []

    vocab_size = len(nodes) + 1  

    for u_syn in nodes:
        u_id = node_ids[u_syn]

        # gather local context: itself + hypernyms (parents) + hyponyms (children)
        context_syns = [u_syn] + u_syn.hypernyms() + u_syn.hyponyms()

        context_ids = []
        for s in context_syns:
            if s in node_ids:
                context_ids.append(node_ids[s])

        real_len = len(context_ids)
        if real_len >= MAX_TOKENS:
            token_ids = context_ids[:MAX_TOKENS]
            mask = [1] * MAX_TOKENS
        else:
            pad_count = MAX_TOKENS - real_len
            token_ids = context_ids + [PAD_ID] * pad_count
            mask = [1] * real_len + [0] * pad_count
        
        sample_tokens[u_id] = token_ids
        sample_masks[u_id] = mask
        train_nodes.append(u_id)
    
    return {
        "train_nodes": train_nodes,
        "labels": labels,
        "sample_tokens": sample_tokens,
        "sample_masks": sample_masks,
        "pad_id": PAD_ID,
        "num_classes": len(roots),
        "max_tokens": MAX_TOKENS,
        "vocab_size": vocab_size
    }


def build_dataset() -> dict:
    roots = [
        wn.synset('animal.n.01'),
        wn.synset('plant.n.02'),
        wn.synset('person.n.01'),
    ]

    subtrees = get_subtrees(roots)
    items = assign_labels(subtrees)

    nodes = list(items.keys())
    node_ids = {s: i for i, s in enumerate(nodes)}
    labels = {node_ids[s]: lbl for s, lbl in items.items()}

    MAX_TOKENS = 20
    PAD_ID = len(nodes)

    dataset = get_data(
        roots=roots,
        nodes=nodes,
        node_ids=node_ids,
        labels=labels,
        MAX_TOKENS=MAX_TOKENS,
        PAD_ID=PAD_ID,
    )

    return dataset
