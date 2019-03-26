import torch
from collections import defaultdict, OrderedDict
from itertools import count
import numpy as np

from sst.util import get_minibatch, prepare_minibatch


def get_histogram_counts(z=None, mask=None, mb=None):
    counts = np.zeros(5).astype(np.int64)

    for i, ex in enumerate(mb):

        tokens = ex.tokens
        token_labels = ex.token_labels

        if z is not None:
            ex_z = z[i][:len(tokens)]

        if mask is not None:
            assert mask[i].sum() == len(tokens), "mismatch mask/tokens"

        for j, tok, lab in zip(count(), tokens, token_labels):
            if z is not None:
                if ex_z[j] > 0:
                    counts[lab] += 1
            else:
                counts[lab] += 1

    return counts


def decorate_token(t, z_):
    """Make selected text boldface in Markdown format"""
    dec = "**" if z_ == 1 else "" 
    return dec + t + dec

def evaluate(model, data, batch_size=25, device=None, iter_i=0, cfg=None):
    """Accuracy of a model on given data set (using minibatches)"""

    model.eval()  # disable dropout

    # z statistics
    totals = OrderedDict()
    z_totals = defaultdict(float)
    histogram_totals = np.zeros(5).astype(np.int64)
    z_histogram_totals = np.zeros(5).astype(np.int64)

    rationales = []
    for mb in get_minibatch(data, batch_size=batch_size, shuffle=False):
        x, targets, reverse_map = prepare_minibatch(mb, model.vocab, device=device)
        mask = (x != 1)
        batch_size = targets.size(0)
        with torch.no_grad():

            py, qz, z = model(x)
            predictions = model.predict(py)

            loss, terms = model.get_loss(
                y=targets,
                py=py, 
                qz=qz,
                z=z,                
                mask=mask,
                iter_i=iter_i,)
                    
            totals['loss'] = totals.get('loss', 0.) + loss.item() * batch_size
            
            for k, v in terms.items():
                totals[k] = totals.get(k, 0.) + v * batch_size
                
            # reverse sort 
            z = z.cpu().numpy()             
            z = z[reverse_map]             
            for idx in range(batch_size):  # iterate over instances in a mini batch
                example = []
                for ti, zi in zip(mb[idx].tokens, z[idx]):  # iterate over tokens in an instance
                    example.append(decorate_token(ti, zi))
                rationales.append((example, predictions[idx]))

        # add the number of correct predictions to the total correct
        totals['acc'] = totals.get('acc', 0.) + (predictions == targets.view(-1)).sum().item()
        totals['total'] = totals.get('total', 0.) + batch_size

    result = OrderedDict()

    # loss, accuracy, optional items
    totals['total'] += 1e-9
    for k, v in totals.items():
        if k != "total":
            result[k] = v / totals["total"]

    # z scores
    z_totals['total'] += 1e-9
    for k, v in z_totals.items():
        if k != "total":
            result[k] = v / z_totals["total"]

    return result, rationales
