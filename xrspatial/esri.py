import pandas as pd
import requests


def featureset_to_dataframe(featureset,
                            convert_geometry=False,
                            use_aliases=False):
    items = [x['attributes'] for x in featureset['features']]
    df = pd.DataFrame(items)
    if use_aliases and featureset.get('fieldAliases'):
        df.rename(columns=featureset['fieldAliases'], inplace=True)
    if convert_geometry:
        pass
    return df


def query_to_dataframe(layer, where, token=None, outFields='*', chunkSize=100,
                       use_aliases=True):
    featureset = query_layer(layer, where, token, outFields, chunkSize)
    return featureset_to_dataframe(featureset, use_aliases=use_aliases)


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def query_layer(layer, where, token=None, outFields='*', chunkSize=100,
                returnGeometry=False):

    url = layer + r'/query'

    params = {}
    params['where'] = where
    params['outFields'] = outFields
    params['returnGeometry'] = returnGeometry
    params['token'] = token
    params['f'] = 'json'
    params['returnIdsOnly'] = True

    ids_req = requests.post(url, data=params)
    ids_req.raise_for_status()
    ids_response = ids_req.json().get('objectIds')
    params['returnIdsOnly'] = False
    params['where'] = ''

    featureset = None
    for ids in chunker(ids_response, chunkSize):
        params['objectIds'] = ','.join(map(str, ids))
        req = requests.post(url, data=params)
        req.raise_for_status()
        feat_response = req.json()
        if not featureset:
            featureset = feat_response
        else:
            featureset['features'] += feat_response['features']
    if not featureset:
        featureset = {}
        featureset['features'] = []

    return featureset
