import pandas as pd
import requests


def export_map(service_url, xmin, ymin, xmax, ymax, height=400, width=400):
    '''
    Parameters
    ----------

    Returns
    -------

    Notes
    -----

    '''
    params = {}
    params['bbox'] = '{},{},{},{}'.format(xmin, ymin, xmax, ymax)
    params['size'] = '{},{}'.format(width, height)
    params['format'] = 'png32'
    params['transparent'] = 'true'

    params['f'] = 'json'
    json_response = requests.get(service_url, params=params)

    params['f'] = 'image'
    response = requests.get(service_url, params=params)
    source_img = Image.open(cStringIO.StringIO(response.content))
    return source_img, json_response.json()['extent']



def featureset_to_dataframe(featureset, convert_geometry=False,
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
