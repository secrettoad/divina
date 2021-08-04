from .aws_backoff import get_products
import json
from pkg_resources import resource_filename

def ec2_pricing(pricing_client, region_name, filter_params=None):
    products_params = {'ServiceCode': 'AmazonEC2',
                       'Filters': [
                           {'Type': 'TERM_MATCH', 'Field': 'operatingSystem', 'Value': 'Linux'},
                           {'Type': 'TERM_MATCH', 'Field': 'location',
                            'Value': '{}'.format(get_region_name(region_name))},
                           {'Type': 'TERM_MATCH', 'Field': 'capacitystatus', 'Value': 'Used'},
                           {'Type': 'TERM_MATCH', 'Field': 'preInstalledSw', 'Value': 'NA'},
                           {'Type': 'TERM_MATCH', 'Field': 'tenancy', 'Value': 'shared'}
                       ]}
    if filter_params:
        products_params['Filters'] = products_params['Filters'] + filter_params
    while True:
        response = get_products(pricing_client, products_params)
        yield from [i for i in response['PriceList']]
        if 'NextToken' not in response:
            break
        products_params['NextToken'] = response['NextToken']
    return response


def get_region_name(region_code):
    default_region = 'EU (Ireland)'
    endpoint_file = resource_filename('botocore', 'data/endpoints.json')
    try:
        with open(endpoint_file, 'r') as f:
            data = json.load(f)
        return data['partitions'][0]['regions'][region_code]['description']
    except IOError:
        return default_region

def unnest_ec2_price(product):
    od = product['terms']['OnDemand']
    id1 = list(od)[0]
    id2 = list(od[id1]['priceDimensions'])[0]
    return {od[id1]['priceDimensions'][id2]['unit'] + '_USD': od[id1]['priceDimensions'][id2]['pricePerUnit']['USD']}