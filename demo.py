import requests
import xmlschema
import os
from lxml import etree
from datetime import datetime
from astropy.time import Time
import pandas as pd
import numpy as np
import re
import random
import glob
import healpy as hp
import ligo.skymap.io
import ligo.skymap.postprocess
import ligo.skymap.bayestar as ligo_bayestar
from scipy.stats import norm, rv_discrete
import time
import hashlib

cfg = {
    "server": {
        "host": "localhost",
        "port": 5000,
        "ssl": False,
    }
}

def api_skyportal(method: str, endpoint: str, data=None, token=None):
    """Make an API call to a SkyPortal instance
    :param method:
    :param endpoint:
    :param data:
    :return:
    """
    method = method.lower()

    if endpoint is None:
        raise ValueError("Endpoint not specified")
    if method not in ["head", "get", "post", "put", "patch", "delete"]:
        raise ValueError(f"Unsupported method: {method}")

    if method == "get":
        response = requests.request(
            method,
            f"{'https' if cfg['server']['ssl'] else 'http'}://"
            f"{cfg['server']['host']}:{cfg['server']['port']}"
            f"{endpoint}",
            params=data,
            headers={"Authorization": f"token {token}"},
        )
    else:
        response = requests.request(
            method,
            f"{'https' if cfg['server']['ssl'] else 'http'}://"
            f"{cfg['server']['host']}:{cfg['server']['port']}"
            f"{endpoint}",
            json=data,
            headers={"Authorization": f"token {token}"},
        )

    return response

def read_notice(payload):

    schema = f'{os.path.dirname(__file__)}/schema/VOEvent-v2.0.xsd'
    voevent_schema = xmlschema.XMLSchema(schema)
    if voevent_schema.is_valid(payload):
        # check if is string
        try:
            payload = payload.encode('ascii')
        except AttributeError:
            pass
        root = etree.fromstring(payload)
    else:
        raise ValueError("xml file is not valid VOEvent")

    return root

def save_notice(root, filename):
    with open(filename, 'wb') as f:
        payload = etree.tostring(root)
        f.write(payload)

def notice_to_date(root, t):
    # set the <Date> tag to current date
    root.find('Who/Date').text = t.datetime.strftime('%Y-%m-%dT%H:%M:%S')
    # set the <ISOTime> tag to current date
    root.find('WhereWhen/ObsDataLocation/ObservationLocation/AstroCoords/Time/TimeInstant/ISOTime').text = t.datetime.strftime('%Y-%m-%dT%H:%M:%S')

    # change the attribute ivorn of the voe:VOEvent tag
    ivorn = root.attrib['ivorn']
    if ivorn is not None:
        # use regex to replace the date
        ivorn = 'ivo://demo/ICARE#' + t.datetime.strftime('%Y-%m-%dT%H:%M:%S')
        root.attrib['ivorn'] = ivorn
    else: 
        raise ValueError("ivorn is None")

    if root.find('What/Param[@name="TrigID"]') is not None:
        # create a random unique number based on the current unix time
        random_number = int(t.unix) + random.randint(0, 1000000)
        # trunc the number to 9 digits
        random_number = random_number % 1000000000

        # in the <What> tag, change the <Param> tag with name="TrigID" to a random number
        
        root.find('What/Param[@name="TrigID"]').attrib['value'] = str(random_number)

    return root

def notice_skymap_path(root, skymap_path):
    # set the <Group name="GW_SKYMAP" type="GW_SKYMAP"> tag to the skymap path
    root.find('What/Group[@name="GW_SKYMAP"]/Param[@name="skymap_fits"]').attrib['value'] = skymap_path

    return root
    

def post_notice(root, token):
    # post the notice to skyportal
    payload = etree.tostring(root)
    payload = str(payload, 'utf-8')
    response = api_skyportal('post', '/api/gcn_event', data={'xml': payload}, token=token)
    if response.status_code == 200:
        print(response.json())
    else:
        print(response.text)

def create_demo_event_point(token):

    filename = f'{os.path.dirname(__file__)}/GRB.xml'
    with open(filename, 'r') as f:
        payload = f.read()
    root = read_notice(payload)
    t = Time(datetime.utcnow(), scale='utc')
    root = notice_to_date(root, t)
    filename = f'{os.path.dirname(__file__)}/GRB_{t.datetime.strftime("%Y%m%dT%H%M%S")}.xml'
    #save_notice(root, filename)
    post_notice(root, token)

    # return the date t in jd
    return (t.jd,
        float(root.find('WhereWhen/ObsDataLocation/ObservationLocation/AstroCoords/Position2D/Value2/C1').text),
        float(root.find('WhereWhen/ObsDataLocation/ObservationLocation/AstroCoords/Position2D/Value2/C2').text),
        float(root.find('WhereWhen/ObsDataLocation/ObservationLocation/AstroCoords/Position2D/Error2Radius').text)
    )

def create_demo_event_fits(token):

    filename = f'{os.path.dirname(__file__)}/GW.xml'
    with open(filename, 'r') as f:
        payload = f.read()
    root = read_notice(payload)

    t = Time(datetime.utcnow(), scale='utc')
    root = notice_to_date(root, t)

    # set the skymap path
    skymap_path = f'{os.path.dirname(__file__)}/GW.fits'
    # root = notice_skymap_path(root, skymap_path)
    # WAS REPLACED BY GITHUB RAW LINK

    filename = f'{os.path.dirname(__file__)}/GW_{t.datetime.strftime("%Y%m%dT%H%M%S")}.xml'
    #save_notice(root, filename)
    post_notice(root, token)

    nside = 512
    order = hp.nside2order(nside)

    skymap = ligo.skymap.io.read_sky_map(skymap_path, moc=True)
    result = ligo_bayestar.rasterize(skymap, order)['PROB']
    prob = hp.reorder(result, 'NESTED', 'RING')

    prob_sort = np.sort(prob)
    prob_sort_cumsum = np.cumsum(prob_sort)
    idx = np.argmin(np.abs(prob_sort_cumsum-0.95))
    prob_thresh = prob_sort[idx]
    prob[prob < prob_thresh] = 0.0
    prob[prob > prob_thresh] = 1.0

    npix = len(prob) 
    nside = hp.npix2nside(npix)

    theta, phi = hp.pix2ang(nside, np.arange(npix)) 
    ra_map = np.rad2deg(phi)
    dec_map = np.rad2deg(0.5*np.pi - theta)

    prob = prob / np.sum(prob) 
    idx = np.where(prob<0)[0] 
    distn = rv_discrete(values=(np.arange(npix), prob))

    return (t.jd, ra_map, dec_map, distn, nside)

def jd_to_iso(jd):
    t = Time(jd, format='jd')
    return t.datetime.strftime('%Y-%m-%dT%H:%M:%S')

def get_public_group(token):
    # get all groups from skyportal
    response = api_skyportal('get', '/api/groups', token=token)
    if response.status_code == 200:
        groups = response.json()['data']
        group_id = [g['id'] for g in groups['user_accessible_groups'] if g['name'] == 'Sitewide Group'][0]
        return group_id
    else:
        print(response.text)
        return None

def create_demo_stream(token):
    # get all streams from skyportal
    response = api_skyportal('get', '/api/streams', token=token)
    if response.status_code == 200:
        streams = response.json()['data']
    else:
        print(response.text)
        return

    stream_id = None
    # check if there is a stream called 'demo', if not, create it
    if 'demo' not in [s['name'] for s in streams]:
        response = api_skyportal('post', '/api/streams', data={
            'name': 'demo',
        }, token=token)
        if response.status_code == 200:
            print(response.json())
            stream_id = response.json()['data']['id']
        else:
            print(response.text)
            return

    # get the stream id
    if stream_id is None:
        for s in streams:
            if s['name'] == 'demo':
                stream_id = s['id']
                break

    return stream_id

def create_demo_filter(stream_id, group_id, token):
    # get all filters from skyportal
    response = api_skyportal('get', '/api/filters', token=token)
    if response.status_code == 200:
        filters = response.json()['data']
    else:
        print(response.text)
        return

    filter_id = None
    # check if there is a filter called 'demo', if not, create it
    if 'demo' not in [f['name'] for f in filters]:
        response = api_skyportal('post', '/api/filters', data={
            'name': 'demo',
            'stream_id': stream_id,
            'group_id': group_id,
        }, token=token)
        if response.status_code == 200:
            filter_id = response.json()['data']['id']
        else:
            print(response.text)
    else:
        filter_id = [f['id'] for f in filters if f['name'] == 'demo'][0]

    return filter_id

def get_ztf_instrument_id(token):
    # get all instruments from skyportal
    response = api_skyportal('get', '/api/instrument', token=token)
    if response.status_code == 200:
        instruments = response.json()['data']
    else:
        print(response.text)
        return

    instrument_id = [i['id'] for i in instruments if (i['name'] == 'ZTF' or i['name'] == 'CFH12k')][0]

    return instrument_id
    

def create_demo_candidates(token, ra, dec, error_radius, jd, nb_obj=100, start_index=0):
    # here, we want to open the csv files contained in the candidates_gw directory
    # in this directory, one can find * .csv files, each containing the candidates for a given object
    # the name of the object is contained in the filename, which is in the format: lc_<object_name>_forced1_stacked0.csv
    # we want to open each file, and create a dataframe with the columns of the csv files + a column with the object name

    # we want to create a candidate for each entry in the dataframe

    # first, we grab the list of files in the directory
    files = sorted(glob.glob('candidates/partnership/*/*.csv'))
    # we loop over the files
    df = pd.DataFrame()
    for file in files:
        # we open the file
        with open(file, 'r') as f:
            # we read the csv file
            df_tmp = pd.read_csv(f)
            # we extract the object name from the filename
            object_name = file.split('_')[1]
            # we add a column with the object name
            df_tmp['object_name'] = object_name
            first_det = np.where(df_tmp.mag < 99)[0]
            if len(first_det) == 0:
                continue
            # we add the dataframe to the main dataframe
            df = df.append(df_tmp, ignore_index=True)

    # remove duplicates, that is entries with the same object name and the same jd
    df = df.drop_duplicates(subset=['object_name', 'jd'])

    # sort the dataframe by object_name
    df = df.sort_values(by=['object_name'])

    # we keep only the first nb_candidates entries, based on the object name and starting at start_index
    df = df[df['object_name'].isin(df['object_name'].unique()[start_index:start_index+nb_obj])]
    print(df.shape)

    # find the earliest jd in the dataframe
    jd_min = df['jd'].min()

    # offset the jd by the jd_min + jd passed as a parameter
    df['jd'] = df['jd'] - jd_min + jd

    group_id = get_public_group(token)
    stream_id = create_demo_stream(token)
    filter_id = create_demo_filter(stream_id, group_id, token)
    instrument_id = get_ztf_instrument_id(token)

    # we create a dataframe with the object names and their ra, dec
    # the ra dec for each object is randomly generated, but within a circle of radius error_radius around the ra, dec given as input
    df_object_names = pd.DataFrame()
    df_object_names['object_name'] = df['object_name'].unique()
    df_object_names['ra'] = np.random.uniform(ra - error_radius, ra + error_radius, len(df_object_names))
    df_object_names['dec'] = np.random.uniform(dec - error_radius, dec + error_radius, len(df_object_names))

    for object_name in df_object_names['object_name']:
        first_det = np.where(df[df['object_name'] == object_name].mag < 99)[0]
        first_det_jd = df[df['object_name'] == object_name].iloc[first_det[0]]['jd']
        df.loc[df['object_name'] == object_name, 'jd'] = df.loc[df['object_name'] == object_name, 'jd'] - first_det_jd + jd + random.uniform(-1, 1)

    # we generate new object names, to avoid conflicts with existing objects
    # it will consist of the obj name + the decimals of the jd 
    df_new_object_names = pd.DataFrame()
    df_new_object_names['object_name'] = df['object_name'].unique()
    df_new_object_names['new_object_name'] = df['object_name'].unique() + '_' + str(jd).split('.')[1]

    # we loop over the entries in the dataframe
    for index, row in df.iterrows():
        # we create a candidate
        data={
            'id': df_new_object_names[df_new_object_names['object_name'] == row['object_name']]['new_object_name'].values[0],
            'ra': df_object_names[df_object_names['object_name'] == row['object_name']]['ra'].values[0],
            'dec': df_object_names[df_object_names['object_name'] == row['object_name']]['dec'].values[0],
            'passed_at': jd_to_iso(row['jd']),
            'filter_ids': [filter_id],
        }
        response = api_skyportal('post', '/api/candidates', data=data, token=token)
        if response.status_code == 200:
            print(response.json())
        else:
            print(response.text)

        # now we create a photometry point
        data = {
            'obj_id': df_new_object_names[df_new_object_names['object_name'] == row['object_name']]['new_object_name'].values[0],
            'instrument_id': instrument_id,
            'mjd': row['jd'] - 2400000.5,
            'filter': 'ztf' + row['filter'],
            'ra': df_object_names[df_object_names['object_name'] == row['object_name']]['ra'].values[0],
            'dec': df_object_names[df_object_names['object_name'] == row['object_name']]['dec'].values[0],
            'origin': 'demo',
        }
        if row['mag'] < 99:
            data['mag'] = row['mag']
            data['magerr'] = row['mag_unc']
            data['magsys'] = 'ab'
            data['limiting_mag'] = row['limmag']
        else:
            data['limiting_mag'] = row['limmag']
            data['magsys'] = 'ab'

        
        response = api_skyportal('post', '/api/photometry', data=data, token=token)
        if response.status_code == 200:
            print(response.json())
        else:
            print(response.text)


def create_demo_candidates_fits(token, ra, dec, distn, nside, jd, nb_obj=100, start_index=0):
    # here, we want to open the csv files contained in the candidates_gw directory
    # in this directory, one can find * .csv files, each containing the candidates for a given object
    # the name of the object is contained in the filename, which is in the format: lc_<object_name>_forced1_stacked0.csv
    # we want to open each file, and create a dataframe with the columns of the csv files + a column with the object name

    # we want to create a candidate for each entry in the dataframe

    # first, we grab the list of files in the directory
    files = sorted(glob.glob('candidates/partnership/*/*.csv'))
    # we loop over the files
    df = pd.DataFrame()
    for file in files:
        # we open the file
        with open(file, 'r') as f:
            # we read the csv file
            df_tmp = pd.read_csv(f)
            # we extract the object name from the filename
            object_name = file.split('_')[1]
            # we add a column with the object name
            df_tmp['object_name'] = object_name
            first_det = np.where(df_tmp.mag < 99)[0]
            if len(first_det) == 0:
                continue
            # we add the dataframe to the main dataframe
            df = df.append(df_tmp, ignore_index=True)

    # remove duplicates, that is entries with the same object name and the same jd
    print(df.shape)
    df = df.drop_duplicates(subset=['object_name', 'jd'])
    print(df.shape)

    # sort the dataframe by object_name
    df = df.sort_values(by=['object_name'])

    # we keep only the first nb_candidates entries, based on the object name and starting at start_index
    df = df[df['object_name'].isin(df['object_name'].unique()[start_index:start_index+nb_obj])]
    print(df.shape)

    group_id = get_public_group(token)
    stream_id = create_demo_stream(token)
    filter_id = create_demo_filter(stream_id, group_id, token)
    instrument_id = get_ztf_instrument_id(token)

    # we create a dataframe with the object names and their ra, dec
    # the ra dec for each object is randomly generated, but within a circle of radius error_radius around the ra, dec given as input
    df_object_names = pd.DataFrame()
    df_object_names['object_name'] = df['object_name'].unique()

    for index, row in df_object_names.iterrows():
        ipix = distn.rvs(size=1)
        ra, dec = hp.pix2ang(nside, ipix, lonlat=True)
        ra = ra[0]
        dec = dec[0]
        df_object_names.loc[index, 'ra'] = ra
        df_object_names.loc[index, 'dec'] = dec

    for object_name in df_object_names['object_name']:
        first_det = np.where(df[df['object_name'] == object_name].mag < 99)[0]
        first_det_jd = df[df['object_name'] == object_name].iloc[first_det[0]]['jd']
        df.loc[df['object_name'] == object_name, 'jd'] = df.loc[df['object_name'] == object_name, 'jd'] - first_det_jd + jd + random.uniform(-1, 1)

    # we generate new object names, to avoid conflicts with existing objects
    # it will consist of the obj name + the decimals of the jd 
    df_new_object_names = pd.DataFrame()
    df_new_object_names['object_name'] = df['object_name'].unique()
    df_new_object_names['new_object_name'] = df['object_name'].unique() + '_' + str(jd).split('.')[1]

    # we loop over the entries in the dataframe
    for index, row in df.iterrows():
    
        # we create a candidate
        data={
            'id': df_new_object_names[df_new_object_names['object_name'] == row['object_name']]['new_object_name'].values[0],
            'ra': df_object_names[df_object_names['object_name'] == row['object_name']]['ra'].values[0],
            'dec': df_object_names[df_object_names['object_name'] == row['object_name']]['dec'].values[0],
            'passed_at': jd_to_iso(row['jd']),
            'filter_ids': [filter_id],
        }
        response = api_skyportal('post', '/api/candidates', data=data, token=token)
        if response.status_code == 200:
            print(response.json())
        else:
            print(response.text)

        # now we create a photometry point
        data = {
            'obj_id': df_new_object_names[df_new_object_names['object_name'] == row['object_name']]['new_object_name'].values[0],
            'instrument_id': instrument_id,
            'mjd': row['jd'] - 2400000.5,
            'filter': 'ztf' + row['filter'],
            'ra': df_object_names[df_object_names['object_name'] == row['object_name']]['ra'].values[0],
            'dec': df_object_names[df_object_names['object_name'] == row['object_name']]['dec'].values[0],
            'origin': 'demo',
        }
        if row['mag'] < 99:
            data['mag'] = row['mag']
            data['magerr'] = row['mag_unc']
            data['magsys'] = 'ab'
            data['limiting_mag'] = row['limmag']
        else:
            data['limiting_mag'] = row['limmag']
            data['magsys'] = 'ab'

        
        response = api_skyportal('post', '/api/photometry', data=data, token=token)
        if response.status_code == 200:
            print(response.json())
        else:
            print(response.text)

def get_ztf_instrument_and_telescope_name(token):
    # get all instruments from skyportal
    response = api_skyportal('get', '/api/instrument', token=token)
    if response.status_code == 200:
        instruments = response.json()['data']
    else:
        print(response.text)
        return

    instrument = [i for i in instruments if (i['name'] == 'ZTF' or i['name'] == 'CFH12k')][0]
    return instrument['name'], instrument['telescope']['name']

def post_observations(token, start_time):
    df = pd.read_csv('observations.csv')
    df = df[['observation_id', 'obstime', 'exposure_time', 'filt', 'seeing', 'limmag', 'field_id', 'ra', 'dec']]
    df.rename(columns={'ra': 'RA', 'dec': 'Dec', 'filt': 'filter'}, inplace=True)

    df['obstime'] = df['obstime'].astype(str)
    min_obstime = Time(df['obstime'].min(), format='isot', scale='utc').jd

    for i in range(len(df['obstime'])):
        df['obstime'][i] = Time(df['obstime'][i], format='isot', scale='utc').jd

    df['obstime'] = df['obstime'] - min_obstime + start_time

    for i in range(len(df['obstime'])):
        df['obstime'][i] = Time(df['obstime'][i], format='jd', scale='utc').isot
        

    # we generate new observation ids, to avoid conflicts with existing observations, which is obs id + the decimal part of the start time (jd)
    df['observation_id'] = df['observation_id'].astype(str) + str(start_time).split('.')[1]

    print(f"Posting {len(df)} observations to SkyPortal...")

    df_dict = df.to_dict(orient='list')

    instrument_name, telescope_name = get_ztf_instrument_and_telescope_name(token)

    data = {
        'instrumentName': instrument_name,
        'telescopeName': telescope_name,
        'observationData': df_dict,
    }
    response = api_skyportal('post', '/api/observation', data=data, token=token)
    if response.status_code == 200:
        print(response.json())
    else:
        print(response.text)

def main():
    # measure time to create the demo
    start_time = time.time()
    token = '31792488-c4f3-4543-bd1f-0260e4cf8672'
    jd1, ra1, dec1, distn1, nside1 = create_demo_event_fits(token) # GW
    time.sleep(5)

    jd2, ra2, dec2, error_radius2 = create_demo_event_point(token) # GRB
    step_1 = time.time()
    print(f'Creating the events took {step_1 - start_time} seconds')

    create_demo_candidates_fits(token, ra1, dec1, distn1, nside1, jd1, nb_obj=100, start_index=0) # CANDIDATES IN GW
    step_2 = time.time()
    print(f'Creating the candidates in the GW took {step_2 - step_1} seconds')

    create_demo_candidates(token, ra2, dec2, error_radius2, jd2, nb_obj=100, start_index=100) # CANDIDATES IN GRB
    step_3 = time.time()
    print(f'Creating the candidates in the GRB took {step_3 - step_2} seconds')

    post_observations(token, jd1)
    step_4 = time.time()
    print(f'Creating the observations took {step_4 - step_3} seconds')

    print(f'Creating the demo took {step_4 - start_time} seconds')

if __name__ == '__main__':
    main()