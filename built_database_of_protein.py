#!/usr/bin/env python
# _*_ coding:utf-8 _*_
"""
This example client takes a PDB file, sends it to the REST service, which
creates HSSP data. The HSSP data is then output to the console.

Example:

    python pdb_to_hssp.py 1crn.pdb http://www.cmbi.umcn.nl/xssp/
"""

import argparse
import json
import requests
import time
import data_PDB_DSSP
import sqlite3 as sql


def pdb_to_hssp(PDB_id, rest_url, num_retries = 30):
    # Read the pdb file data into a variable
    # files = {'file_': open(pdb_file_path, 'rb')}

    # Send a request to the server to create hssp data from the pdb file data.
    # If an error occurs, an exception is raised and the program exits. If the
    # request is successful, the id of the job running on the server is
    # returned.

    url_create = '{}api/create/pdb_id/dssp/'.format(rest_url)
    try:
        r = requests.post(url_create, data ={'data': PDB_id})
        r.raise_for_status()

        job_id = json.loads(r.text)['id']
        print ("Job submitted successfully. Id is: '{}'".format(job_id))
    except requests.HTTPError as e:
        print("retry time :",num_retries)
        if num_retries > 0:
            return pdb_to_hssp(PDB_id, rest_url, num_retries - 1)
    except:
        return


    # Loop until the job running on the server has finished, either successfully
    # or due to an error.
    ready = False
    while not ready:
        # Check the status of the running job. If an error occurs an exception
        # is raised and the program exits. If the request is successful, the
        # status is returned.
        url_status = '{}api/status/pdb_file/hssp_hssp/{}/'.format(rest_url,
                                                                  job_id)
        try:
            r = requests.get(url_status)
            r.raise_for_status()

            status = json.loads(r.text)['status']
            print ("Job status is: '{}'".format(status))
        except requests.HTTPError as e:
            print("retry time :", num_retries)
            if num_retries > 0:
                return pdb_to_hssp(PDB_id, rest_url, num_retries - 1)
        except:
            return

        # If the status equals SUCCESS, exit out of the loop by changing the
        # condition ready. This causes the code to drop into the `else` block
        # below.
        #
        # If the status equals either FAILURE or REVOKED, an exception is raised
        # containing the error message. The program exits.
        #
        # Otherwise, wait for five seconds and start at the beginning of the
        # loop again.
        if status == 'SUCCESS':
            ready = True
        elif status in ['FAILURE', 'REVOKED']:
            raise Exception(json.loads(r.text)['message'])
        else:
            time.sleep(5)
    else:
        # Requests the result of the job. If an error occurs an exception is
        # raised and the program exits. If the request is successful, the result
        # is returned.
        url_result = '{}api/result/pdb_file/hssp_hssp/{}/'.format(rest_url,
                                                                  job_id)
        try:
            r = requests.get(url_result)
            r.raise_for_status()
            result = json.loads(r.text)['result']
        except requests.HTTPError as e:
            print("retry time :", num_retries)
            if num_retries > 0:
                return pdb_to_hssp(PDB_id, rest_url, num_retries - 1)
        except:
            return

        # Return the result to the caller, which prints it to the screen.
        return result

def read_protein_feature(PDB_id, result, c):
    with open('dssp','w') as f:
        f.write(result)
    with open('dssp', 'r') as f:
        result = f.readline()
        line = 0
        while result:
            if line > 27:
                seq = result[13:14]
                sturct = result[16]
                TCO = (float(result[86:91]))
                KAPPA = (float(result[92:97]))
                ALPHA = (float(result[98:103]))
                PHI = (float(result[104:109]))
                PSI = (float(result[110:115]))
                X_CA = (float(result[118:122]))
                Y_CA = (float(result[125:129]))
                Z_CA = (float(result[132:136]))
                data = {'PDB_id':PDB_id, 'seq':seq, 'sturct':sturct, 'TCO':TCO, 'KAPPA':KAPPA, 'ALPHA':ALPHA, 'PHI':PHI, 'PSI':PSI, 'X_CA':X_CA, 'Y_CA':Y_CA, 'Z_CA':Z_CA}
                c.execute(" INSERT INTO PDB_DSSP VALUES ('%s', '%s', '%s', '%f', '%f', '%f','%f','%f','%f','%f','%f')"
                          % (data['PDB_id'], data['seq'], data['sturct'], data['TCO'], data['KAPPA'], data['ALPHA'],
                             data['PHI'],
                             data['PSI'], data['X_CA'], data['Y_CA'], data['Z_CA']))
                conn.commit()
            result = f.readline()
            line += 1

    # data_PDB_DSSP.write_in_PDB_DSSP_table(data, c)

def select_PDBid():
    conn = sql.connect('protein_database.db')
    c = conn.cursor()
    cursor = c.execute('SELECT PDB_id FROM PDB_id')
    for row in cursor:
        row = row[0]
        return row[0:4]

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Create HSSP from a PDB file')
    rest_url = 'http://www.cmbi.umcn.nl/xssp/'

    conn = sql.connect('protein_database.db')
    c = conn.cursor()
    cursor = c.execute('SELECT PDB_id FROM PDB_id')
    count = 0
    line = []
    for row in cursor:
        row = row[0]
        line.append(row[0:4])
        count += 1
    # print(count)
    # print(line[0])
    for i in range(5727, count):#4V4M 第5436个，NO FOUND
        PDB_id = line[i]
        print('READING:',PDB_id, '   ', i,'/6626')
        result = pdb_to_hssp(PDB_id, rest_url)
        read_protein_feature(PDB_id, result, c)
    print('READ FROM DSSP SUCCESSFUL!')
    # conn.commit()
    conn.close()