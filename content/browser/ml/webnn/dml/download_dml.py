#!/usr/bin/env python3
# Copyright 2022 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

import json
import os
import sys
import zipfile

try:
  from urllib2 import HTTPError, URLError, urlopen
except ImportError: # For Py3 compatibility
  from urllib.error import HTTPError, URLError
  from urllib.request import urlopen

# The code is based on https://github.com/microsoft/DirectML/blob/master/Python/setup.py that uses
# the MIT license (https://github.com/microsoft/DirectML/blob/master/LICENSE).

dml_feed_url = 'https://api.nuget.org/v3/index.json'
dml_resource_id = 'microsoft.ai.directml'
dml_resource_version = '1.8.2'

base_path = os.path.dirname(os.path.realpath(__file__))
dependency_dir = os.path.abspath(os.path.join(
    base_path, os.pardir, os.pardir, os.pardir,
    os.pardir, os.pardir, 'third_party'))

dml_resource_name = '.'.join([dml_resource_id, dml_resource_version])
dml_path = '%s\%s' % (dependency_dir, dml_resource_name)

def get_resource_url(feed_url, resource_id, resource_version):
  index = urlopen(feed_url)
  resources = json.loads(index.read())['resources']

  for resource in resources:
    if resource['@type'] == 'PackageBaseAddress/3.0.0':
      return resource['@id'] + '/'.join([resource_id, resource_version]) + \
             '/' + '.'.join([resource_id, resource_version]) + '.nupkg'

  return ''

def download_nupkg(feed_url, resource_id, resource_version, resource_path):
  if os.path.exists(resource_path):
    return

  url = get_resource_url(feed_url, resource_id, resource_version)
  if not url:
    return

  print('Downloading ' + url + ' to ' + resource_path)
  # download the package
  resource_file = resource_path + '.nupkg'
  with open(resource_file, 'wb') as file:
    result = urlopen(url)
    while True:
      block = result.read(1024)
      if not block:
        break
      file.write(block)

  if os.path.exists(resource_file):
    # nupkg is just a zip, unzip it
    with zipfile.ZipFile(resource_file, "r") as zip_ref:
      zip_ref.extractall(resource_path)
    os.remove(resource_file)

def main():
  download_nupkg(dml_feed_url, dml_resource_id, dml_resource_version, dml_path)

if __name__ == '__main__':
  sys.exit(main())
