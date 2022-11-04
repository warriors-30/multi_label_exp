def getting_data(url,path):
  data = urllib.request.urlopen(url)
  tar_package = tarfile.open(fileobj=data, mode='r:gz')
  tar_package.extractall(path)
  tar_package.close()
  return print("Data extracted and saved.")

getting_data("http://ai.stanford.edu/~jkrause/car196/car_ims.tgz","/content/carimages")

def getting_metadata(url,filename):
  '''
  Downloading a metadata file from a specific url and save it to the disc.
  '''
  labels = urllib2.urlopen(url)
  file = open(filename, 'wb')
  file.write(labels.read())
  file.close()
  return print("Metadata downloaded and saved.")

getting_metadata("http://ai.stanford.edu/~jkrause/car196/cars_annos.mat","car_metadata.mat")