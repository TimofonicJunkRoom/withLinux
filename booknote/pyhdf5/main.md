Python and HDF5
===

> Oreilly, Andrew Collete  

# Introduction

```python
import h5py
f = h5py.File("weather.hdf5")
f["/15/temperature"] = temperature
f["/15/temperature"].attrs["dt"] = 10.0
f["/15/temperature"].attrs["start_time"] = 1375204299
f["/15/wind"] = wind
f["/15/wind"].attrs["dt"] = 5.0
f["/20/temperature"] = temperature_from_station_20
```
This example illustrates two of the “killer features” of HDF5:
organization in hierarchical groups and attributes.
```
>>> dataset = f["/15/temperature"]
>>> for key, value in dataset.attrs.iteritems():
...     print "%s: %s" % (key, value)
```

# Getting Started
